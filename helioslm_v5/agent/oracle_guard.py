"""OracleGuard — the self-grading invariant as a mechanism (v5.51, M1).

Motivation: an agent that can WRITE its grader or READ its answer key
can self-grade. This is not hypothetical — it is the failure class
behind the openai/math repository's day-one retractions (unverified
artifacts shipped as results), and the agent-ified version of "never
let the model modify the test that grades it".

tools.py sandboxing contains the agent inside a directory. It does NOT
distinguish ordinary workspace files from oracle files — answer keys,
graders, scoring configs — which may legitimately live INSIDE the
sandbox (a task file that happens to contain the answer) or BESIDE it
(a key file the harness reads after the run). OracleGuard wraps the
tool impls and makes the distinction explicit:

- any guarded op (default: file_read AND file_write) whose resolved
  target is a protected path, or lives inside a protected directory,
  raises ToolCallError loudly — the agent sees a TOOL_ERROR, the run
  continues, nothing is silently corrupted;
- every blocked attempt is appended to `guard.audit` with the raw and
  resolved path — the attempt itself becomes reviewable evidence;
- allowed ops are NOT logged (the audit answers "who attacked the
  oracle", not "what did the agent do" — trajectories already do that).

Path resolution uses os.path.realpath (symlinks included) and
Path.relative_to containment — never string prefixes (see tools.py's
v5.51 fix for why the prefix check is not a containment check).

Design honesty notes:
- This is a MECHANISM, not a detector: it does not try to recognize
  malicious intent, it makes a declared set of files unreachable.
  Files not declared protected are not protected.
- The guard wraps impls, not the model: an agent that never touches
  the filesystem tools is unaffected; a gate (TrustGate et al.) still
  governs routing independently.
"""
import os
from pathlib import Path

try:
    from .schema import ToolCallError
except ImportError:
    from schema import ToolCallError

# ops whose first positional argument is a filesystem path
_GUARDABLE_OPS = ("file_read", "file_write")


class OracleGuard:
    def __init__(self, protected, sandbox_root, ops=_GUARDABLE_OPS):
        """protected: iterable of file/dir paths (oracle files).
        sandbox_root: the root the wrapped impls resolve relative
        paths against (must match build_default_registry's root).
        ops: which impls to wrap."""
        self.protected = tuple(self._resolve(p) for p in protected)
        if not self.protected:
            raise ValueError("OracleGuard: empty protected set — a guard "
                             "that protects nothing is theatre")
        self.sandbox_root = Path(os.path.realpath(sandbox_root))
        self.ops = tuple(ops)
        for op in self.ops:
            if op not in _GUARDABLE_OPS:
                raise ValueError(f"OracleGuard: {op!r} has no path-first "
                                 f"signature in _GUARDABLE_OPS")
        self.audit = []

    @staticmethod
    def _resolve(p) -> Path:
        return Path(os.path.realpath(str(p)))

    def _resolve_arg(self, path) -> Path:
        """Resolve a tool path argument the way the impl will: absolute
        paths stay absolute, relative paths join the sandbox root."""
        path = str(path)
        if os.path.isabs(path):
            return self._resolve(path)
        return self._resolve(self.sandbox_root / path)

    def _protector(self, resolved: Path):
        """Return the protected entry that covers `resolved`, or None."""
        for q in self.protected:
            if resolved == q:
                return q
            try:
                resolved.relative_to(q)  # inside a protected directory
                return q
            except ValueError:
                continue
        return None

    def check(self, op: str, path) -> None:
        """Raise ToolCallError (and audit) iff `path` hits a protected
        file. Exported for callers that want to check without wrapping."""
        resolved = self._resolve_arg(path)
        hit = self._protector(resolved)
        if hit is not None:
            self.audit.append({"op": op, "path": str(path),
                               "resolved": str(resolved),
                               "protected_by": str(hit),
                               "verdict": "BLOCKED"})
            raise ToolCallError(
                f"{op}: '{path}' is a protected oracle file "
                f"(covered by {hit}) — attempt audited")

    def wrap(self, impls: dict) -> dict:
        """Return a NEW impls dict with the guard applied; the input is
        not mutated."""
        wrapped = dict(impls)
        for op in self.ops:
            if op not in impls:
                continue  # impls without this op need no wrapping
            inner = impls[op]

            def _make(inner, op):
                def guarded(path, *args, **kwargs):
                    self.check(op, path)
                    return inner(path, *args, **kwargs)
                guarded.__name__ = f"oracle_guarded_{op}"
                return guarded
            wrapped[op] = _make(inner, op)
        return wrapped
