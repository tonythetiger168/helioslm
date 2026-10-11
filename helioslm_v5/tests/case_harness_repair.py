"""HeliosLM v5 test cases — harness repair locks + import smoke.

Split out of the monolithic test_v5.py layout on 2026-10-11; the
test_v5.py runner imports every case module and executes the tests in
the historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.

Theme (merged from main's v5.48 limitations round, 2026-10-10):
regression locks for bugs that reached main as silent failures —
MemoryPlugin's falsy-store trap, SQLitePlugin's per-call-connect
":memory:" vanishing act (and the Windows file-lock flip side), the
FilesystemPlugin "../" escape — plus the import-smoke line of defense
that imports EVERY module in the tree so an import-time failure fails
the same-day suite instead of shipping.
"""

import sys
from pathlib import Path

from helioslm_v5.tests.case_common import _REPO_ROOT, _pass


# ----------------------------------------------------------------------
# limitations round 2026-10-10: import-smoke line of defense (daily-report
# candidate 2026-10-09 #2). v5.47 repaired decision_head.py, which had
# reached main as a SyntaxError-on-import; v5.48 repaired harness_plugins
# modules that died at import on Windows. No test imported those modules,
# so the breakage shipped. This test imports EVERY module in the tree so
# an import-time failure (syntax, missing sibling, bad path) fails the
# same-day suite instead of reaching main.
#
# Two tiers, matching the two real consumption contracts:
#   Tier 1 — dotted namespace-package imports over helioslm_v5/ (src,
#     configs, ...) and harness/. The tree has NO __init__.py outside
#     agent/envs, so pkgutil.walk_packages silently skips every
#     subpackage — that silence is exactly how a broken module hid. The
#     scan is therefore filesystem-based (os.walk).
#   Tier 2 — BARE top-level imports of helioslm_v5/agent/** (the modules
#     use bare sibling imports like `from decision import ...` and rely
#     on harness_plugins inserting the agent dir into sys.path).
#     harness_plugins is imported first so its v5.47-fixed path
#     insertion runs before any bare import; if it fails, the bare tier
#     is skipped so the root cause is not drowned in secondary
#     ModuleNotFoundErrors.
# ----------------------------------------------------------------------


def test_import_smoke():
    import importlib
    import os

    failures = []
    seen = []

    def _try(mod, label=None):
        try:
            importlib.import_module(mod)
            seen.append(label or mod)
        except Exception as exc:
            failures.append(f"{label or mod}: {type(exc).__name__}: {exc}")

    # Tier 1: dotted imports over the namespace tree.
    skip_dirs = {"__pycache__", "tests", "agent"}
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT / "helioslm_v5"):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn[:-3]),
                                  _REPO_ROOT)
            _try(rel.replace(os.sep, "."))
    # harness/ (namespace package; harness_plugins carries the v5.47-fixed
    # agent-path insertion — importing it exercises exactly that logic).
    for fn in ("harness_core", "harness_plugins"):
        _try(f"harness.{fn}")

    # Tier 2: bare agent imports, only once the path-inserting module
    # itself imported cleanly.
    if "harness.harness_plugins" in seen:
        agent_dir = _REPO_ROOT / "helioslm_v5" / "agent"
        for dirpath, dirnames, filenames in os.walk(agent_dir):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if not fn.endswith(".py") or fn == "__init__.py":
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn[:-3]),
                                      agent_dir)
                dotted = rel.replace(os.sep, ".")
                _try(dotted, label=f"bare:{dotted}")

    # Regression anchors (the modules that historically broke at import):
    # decision_head (v5.47 SyntaxError incident) and harness_plugins
    # (v5.48 Windows path repair) must both be discovered and importable.
    assert "helioslm_v5.src.model_v5" in seen, (
        "filesystem scan found no src modules — wrong root?")
    assert "harness.harness_plugins" in seen
    assert "bare:decision_head" in seen
    # Floor: 66 modules measured on 2026-10-10 (44 dotted + harness 2 +
    # 20 bare agent). A much lower count means the scan silently shrank
    # (e.g. a moved directory) rather than modules being deleted on
    # purpose — investigate before lowering.
    assert len(seen) >= 50, (
        f"only {len(seen)} modules imported — the scan is not covering "
        "the tree it claims to cover")
    assert not failures, (
        f"{len(failures)} module(s) fail at import time:\n"
        + "\n".join(failures[:20]))
    _pass("test_import_smoke",
          f"{len(seen)} modules import cleanly (dotted tree + harness + "
          "bare agent tier), anchors decision_head/harness_plugins ok")


# ----------------------------------------------------------------------
# v5.48 harness plugin repair regression locks (memory falsy-store trap,
# sqlite :memory: connection policy, filesystem path escape)
# ----------------------------------------------------------------------
def test_harness_memory_sqlite_fs_repair():
    """v5.48: helios-harness plugin repair regression locks. Each lock
    pins one bug that reached main as a silent failure:

    (a) MemoryPlugin falsy-store trap: `if store:` is always False for a
        fresh ExperienceStore (it defines __len__, so empty == falsy —
        think.py's own T25 note documents the trap). remember/recall
        no-op'd silently. Locked: remember then recall round-trips.
    (b) SQLitePlugin :memory: persistence + tuple-row contract: per-call
        connect() gave every statement a fresh empty db ("no such
        table"); a lifetime connection fixed that but held a Windows
        file lock on file dbs (tempfile cleanup died with WinError 32).
        Locked: create->insert->query returns tuple rows + columns, and
        execute() reports affected rows; SELECT via execute() is a loud
        error dict, not a silent run.
    (c) FilesystemPlugin escape: the raw `full.startswith(root)` check
        never fired for "../x" (the joined string still starts with the
        root prefix). Locked: reads and writes outside the root are
        rejected with "blocked".
    """
    _agent_dir = str(Path(__file__).resolve().parent.parent / "agent")
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from harness.harness_core import Context
    from harness.harness_plugins import (FilesystemPlugin, MemoryPlugin,
                                         SQLitePlugin)

    # (a) memory round-trip through a fresh (therefore empty/falsy) store
    ctx = Context()
    ctx.use(MemoryPlugin())
    ctx.memory.remember("k1", "thought one", "ok")
    recalled = ctx.memory.recall("k1")
    assert recalled is not None, \
        "memory recall returned None: falsy-store trap is back " \
        "(empty ExperienceStore must not disable remember/recall)"

    # (b) sqlite :memory: create->insert->query
    ctx = Context()
    ctx.use(SQLitePlugin())
    ctx.sqlite.execute("CREATE TABLE t (a, b)")
    r = ctx.sqlite.execute("INSERT INTO t VALUES (?, ?)", ["x", 1])
    assert r.get("affected") == 1, r
    r = ctx.sqlite.query("SELECT * FROM t")
    assert r["rows"] == [("x", 1)], r
    assert r["columns"] == ["a", "b"], r
    r = ctx.sqlite.execute("SELECT * FROM t")  # loud misuse guard
    assert "error" in r, f"execute(SELECT) must return an error dict: {r}"

    # (c) filesystem escape blocked in both directions
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        ctx = Context()
        ctx.use(FilesystemPlugin(root=root))
        r = ctx.fs.read("../escape.txt")
        assert "blocked" in r.get("error", ""), r
        r = ctx.fs.write("../escape.txt", "nope")
        assert "blocked" in r.get("error", ""), r
        r = ctx.fs.write("in.txt", "hello")
        assert r.get("ok"), r
        assert ctx.fs.read("in.txt")["content"] == "hello"

    _pass("test_harness_memory_sqlite_fs_repair",
          "memory round-trip on empty store; sqlite :memory: tuple "
          "rows + execute affected + loud SELECT guard; fs escape blocked")
