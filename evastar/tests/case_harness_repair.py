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

import os
import sys
from pathlib import Path

from evastar.tests.case_common import _REPO_ROOT, _pass


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
#   Tier 1 — dotted namespace-package imports over evastar/ (src,
#     configs, ...) and harness/. The tree has NO __init__.py outside
#     agent/envs, so pkgutil.walk_packages silently skips every
#     subpackage — that silence is exactly how a broken module hid. The
#     scan is therefore filesystem-based (os.walk).
#   Tier 2 — BARE top-level imports of evastar/agent/** (the modules
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
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT / "evastar"):
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
        agent_dir = _REPO_ROOT / "evastar" / "agent"
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
    assert "evastar.src.model_v5" in seen, (
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


# ---------------------------------------------------------------------------
# v5.110 (2026-10-23): Windows checkout-repair locks. A fresh Windows clone
# failed four harness behaviours that POSIX CI never saw; the fixes live in
# eval/term_bench.py (_sandbox_path / _kill_tree), eval/swe_bench.py
# (base64 patch transport) and the root .gitattributes (LF policy). These
# tests lock the repaired behaviours so a later "simplification"
# reintroduces the failures loudly. Root causes + measurements: CHANGELOG
# v5.110.
# ---------------------------------------------------------------------------

_TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".yml", ".yaml",
             ".csv", ".sh", ".toml", ".cfg", ".ini"}


def test_pinned_artifacts_lf_eol():
    """The digest-pinned benchmark/data artifacts must stay LF. A Windows
    checkout (core.autocrlf) serves CRLF bytes, and CRLF sha256 != the
    pinned LF sha256 (measured: api_probe_deepseek CRLF 104978f5... vs
    pinned LF 9bd4ae7d...), so calibration-board/cost/std-bench digest
    gates failed on content that was actually right. The root
    .gitattributes now pins `* text=auto eol=lf` plus a binary list.
    Locked: the policy file exists and benchmarks/ + data/ text files
    carry no CR byte."""
    ga = _REPO_ROOT / ".gitattributes"
    assert ga.is_file(), ".gitattributes missing: LF checkout policy lost"
    body = ga.read_text(encoding="utf-8")
    assert "text=auto" in body and "eol=lf" in body, body
    bad = []
    for sub in ("evastar/benchmarks", "evastar/data"):
        base = _REPO_ROOT / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in _TEXT_EXT:
                if b"\r" in p.read_bytes():
                    bad.append(str(p.relative_to(_REPO_ROOT)))
    assert not bad, f"CRLF bytes in pinned artifacts: {bad[:5]}"
    _pass("test_pinned_artifacts_lf_eol",
          ".gitattributes eol=lf; benchmarks/ + data/ CRLF-free")


def test_sandbox_python3_resolvable():
    """Git Bash's /usr/bin ships the GNU tools but NOT python3 on Windows,
    so every `python3 ...` sandbox command exited 127 — and because the
    shell creates redirect targets first, it failed SILENTLY as empty
    artifacts (term/08, all swe/* patch + hidden-test runs). term_bench
    now appends the real interpreter directories to the sandbox PATH.
    Locked end-to-end: `python3 -c` runs inside the sandbox PATH."""
    import os as _os
    import tempfile
    from evastar.eval import term_bench as tb
    path = tb._sandbox_path()
    assert "/usr/bin" in path, path
    if _os.name == "nt":
        # the appended dirs must be POSIX-form (/c/...) for Git Bash
        extras = [d for d in path.split(":") if d.startswith("/") and
                  len(d) > 2 and d[2] == "/"]
        assert extras, f"no Windows interpreter dir appended: {path}"
    with tempfile.TemporaryDirectory() as root:
        rc, out = tb.run_command("python3 -c \"print('py3-ok')\"", root,
                                 timeout=30)
    assert rc == 0 and "py3-ok" in out, (rc, out)
    _pass("test_sandbox_python3_resolvable",
          "sandbox ran python3 end-to-end on Windows")


def test_sandbox_timeout_kills_tree():
    """swe/07-binary-search's bug IS an infinite loop, so sandbox timeouts
    are routine. subprocess' timeout only killed the direct child (bash);
    a python3 grandchild — on this host a shim re-execing the real
    interpreter — survived as an orphan and burned a core (measured:
    81188 s CPU over ~22.7 h; its inherited pipe also deadlocked a later
    communicate()). run_command now kills the whole tree (POSIX killpg on
    the child session, Windows taskkill /T). Locked: an infinite-loop
    command returns 124 promptly and the grandchild interpreter is dead."""
    import os as _os
    import tempfile
    import time
    from evastar.eval import term_bench as tb
    with tempfile.TemporaryDirectory() as root:
        t0 = time.time()
        rc, out = tb.run_command(
            "python3 -c \"import os,time; "
            "open('pid.txt','w').write(str(os.getpid())); "
            "time.sleep(1000)\"",
            root, timeout=5)
        dt = time.time() - t0
        assert rc == 124, (rc, out)
        assert dt < 20, f"timeout path took {dt:.1f}s (kill tree broken?)"
        pid = int((Path(root) / "pid.txt").read_text().strip())
    alive = True
    try:
        if _os.name == "posix":
            _os.kill(pid, 0)
        else:
            import subprocess as _sp
            r = _sp.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV",
                         "/NH"], capture_output=True, text=True)
            alive = str(pid) in r.stdout
    except (ProcessLookupError, PermissionError):
        alive = False
    assert not alive, f"grandchild pid {pid} survived the timeout kill"
    _pass("test_sandbox_timeout_kills_tree",
          f"infinite-loop command returned 124 in {dt:.1f}s and the "
          "grandchild interpreter is dead")


def test_patch_command_base64_roundtrip():
    r"""swe/13-parse-duration's fix carries a regex (\d+); passed inline as
    JSON argv, the Windows command-line chain (CreateProcess quoting ->
    Git Bash CRT parse -> python3 launcher shim re-exec) re-serialized it
    and dropped one backslash layer -> json.decoder.JSONDecodeError:
    Invalid \escape at the sandbox python3. patch_command now ships the
    payload base64-encoded (the alphabet survives every layer). Locked
    end-to-end through the REAL sandbox: a backslash-bearing edit applies
    cleanly, and the command line carries no raw backslash at all."""
    import tempfile
    from evastar.eval import swe_bench as sb
    from evastar.eval import term_bench as tb
    old = 'm = re.match(r"^(\\d+)h(?:(\\d+)m)?$", s)'
    new = ('m = re.match(r"^(\\d+(?:\\.\\d+)?)h'
           '(?:(\\d+(?:\\.\\d+)?)m)?$", s)')
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / "parse.py"
        target.write_text("import re\n" + old + "\n", encoding="utf-8",
                          newline="\n")
        cmd = sb.patch_command([("parse.py", old, new)])
        assert "\\" not in cmd, \
            "raw backslash leaked into the command line (base64 lost?)"
        rc, out = tb.run_command(cmd, root, timeout=30)
        assert rc == 0, (rc, out)
        body = target.read_text(encoding="utf-8")
        assert new in body and old not in body, body
    _pass("test_patch_command_base64_roundtrip",
          "regex-bearing patch applied via base64 payload in the real "
          "sandbox")


# ---------------------------------------------------------------------------
# v5.132 (2026-10-24): three repo-hygiene locks — the sandbox pwd/$HOME
# POSIX-form semantics the repaired test_sandbox_limits relies on, the CI
# LF guard's presence, and the model_name <= changelog-head invariant.
# See CHANGELOG v5.132.
# ---------------------------------------------------------------------------

def test_sandbox_pwd_home_posix_form():
    """test_term_bench.py::test_sandbox_limits asserted
    `out.split() == [real, root]` with Windows-form paths, but Git Bash
    prints the cwd in POSIX form (/c/..., or an MSYS /tmp path when TMPDIR
    redirects tempfile) and the MSYS runtime rewrites even the HOME we
    pass in — so the test failed on every Windows host while CI (Linux)
    stayed green. v5.132 repaired that comparison to assert semantics
    (pwd == $HOME, marker file lands in `root`) instead of path spelling;
    this test locks the underlying sandbox behaviour it relies on."""
    import os as _os
    import tempfile
    from evastar.eval import term_bench as tb
    with tempfile.TemporaryDirectory() as root:
        rc, out = tb.run_command("pwd; echo $HOME; echo hi > .pwd_probe",
                                 root, timeout=30)
        assert rc == 0, (rc, out)
        lines = out.split()
        assert len(lines) == 2 and lines[0] == lines[1], out
        assert (Path(root) / ".pwd_probe").is_file(), out
        if _os.name == "nt":
            assert lines[0].startswith("/") and ":" not in lines[0], out
        else:
            assert lines[0] == str(Path(root).resolve()), out
    _pass("test_sandbox_pwd_home_posix_form",
          "sandbox pwd == $HOME; POSIX path form on Windows")


def test_ci_lf_guard():
    """The v5.110 .gitattributes policy governs future checkouts, but only
    CI can stop a CRLF blob from reaching the index (a Windows contributor
    committing without the policy applied). v5.132 added an "LF guard"
    step to the syntax-gate job. Locked: the workflow still contains it —
    silently dropping the guard reopens the digest-drift failure mode."""
    ci = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.is_file(), ".github/workflows/ci.yml missing"
    body = ci.read_text(encoding="utf-8")
    assert "LF guard" in body and "git ls-files" in body, \
        "LF guard step missing from ci.yml"
    _pass("test_ci_lf_guard", "ci.yml syntax-gate carries the LF guard step")


def test_version_never_ahead_of_changelog():
    """`model_name` may LAG the changelog head: upstream entries do not
    update it (the 2026-10-23 calibration found it stuck at v5.49 with the
    changelog at v5.109 — a 60-minor drift), and the daily maintenance job
    re-calibrates. But it must never claim a version NEWER than the
    changelog head: that would assert an undocumented release. Locked:
    model_version <= changelog_head_version."""
    import re
    cfg = (_REPO_ROOT / "evastar" / "configs" / "config_v5.py") \
        .read_text(encoding="utf-8")
    m = re.search(r'model_name:\s*str\s*=\s*"(?:Evastar|HeliosLM)-v(\d+)\.(\d+)"', cfg)
    assert m, "model_name not found or unparseable in config_v5.py"
    ch = (_REPO_ROOT / "evastar" / "CHANGELOG.md") \
        .read_text(encoding="utf-8")
    h = re.search(r"^## v(\d+)\.(\d+)", ch, re.M)
    assert h, "no version heading found in CHANGELOG.md"
    model_v = (int(m.group(1)), int(m.group(2)))
    head_v = (int(h.group(1)), int(h.group(2)))
    assert model_v <= head_v, \
        f"model_name v{model_v[0]}.{model_v[1]} claims a version newer " \
        f"than the changelog head v{head_v[0]}.{head_v[1]}"
    _pass("test_version_never_ahead_of_changelog",
          f"model_name v{model_v[0]}.{model_v[1]} <= changelog head "
          f"v{head_v[0]}.{head_v[1]} (lag tolerated, calibrated daily)")


# ---------------------------------------------------------------------------
# v5.153 (2026-10-25): the v5.134 package rename (helioslm_v5 -> evastar,
# alias shim in helioslm_v5/__init__.py) broke `python -m
# helioslm_v5.tests.test_v5` — the daily-maintenance mandated gate command —
# because runpy._get_module_details requires loader.get_code and the alias
# loader did not provide it (AttributeError: '_AliasLoader' object has no
# attribute 'get_code'). v5.153 adds get_code/get_source to _AliasLoader and
# a `--list` probe to the suite's main(); these two tests lock both halves.
# See CHANGELOG v5.153.
# ---------------------------------------------------------------------------

def test_alias_loader_get_code():
    """Locked: the helioslm_v5 -> evastar alias loader exposes get_code
    (runpy's requirement for `python -m`) and get_source (inspect/pdb),
    both compiled/read from the real evastar file so tracebacks point at
    the canonical path."""
    import importlib.util as _iu
    import helioslm_v5  # noqa: F401 — registers the meta-path alias
    spec = _iu.find_spec("helioslm_v5.tests.test_v5")
    assert spec is not None and spec.loader is not None, \
        "alias spec for helioslm_v5.tests.test_v5 not found"
    loader = spec.loader
    assert hasattr(loader, "get_code") and hasattr(loader, "get_source"), \
        "alias loader lost get_code/get_source (python -m will break)"
    code = loader.get_code("helioslm_v5.tests.test_v5")
    assert hasattr(code, "co_names") and "main" in code.co_names, \
        "get_code did not compile the real test_v5 module"
    assert code.co_filename.replace("/", os.sep).endswith(
        os.path.join("evastar", "tests", "test_v5.py")), code.co_filename
    src = loader.get_source("helioslm_v5.tests.test_v5")
    assert "TESTS = [" in src, "get_source did not read the real file"
    _pass("test_alias_loader_get_code",
          "alias loader get_code/get_source compile/read the real evastar "
          "test_v5.py (python -m entry point repaired)")


def test_alias_dash_m_smoke():
    """End-to-end lock for the mandated daily gate: `python -m
    helioslm_v5.tests.test_v5 --list` must exit 0 and report the same
    test count as the in-process TESTS registry. Runs the real subprocess
    through the alias path; --list keeps it under a few seconds (module
    import only, no tests executed)."""
    import re as _re
    import subprocess as _sp
    r = _sp.run([sys.executable, "-m", "helioslm_v5.tests.test_v5",
                 "--list"], capture_output=True, text=True, timeout=180,
                cwd=str(_REPO_ROOT))
    assert r.returncode == 0, \
        f"mandated -m command failed rc={r.returncode}: {r.stderr[-500:]}"
    m = _re.search(r"TESTS=(\d+)", r.stdout)
    assert m, f"--list probe output missing TESTS count: {r.stdout[-300:]}"
    from evastar.tests.test_v5 import TESTS as _TESTS
    assert int(m.group(1)) == len(_TESTS), \
        f"subprocess reports {m.group(1)} tests, registry has {len(_TESTS)}"
    _pass("test_alias_dash_m_smoke",
          f"python -m helioslm_v5.tests.test_v5 --list exits 0, "
          f"TESTS={m.group(1)} matches registry")
