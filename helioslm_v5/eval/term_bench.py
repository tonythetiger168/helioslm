"""Terminal-style sandbox tasks (v5.56, roadmap Phase A).

Terminal-Bench-style evaluation in miniature: each task seeds a fresh
sandbox directory, the model drives a bash shell one command per turn,
and a verifier checks the FINAL STATE of the directory. Ten original
tasks cover log filtering, renaming, text extraction, fixing a script,
directory layout, CSV aggregation, file metadata, in-place edits, JSON
filtering and archiving.

Protocol (one turn = one model reply):
- reply with one command in a ```bash block (or a bare command line);
- or finish with ``DONE confidence=<0..1>`` — the stated confidence is
  recorded as a VERBALIZED confidence, so terminal runs go onto the
  calibration board like every other probe-schema artifact;
- each command's stdout+stderr and exit code are shown on the next turn;
  the episode ends at DONE or after ``max_steps`` commands.

Self-grading guard (M1): verifiers are Python functions in this module
and never enter the sandbox, so the agent cannot read or edit them.

Execution safety: commands are model output. Each runs as
``bash -c`` with the sandbox as cwd and HOME, a minimal env, a
wall-clock timeout and, on POSIX, CPU / address-space / file-size
limits. Nothing stops a command from touching paths outside the sandbox
— run untrusted models inside a container.
"""
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CMD_TIMEOUT_S = 10
CMD_MEMORY_BYTES = 1 << 30
CMD_FILE_BYTES = 16 << 20
OUTPUT_CHARS = 2000
MAX_STEPS = 10
_DONE = re.compile(r"^\s*DONE\b(?:.*?confidence\s*=\s*([0-9]*\.?[0-9]+))?",
                   re.I | re.M)
_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.S)


class TermBenchError(ValueError):
    """Raised for malformed tasks or generator output. Never swallowed."""


# -------------------------------------------------------------------- tasks

def _read(root, rel):
    p = Path(root) / rel
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _answer_is(root, rel, want):
    got = _read(root, rel)
    if got is None:
        return False, f"{rel} missing"
    return (got.strip() == want), f"{rel}={got.strip()[:60]!r}"


_LOG = "\n".join(
    f"2026-10-15T10:{i:02d}:00 {lvl} {msg}" for i, (lvl, msg) in enumerate([
        ("INFO", "boot"), ("ERROR", "disk full"), ("WARN", "slow io"),
        ("ERROR", "disk full"), ("INFO", "retry"), ("ERROR", "timeout"),
        ("INFO", "ok"), ("ERROR", "timeout"), ("WARN", "ERRORS suppressed"),
        ("ERROR", "oom"), ("INFO", "restart"), ("ERROR", "oom"),
        ("ERROR", "disk full"), ("INFO", "done")])) + "\n"

_CALC_BUGGY = '''def total(n):
    """Sum of 1..n inclusive."""
    s = 0
    for i in range(1, n):
        s += i
    return s


if __name__ == "__main__":
    print(total(10))
'''

_USERS = [
    {"name": "mei", "age": 34, "active": True},
    {"name": "arjun", "age": 29, "active": True},
    {"name": "zoe", "age": 41, "active": False},
    {"name": "li", "age": 30, "active": True},
    {"name": "omar", "age": 52, "active": True},
    {"name": "ana", "age": 30, "active": False},
]


def _v_count_errors(root):
    # a line counts only when ERROR is the level field, not inside the
    # message ("ERRORS suppressed" is a WARN line)
    return _answer_is(root, "answer.txt", "7")


def _v_rename(root):
    d = Path(root) / "notes"
    names = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    want = ["a.md", "b.md", "c.md", "keep.md"]
    ok = names == want and _read(root, "notes/b.md") == "beta\n"
    return ok, f"notes={names}"


def _v_emails(root):
    return _answer_is(root, "emails.txt",
                      "ana@example.org\nbo@corp.example\nli@example.org")


def _v_fix_script(root):
    code = ("import sys; sys.path.insert(0, '.'); import calc; "
            "print(calc.total(4), calc.total(10), calc.total(1))")
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=root,
                       capture_output=True, text=True, timeout=10,
                       env={})
    got = r.stdout.strip()
    return got == "10 55 1", f"total(4,10,1)={got or r.stderr.strip()[-80:]!r}"


def _v_layout(root):
    r = Path(root)
    ok = ((r / "project" / "src").is_dir()
          and (r / "project" / "tests").is_dir()
          and _read(root, "project/README.md") is not None
          and _read(root, "project/README.md").strip() == "# project")
    return ok, "project/{src,tests,README.md}" + ("" if ok else " incomplete")


def _v_csv_sum(root):
    return _answer_is(root, "total.txt", "1250")


def _v_largest(root):
    return _answer_is(root, "largest.txt", "gamma.bin")


def _v_replace(root):
    a, b = _read(root, "config/app.ini"), _read(root, "config/db.ini")
    bak = _read(root, "config/old.ini.bak")
    ok = (a == "[app]\nhost=db.internal\nport=8080\n"
          and b == "[db]\nhost=db.internal\nreplica=db.internal\n"
          and bak == "[old]\nhost=localhost\n")
    return ok, "ini files rewritten, .bak untouched" if ok else "mismatch"


def _v_json(root):
    return _answer_is(root, "active.txt", "li\nmei\nomar")


def _v_archive(root):
    p = Path(root) / "backup.tar.gz"
    if not p.is_file():
        return False, "backup.tar.gz missing"
    try:
        with tarfile.open(p, "r:gz") as tf:
            members = {m.name.lstrip("./") for m in tf.getmembers()
                       if m.isfile()}
            readme = tf.extractfile("docs/guide/readme.txt")
            content = readme.read().decode() if readme else None
    except (tarfile.TarError, KeyError, OSError) as e:
        return False, f"bad archive: {e}"
    want = {"docs/intro.txt", "docs/guide/readme.txt"}
    return (members == want and content == "read me\n"), f"members={sorted(members)}"


TASKS = [
    {"id": "term/00-count-errors",
     "instruction": "app.log is a log file whose third field is the log "
                    "level. Write the number of lines whose level is ERROR "
                    "to answer.txt (just the number).",
     "files": {"app.log": _LOG},
     "solution": ["awk '$2 == \"ERROR\"' app.log | wc -l > answer.txt"],
     "verify": _v_count_errors},
    {"id": "term/01-rename-ext",
     "instruction": "In the notes/ directory, rename every .txt file to the "
                    "same name with a .md extension. Leave other files alone.",
     "files": {"notes/a.txt": "alpha\n", "notes/b.txt": "beta\n",
               "notes/c.txt": "gamma\n", "notes/keep.md": "keep\n"},
     "solution": ["cd notes && for f in *.txt; do mv \"$f\" \"${f%.txt}.md\"; "
                  "done"],
     "verify": _v_rename},
    {"id": "term/02-unique-emails",
     "instruction": "contacts.txt contains free text with email addresses. "
                    "Write every distinct address, lowercased and sorted, "
                    "one per line, to emails.txt.",
     "files": {"contacts.txt": "Reach Ana at ana@example.org or ANA@example.org.\n"
                               "Bo: bo@corp.example; li@example.org (Li)\n"
                               "cc ana@example.org, see li@Example.org\n"},
     "solution": ["grep -oiE '[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]+' "
                  "contacts.txt | tr 'A-Z' 'a-z' | sort -u > emails.txt"],
     "verify": _v_emails},
    {"id": "term/03-fix-script",
     "instruction": "calc.py defines total(n), which should return the sum "
                    "of 1..n inclusive, but it is off by one. Fix the "
                    "function (keep its name and signature).",
     "files": {"calc.py": _CALC_BUGGY},
     "solution": ["sed -i 's/range(1, n)/range(1, n + 1)/' calc.py"],
     "verify": _v_fix_script},
    {"id": "term/04-project-layout",
     "instruction": "Create a project/ directory containing empty "
                    "directories src/ and tests/ and a README.md whose only "
                    "line is '# project'.",
     "files": {},
     "solution": ["mkdir -p project/src project/tests && "
                  "echo '# project' > project/README.md"],
     "verify": _v_layout},
    {"id": "term/05-csv-sum",
     "instruction": "sales.csv has a header row and columns region,amount. "
                    "Write the total amount for region 'north' to total.txt "
                    "as an integer.",
     "files": {"sales.csv": "region,amount\nnorth,400\nsouth,90\n"
                            "north,250\neast,1000\nnorth,600\nnorthwest,77\n"},
     "solution": ["awk -F, '$1 == \"north\" {s += $2} END {print s}' "
                  "sales.csv > total.txt"],
     "verify": _v_csv_sum},
    {"id": "term/06-largest-file",
     "instruction": "Write the file name (name only, no directory) of the "
                    "largest file in data/ to largest.txt.",
     "files": {"data/alpha.bin": "a" * 120, "data/beta.bin": "b" * 300,
               "data/gamma.bin": "c" * 2048, "data/delta.bin": "d" * 64},
     "solution": ["ls -S data | head -n 1 > largest.txt"],
     "verify": _v_largest},
    {"id": "term/07-replace-host",
     "instruction": "Replace every occurrence of 'localhost' with "
                    "'db.internal' in all .ini files in config/. Do not "
                    "modify any other file.",
     "files": {"config/app.ini": "[app]\nhost=localhost\nport=8080\n",
               "config/db.ini": "[db]\nhost=localhost\nreplica=localhost\n",
               "config/old.ini.bak": "[old]\nhost=localhost\n"},
     "solution": ["sed -i 's/localhost/db.internal/g' config/*.ini"],
     "verify": _v_replace},
    {"id": "term/08-json-filter",
     "instruction": "users.json is a list of users with name, age and "
                    "active fields. Write the names of active users aged 30 "
                    "or older to active.txt, sorted, one per line.",
     "files": {"users.json": json.dumps(_USERS, indent=1) + "\n"},
     "solution": ["python3 -c \"import json; u = json.load(open('users.json')); "
                  "print('\\n'.join(sorted(x['name'] for x in u "
                  "if x['active'] and x['age'] >= 30)))\" > active.txt"],
     "verify": _v_json},
    {"id": "term/09-archive",
     "instruction": "Create backup.tar.gz in the current directory "
                    "containing the docs/ directory (paths inside the "
                    "archive start with docs/) and nothing else.",
     "files": {"docs/intro.txt": "intro\n",
               "docs/guide/readme.txt": "read me\n",
               "scratch.tmp": "not this\n"},
     "solution": ["tar -czf backup.tar.gz docs"],
     "verify": _v_archive},
]


def load_tasks():
    ids = [t["id"] for t in TASKS]
    if len(set(ids)) != len(ids):
        raise TermBenchError("duplicate task id")
    for t in TASKS:
        for key in ("id", "instruction", "files", "solution", "verify"):
            if key not in t:
                raise TermBenchError(f"{t.get('id', '?')}: missing {key!r}")
    return list(TASKS)


# ----------------------------------------------------------------- sandbox

def seed_sandbox(task, root):
    for rel, content in task["files"].items():
        p = Path(root) / rel
        if not p.resolve().is_relative_to(Path(root).resolve()):
            raise TermBenchError(f"{task['id']}: setup path escapes: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": task contents are defined with LF. Text mode's
        # default newline translation rewrites them to CRLF on Windows,
        # which survives into byte-exact artifacts (term/09's tar member
        # "read me\r\n") while _read() silently translates back and hides
        # it everywhere else. Write the bytes the task actually defines.
        p.write_text(content, encoding="utf-8", newline="\n")


def _limit_resources():   # pragma: no cover - runs in the child process
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (CMD_TIMEOUT_S, CMD_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (CMD_MEMORY_BYTES, CMD_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CMD_FILE_BYTES, CMD_FILE_BYTES))


def _sandbox_path():
    """PATH for the sandbox shell.

    On POSIX the pinned minimal PATH is enough: bash, awk, sed, tar and
    python3 all live in /usr/bin. On Windows the shell is Git Bash, whose
    /usr/bin carries the GNU tools but NOT python3. Without a python3 on
    PATH, every python3-dependent command (term/08-json-filter, every
    swe/* patch and hidden-test run) fails with exit 127 and, because the
    shell redirect creates the output file first, fails SILENTLY as an
    empty artifact.

    Two candidate locations are appended (whichever exist): the directory
    of the python3 on the current process PATH, and the directory of the
    running interpreter. Both are needed: this repo's automation runs
    under two different interpreters (the managed .venv, whose Scripts
    dir ships python3.exe, and the bundle cpython, whose dir does not),
    and ``sys.executable`` alone therefore only covers one of them.
    Paths are converted to /c/... POSIX form for Git Bash. POSIX
    behaviour is unchanged.
    """
    base = "/usr/local/bin:/usr/bin:/bin"
    if os.name != "nt":
        return base
    import shutil
    cands = []
    w = shutil.which("python3")
    if w:
        cands.append(os.path.dirname(os.path.abspath(w)))
    cands.append(os.path.dirname(os.path.abspath(sys.executable)))
    extra = []
    for d in cands:
        if not os.path.isdir(d):
            continue
        p = d.replace("\\", "/")
        if len(p) > 1 and p[1] == ":":
            p = "/" + p[0].lower() + p[2:]
        if p not in extra:
            extra.append(p)
    return base + "".join(":" + d for d in extra)


def _kill_tree(proc):
    """Kill ``proc`` and every descendant (timeout path).

    ``swe/XX-binary-search``'s bug IS an infinite loop: the hidden test
    against the untouched/trap repo spins forever, so sandbox timeouts
    are routine, not exceptional. ``subprocess``'s timeout only kills the
    direct child (bash); a bash child such as ``python3 _hidden_test.py``
    — on this automation host itself a shim that re-execs a grandchild
    interpreter — survives as an ORPHAN and burns a core indefinitely
    (measured: one such orphan accumulated 81188 s of CPU over ~22.7 h,
    and its inherited pipe handle deadlocked a later ``communicate()``).
    POSIX: the child runs in its own session (start_new_session), so
    killpg reaches the whole group. Windows: taskkill /T walks the
    process tree. Loud on unexpected failure is unnecessary here — a
    kill failure leaves the old behaviour, not wrong results.
    """
    if os.name == "posix":
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)


def run_command(cmd, root, timeout=CMD_TIMEOUT_S):
    """Run one shell command in the sandbox -> (exit code, output)."""
    env = {"PATH": _sandbox_path(), "HOME": str(root),
           "LC_ALL": "C", "LANG": "C"}
    kwargs = ({"preexec_fn": _limit_resources, "start_new_session": True}
              if os.name == "posix" else {})
    p = subprocess.Popen(["bash", "-c", cmd], cwd=root, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, **kwargs)
    try:
        so, se = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(p)
        so, se = p.communicate()
        return 124, f"[timed out after {timeout}s]"
    out = (so + se)
    if len(out) > OUTPUT_CHARS:
        out = out[:OUTPUT_CHARS] + f"\n[... {len(out) - OUTPUT_CHARS} chars truncated]"
    return rc, out


# ---------------------------------------------------------------- protocol

def render_prompt(task, history, step, max_steps=MAX_STEPS):
    lines = [
        "You are operating a bash shell. Your working directory is a "
        "sandbox that already contains the task's files. Complete the task "
        "by running shell commands, one per reply.",
        "Reply with exactly one command in a ```bash code block. When the "
        "task is complete, reply with 'DONE confidence=<p>' where <p> is "
        "your probability (0 to 1) that the task is solved correctly.",
        "",
        f"[{task['id']}, step {step + 1} of {max_steps}]",
        f"Task: {task['instruction']}",
    ]
    if history:
        lines += ["", "Commands so far:"]
        for cmd, code, out in history:
            lines += [f"$ {cmd}", out.rstrip() if out.strip() else "(no output)",
                      f"(exit {code})"]
    return "\n".join(lines) + "\n"


def parse_reply(text):
    """-> ('done', conf|None) or ('cmd', command) or ('invalid', reason)."""
    text = text or ""
    fence = _FENCE.search(text)
    if fence and fence.group(1).strip():
        return "cmd", fence.group(1).strip()
    m = _DONE.search(text)
    if m:
        conf = float(m.group(1)) if m.group(1) else None
        if conf is not None and not 0.0 <= conf <= 1.0:
            conf = None   # out-of-range verbal confidence is not recorded
        return "done", conf
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first.startswith("$ "):
        return "cmd", first[2:].strip()
    return "invalid", "no ```bash block and no DONE line"


def run_episode(task, generate_fn, sample=0, max_steps=MAX_STEPS):
    """One episode in a fresh sandbox -> result dict (probe schema row)."""
    with tempfile.TemporaryDirectory(prefix="term_bench_") as root:
        seed_sandbox(task, root)
        history, conf, ended, invalid = [], None, "max_steps", 0
        for step in range(max_steps):
            reply = generate_fn(render_prompt(task, history, step, max_steps),
                                sample)
            if isinstance(reply, tuple):
                reply = reply[0]
            kind, value = parse_reply(reply)
            if kind == "done":
                conf, ended = value, "done"
                break
            if kind == "invalid":
                invalid += 1
                history.append(("<no command>", 2, value))
                continue
            code, out = run_command(value, root)
            history.append((value, code, out))
        ok, detail = task["verify"](root)
    return {"family": "terminal", "task": task["id"],
            "expected": "verifier passes", "final": detail,
            "correct": bool(ok), "steps": len(history), "ended": ended,
            "invalid_replies": invalid,
            "commands": [cmd for cmd, _, _ in history], "conf": conf,
            "conf_type": "verbalized" if conf is not None else None}


def run_term_bench(tasks, generate_fn, n_samples=1, max_steps=MAX_STEPS):
    """``correct`` is episode 0; ``pass_at_1`` is c/n over n episodes."""
    results = []
    for t in tasks:
        eps = [run_episode(t, generate_fn, i, max_steps)
               for i in range(n_samples)]
        row = dict(eps[0])
        row["pass_at_1"] = sum(e["correct"] for e in eps) / n_samples
        row["n_samples"] = n_samples
        results.append(row)
    return results
