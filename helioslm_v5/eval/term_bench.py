"""Terminal-style sandbox tasks (v5.56, roadmap Phase A).

Terminal-Bench-style evaluation in miniature: each task seeds a fresh
sandbox directory, the model drives a bash shell one command per turn,
and a verifier checks the FINAL STATE of the directory. Fifty tasks in
three difficulty tiers cover log surgery, config repair, CSV/JSON/JSONL
pipelines, git archaeology on a fixture repository, static cron/systemd
authoring, and multi-file refactors with hidden seeded test oracles.

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
import random
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
     "instruction": "app.log is a log file whose second field is the log "
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


# Every built-in task carries a plausible-but-wrong command. The generic
# test gate applies the same no-op / reference / trap contract to all 50.
_ORIGINAL_META = {
    "term/00-count-errors": ("easy", "log", "grep -c ERROR app.log > answer.txt"),
    "term/01-rename-ext": ("easy", "filesystem", "mv notes/a.txt notes/a.md"),
    "term/02-unique-emails": ("easy", "text", "grep -oE '[a-zA-Z0-9.]+@[a-zA-Z.]+' contacts.txt | sort -u > emails.txt"),
    "term/03-fix-script": ("easy", "repair", "printf 'def total(n):\\n    return 55\\n' > calc.py"),
    "term/04-project-layout": ("easy", "filesystem", "mkdir -p project/src"),
    "term/05-csv-sum": ("easy", "data", "awk -F, '$1 ~ /north/ {s+=$2} END {print s}' sales.csv > total.txt"),
    "term/06-largest-file": ("easy", "filesystem", "echo alpha.bin > largest.txt"),
    "term/07-replace-host": ("easy", "config", "sed -i 's/localhost/db.internal/g' config/*"),
    "term/08-json-filter": ("easy", "data", "python3 -c \"import json;print('\\n'.join(x['name'] for x in json.load(open('users.json')) if x['active']))\" > active.txt"),
    "term/09-archive": ("easy", "filesystem", "tar -czf backup.tar.gz ."),
}
for _task in TASKS:
    _tier, _family, _trap = _ORIGINAL_META[_task["id"]]
    _task.update(tier=_tier, family=_family, verification="exact", trap=_trap)


def _commands(value):
    return [value] if isinstance(value, str) else list(value)


def _v_exact(expected, absent=()):
    """Byte-for-byte final files, plus optionally forbidden artifacts."""
    def verify(root):
        for rel, want in expected.items():
            got = _read(root, rel)
            if got != want:
                return False, f"{rel}: {got!r} != {want!r}"
        extra = [rel for rel in absent if (Path(root) / rel).exists()]
        return (not extra), ("exact final state" if not extra else f"unexpected: {extra}")
    return verify


def _v_function_cases(module, function, cases):
    """Hidden function calls; ``cases`` is [(args tuple, expected), ...]."""
    def verify(root):
        code = ("import json,sys;sys.path.insert(0,'.');"
                f"from {module} import {function} as f;cases={cases!r};"
                "out=[]\nfor args,want in cases:\n"
                " try: out.append(f(*args))\n"
                " except Exception as e: out.append(type(e).__name__)\n"
                "print(json.dumps(out))")
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=root,
                               capture_output=True, text=True, timeout=10, env={})
            got = json.loads(r.stdout) if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            got = None
        want = [want for _, want in cases]
        return got == want, f"hidden cases: {got!r} != {want!r}"
    return verify


def _v_json_script(script, cases):
    """Run a JSON-list transformer over deterministic hidden cases."""
    def verify(root):
        got = []
        try:
            for value, _ in cases:
                # A scalar case is a one-item input batch; a list case is a
                # complete batch (including the important empty-list case).
                payload = value if isinstance(value, list) else [value]
                r = subprocess.run([sys.executable, "-I", script], cwd=root,
                                   input=json.dumps(payload), capture_output=True,
                                   text=True, timeout=10, env={})
                got.append(json.loads(r.stdout) if r.returncode == 0 else None)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            got.append(None)
        want = [y if isinstance(y, list) else [y] for _, y in cases]
        return got == want, f"seeded transform: {got!r} != {want!r}"
    return verify


def _new_task(tid, instruction, files, solution, trap, tier, family,
              expected=None, verify=None, verification="exact", setup=()):
    if verify is None:
        verify = _v_exact(expected)
    return {"id": tid, "instruction": instruction, "files": files,
            "solution": _commands(solution), "verify": verify, "trap": trap,
            "tier": tier, "family": family, "verification": verification,
            "setup": _commands(setup)}


TASKS.extend([
    _new_task("term/10-timeout-errors",
              "Extract only the ERROR lines whose message is timeout from app.log to timeouts.txt, preserving order.",
              {"app.log": _LOG},
              "awk '$2 == \"ERROR\" && $3 == \"timeout\"' app.log > timeouts.txt",
              "grep ERROR app.log > timeouts.txt", "easy", "log",
              {"timeouts.txt": "2026-10-15T10:05:00 ERROR timeout\n2026-10-15T10:07:00 ERROR timeout\n"}),
    _new_task("term/11-error-messages",
              "Write the distinct messages from ERROR lines in app.log to messages.txt, sorted, one per line.",
              {"app.log": _LOG},
              "awk '$2 == \"ERROR\" {$1=$2=\"\"; sub(/^  /, \"\"); print}' app.log | sort -u > messages.txt",
              "grep ERROR app.log | sort -u > messages.txt", "easy", "log",
              {"messages.txt": "disk full\noom\ntimeout\n"}),
    _new_task("term/12-redact-network",
              "Copy network.log to redacted.log, replacing every IPv4 address with <IP> and leaving all other text unchanged.",
              {"network.log": "accept 10.0.0.1 port 443\nreject 192.168.1.20\nhost 10.0.0.1 again\n"},
              "sed -E 's/([0-9]{1,3}\\.){3}[0-9]{1,3}/<IP>/g' network.log > redacted.log",
              "sed 's/10.0.0.1/<IP>/' network.log > redacted.log", "medium", "log",
              {"redacted.log": "accept <IP> port 443\nreject <IP>\nhost <IP> again\n"}),
    _new_task("term/13-endpoint-counts",
              "Count requests by endpoint in access.log. Write 'count endpoint' lines to counts.txt, sorted by descending count then ascending endpoint.",
              {"access.log": "GET /a\nGET /b\nGET /a\nPOST /c\nGET /b\nGET /a\n"},
              "python3 -c \"from collections import Counter;c=Counter(x.split()[1] for x in open('access.log'));print('\\n'.join(f'{n} {p}' for p,n in sorted(c.items(),key=lambda x:(-x[1],x[0]))))\" > counts.txt",
              "sort access.log | uniq -c > counts.txt", "medium", "log",
              {"counts.txt": "3 /a\n2 /b\n1 /c\n"}),
    _new_task("term/14-log-window",
              "From events.log, copy WARN or ERROR lines with timestamps from 2026-10-15T10:02:00 through 2026-10-15T10:04:00 inclusive to window.log.",
              {"events.log": "2026-10-15T10:01:59 ERROR early\n2026-10-15T10:02:00 WARN start\n2026-10-15T10:03:00 INFO skip\n2026-10-15T10:04:00 ERROR end\n2026-10-15T10:04:01 ERROR late\n"},
              "awk '$1 >= \"2026-10-15T10:02:00\" && $1 <= \"2026-10-15T10:04:00\" && ($2 == \"WARN\" || $2 == \"ERROR\")' events.log > window.log",
              "grep -E 'WARN|ERROR' events.log > window.log", "medium", "log",
              {"window.log": "2026-10-15T10:02:00 WARN start\n2026-10-15T10:04:00 ERROR end\n"}),
    _new_task("term/15-level-summary",
              "Write levels.json containing exactly the counts of INFO, WARN and ERROR lines in app.log, with keys sorted.",
              {"app.log": _LOG},
              "python3 -c \"import json,collections;c=collections.Counter(x.split()[1] for x in open('app.log'));open('levels.json','w').write(json.dumps(dict(sorted(c.items())),indent=2)+'\\n')\"",
              "grep -c ERROR app.log > levels.json", "medium", "log",
              {"levels.json": "{\n  \"ERROR\": 7,\n  \"INFO\": 5,\n  \"WARN\": 2\n}\n"}),
    _new_task("term/16-production-port",
              "In deploy.ini change port to 9090 only in the [production] section; leave [development] unchanged.",
              {"deploy.ini": "[development]\nport=8000\n[production]\nport=8000\nworkers=2\n"},
              "python3 -c \"from pathlib import Path;p=Path('deploy.ini');s=p.read_text();s=s.replace('[production]\\nport=8000','[production]\\nport=9090');p.write_text(s)\"",
              "sed -i 's/port=8000/port=9090/g' deploy.ini", "medium", "config",
              {"deploy.ini": "[development]\nport=8000\n[production]\nport=9090\nworkers=2\n"}),
    _new_task("term/17-json-merge",
              "Merge override.json on top of defaults.json into merged.json. Do not modify either input; format with sorted keys and two-space indentation.",
              {"defaults.json": "{\"debug\": false, \"port\": 80, \"host\": \"local\"}\n",
               "override.json": "{\"port\": 443, \"debug\": true}\n"},
              "python3 -c \"import json;a=json.load(open('defaults.json'));a.update(json.load(open('override.json')));open('merged.json','w').write(json.dumps(a,sort_keys=True,indent=2)+'\\n')\"",
              "cp override.json merged.json", "easy", "config",
              {"merged.json": "{\n  \"debug\": true,\n  \"host\": \"local\",\n  \"port\": 443\n}\n"}),
    _new_task("term/18-env-to-json",
              "Parse app.env into env.json as string key/value pairs. Ignore blank lines and comments and remove the optional 'export ' prefix; sort keys.",
              {"app.env": "# settings\nexport HOST=example.org\nPORT=8080\n\nMODE=prod\n"},
              "python3 -c \"import json;d={};lines=open('app.env').read().splitlines();[(d.__setitem__(*x.removeprefix('export ').split('=',1))) for x in lines if x and not x.startswith('#')];open('env.json','w').write(json.dumps(d,sort_keys=True,indent=2)+'\\n')\"",
              "cp app.env env.json", "medium", "config",
              {"env.json": "{\n  \"HOST\": \"example.org\",\n  \"MODE\": \"prod\",\n  \"PORT\": \"8080\"\n}\n"}),
    _new_task("term/19-config-keys",
              "In settings.conf set mode=safe and timeout=30, preserving the order and every other line exactly.",
              {"settings.conf": "# runtime\nmode=fast\ntimeout=5\nretries=3\n"},
              "sed -i -e 's/^mode=.*/mode=safe/' -e 's/^timeout=.*/timeout=30/' settings.conf",
              "printf 'mode=safe\\ntimeout=30\\n' > settings.conf", "easy", "config",
              {"settings.conf": "# runtime\nmode=safe\ntimeout=30\nretries=3\n"}),
    _new_task("term/20-enabled-site",
              "In sites/enabled.conf replace old.example with new.example. Leave sites/available.conf byte-for-byte unchanged.",
              {"sites/enabled.conf": "server {\n  server_name old.example;\n}\n",
               "sites/available.conf": "server_name old.example;\n"},
              "sed -i 's/old\\.example/new.example/g' sites/enabled.conf",
              "sed -i 's/old\\.example/new.example/g' sites/*", "medium", "config",
              {"sites/enabled.conf": "server {\n  server_name new.example;\n}\n",
               "sites/available.conf": "server_name old.example;\n"}),
    _new_task("term/21-database-configs",
              "Replace db-old with db-new in every .conf file directly under conf/. Do not change conf/archive.conf.bak.",
              {"conf/a.conf": "primary=db-old\n", "conf/b.conf": "replica=db-old\n",
               "conf/archive.conf.bak": "primary=db-old\n"},
              "sed -i 's/db-old/db-new/g' conf/*.conf",
              "sed -i 's/db-old/db-new/g' conf/*", "easy", "config",
              {"conf/a.conf": "primary=db-new\n", "conf/b.conf": "replica=db-new\n",
               "conf/archive.conf.bak": "primary=db-old\n"}),
])

# ----------------------------------------------------------------- sandbox

TASKS.extend([
    _new_task("term/22-csv-averages",
              "Compute the mean amount for each region in amounts.csv. Write averages.json with sorted region keys and numeric values.",
              {"amounts.csv": "region,amount\nnorth,10\nnorth,30\neast,15\nwest,7\nwest,13\n"},
              "python3 -c \"import csv,json,collections;d=collections.defaultdict(list);[(d[r['region']].append(float(r['amount']))) for r in csv.DictReader(open('amounts.csv'))];o={k:sum(v)/len(v) for k,v in sorted(d.items())};open('averages.json','w').write(json.dumps(o,sort_keys=True,indent=2)+'\\n')\"",
              "awk -F, '{s[$1]+=$2;n[$1]++}END{for(k in s)print k,s[k]/n[k]}' amounts.csv > averages.json", "medium", "data",
              {"averages.json": "{\n  \"east\": 15.0,\n  \"north\": 20.0,\n  \"west\": 10.0\n}\n"}),
    _new_task("term/23-quoted-csv",
              "people.csv is quoted CSV. Write the city field for rows whose name contains a comma to comma-cities.txt, sorted.",
              {"people.csv": "name,city\n\"Smith, Ana\",Paris\nBob,Berlin\n\"Lee, Li\",Taipei\n"},
              "python3 -c \"import csv;r=[x['city'] for x in csv.DictReader(open('people.csv')) if ',' in x['name']];print('\\n'.join(sorted(r)))\" > comma-cities.txt",
              "awk -F, 'NF>2{print $2}' people.csv > comma-cities.txt", "medium", "data",
              {"comma-cities.txt": "Paris\nTaipei\n"}),
    _new_task("term/24-csv-pivot",
              "Build pivot.json from status.csv: nested region objects containing counts for each status, with all keys sorted.",
              {"status.csv": "region,status\nnorth,ok\nnorth,fail\nsouth,ok\nnorth,ok\nsouth,ok\n"},
              "python3 -c \"import csv,json,collections;d=collections.defaultdict(collections.Counter);[(d[r['region']].update([r['status']])) for r in csv.DictReader(open('status.csv'))];o={k:dict(sorted(v.items())) for k,v in sorted(d.items())};open('pivot.json','w').write(json.dumps(o,sort_keys=True,indent=2)+'\\n')\"",
              "sort status.csv | uniq -c > pivot.json", "hard", "data",
              {"pivot.json": "{\n  \"north\": {\n    \"fail\": 1,\n    \"ok\": 2\n  },\n  \"south\": {\n    \"ok\": 2\n  }\n}\n"}),
    _new_task("term/25-jsonl-filter",
              "events.jsonl has one JSON object per line. Write only objects with status 'ok' to ok.jsonl, preserving order and using compact sorted-key JSON.",
              {"events.jsonl": "{\"id\":1,\"status\":\"ok\"}\n{\"id\":2,\"status\":\"bad\"}\n{\"status\":\"ok\",\"id\":3}\n"},
              "python3 -c \"import json;rows=[json.loads(x) for x in open('events.jsonl')];open('ok.jsonl','w').write(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n' for x in rows if x['status']=='ok'))\"",
              "grep ok events.jsonl > ok.jsonl", "medium", "data",
              {"ok.jsonl": "{\"id\":1,\"status\":\"ok\"}\n{\"id\":3,\"status\":\"ok\"}\n"}),
    _new_task("term/26-jsonl-normalize",
              "Convert users.jsonl to normalized.jsonl, keeping only active users and objects with id plus lowercased name. Preserve input order and use compact sorted-key JSON.",
              {"users.jsonl": "{\"id\":1,\"name\":\"MEI\",\"active\":true}\n{\"id\":2,\"name\":\"Bob\",\"active\":false}\n{\"id\":3,\"name\":\"LI\",\"active\":true}\n"},
              "python3 -c \"import json;rows=[json.loads(x) for x in open('users.jsonl')];open('normalized.jsonl','w').write(''.join(json.dumps({'id':x['id'],'name':x['name'].lower()},sort_keys=True,separators=(',',':'))+'\\n' for x in rows if x['active']))\"",
              "grep true users.jsonl > normalized.jsonl", "medium", "data",
              {"normalized.jsonl": "{\"id\":1,\"name\":\"mei\"}\n{\"id\":3,\"name\":\"li\"}\n"}),
    _new_task("term/27-nested-orders",
              "orders.json is a list of customers, each with name and an orders list containing total values. Write totals.json mapping each customer name to their sum, sorted by key.",
              {"orders.json": json.dumps([{"name":"ana","orders":[{"total":2.5},{"total":3}]},{"name":"bo","orders":[{"total":4}]}])+"\n"},
              "python3 -c \"import json;d=json.load(open('orders.json'));o={x['name']:sum(y['total'] for y in x['orders']) for x in d};open('totals.json','w').write(json.dumps(o,sort_keys=True,indent=2)+'\\n')\"",
              "python3 -c \"import json;print(len(json.load(open('orders.json'))))\" > totals.json", "hard", "data",
              {"totals.json": "{\n  \"ana\": 5.5,\n  \"bo\": 4\n}\n"}),
    _new_task("term/28-add-defaults",
              "Add missing defaults retries=3 and debug=false to settings.json without overwriting existing values. Keep the file two-space-indented with sorted keys.",
              {"settings.json": "{\"debug\": true, \"name\": \"worker\"}\n"},
              "python3 -c \"import json;p='settings.json';d=json.load(open(p));d.setdefault('retries',3);d.setdefault('debug',False);open(p,'w').write(json.dumps(d,sort_keys=True,indent=2)+'\\n')\"",
              "printf '{\"debug\":false,\"retries\":3}\\n' > settings.json", "medium", "data",
              {"settings.json": "{\n  \"debug\": true,\n  \"name\": \"worker\",\n  \"retries\": 3\n}\n"}),
    _new_task("term/29-dedupe-latest",
              "records.jsonl contains id/value objects. Deduplicate by id, keeping the LAST occurrence, and write latest.jsonl sorted by numeric id with compact sorted-key JSON.",
              {"records.jsonl": "{\"id\":2,\"value\":\"old\"}\n{\"id\":1,\"value\":\"one\"}\n{\"id\":2,\"value\":\"new\"}\n"},
              "python3 -c \"import json;d={};[(d.__setitem__(x['id'],x)) for x in map(json.loads,open('records.jsonl'))];open('latest.jsonl','w').write(''.join(json.dumps(d[k],sort_keys=True,separators=(',',':'))+'\\n' for k in sorted(d)))\"",
              "sort -u records.jsonl > latest.jsonl", "hard", "data",
              {"latest.jsonl": "{\"id\":1,\"value\":\"one\"}\n{\"id\":2,\"value\":\"new\"}\n"}),
    _new_task("term/30-csv-jsonl",
              "Convert inventory.csv to inventory.jsonl with keys name, qty and price. qty must be an integer, price a number; preserve order and use compact sorted-key JSON.",
              {"inventory.csv": "name,qty,price\npen,4,1.25\nbook,2,8\n"},
              "python3 -c \"import csv,json;rows=[{'name':r['name'],'qty':int(r['qty']),'price':float(r['price'])} for r in csv.DictReader(open('inventory.csv'))];open('inventory.jsonl','w').write(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\\n' for r in rows))\"",
              "cp inventory.csv inventory.jsonl", "medium", "data",
              {"inventory.jsonl": "{\"name\":\"pen\",\"price\":1.25,\"qty\":4}\n{\"name\":\"book\",\"price\":8.0,\"qty\":2}\n"}),
    _new_task("term/31-join-report",
              "Join users.json and purchases.json on user id. Write report.json as a list sorted by user name with objects name and spent (sum of amount).",
              {"users.json": "[{\"id\":1,\"name\":\"mei\"},{\"id\":2,\"name\":\"bo\"}]\n",
               "purchases.json": "[{\"user\":1,\"amount\":3},{\"user\":2,\"amount\":5},{\"user\":1,\"amount\":2}]\n"},
              "python3 -c \"import json,collections;u=json.load(open('users.json'));p=json.load(open('purchases.json'));s=collections.defaultdict(int);[(s.__setitem__(x['user'],s[x['user']]+x['amount'])) for x in p];o=[{'name':x['name'],'spent':s[x['id']]} for x in sorted(u,key=lambda x:x['name'])];open('report.json','w').write(json.dumps(o,indent=2)+'\\n')\"",
              "cp purchases.json report.json", "hard", "data",
              {"report.json": "[\n  {\n    \"name\": \"bo\",\n    \"spent\": 5\n  },\n  {\n    \"name\": \"mei\",\n    \"spent\": 5\n  }\n]\n"}),
])


_GIT_SETUP = (
    "git init -q && git config user.email fixture@example.invalid && "
    "git config user.name Fixture && "
    "printf 'alpha\\n' > app.txt && git add app.txt && git commit -qm initial && "
    "printf 'beta\\n' >> app.txt && git commit -qam 'add beta' && "
    "printf 'secret\\n' > token.txt && git add token.txt && git commit -qm 'add token' && "
    "git rm -q token.txt && git commit -qm 'remove token' && "
    "printf 'gamma\\n' >> app.txt && git commit -qam 'add gamma'")

TASKS.extend([
    _new_task("term/32-latest-app-commit",
              "In the fixture git repository, write the subject of the newest commit that touched app.txt to answer.txt.",
              {}, "git log -1 --format=%s -- app.txt > answer.txt",
              "git log --skip=1 -1 --format=%s > answer.txt", "medium", "git",
              {"answer.txt": "add gamma\n"}, setup=_GIT_SETUP),
    _new_task("term/33-token-introduction",
              "Write the subject of the commit that FIRST ADDED token.txt to answer.txt, not the later commit that removed it.",
              {}, "git log --diff-filter=A --format=%s -- token.txt > answer.txt",
              "git log -1 --format=%s -- token.txt > answer.txt", "hard", "git",
              {"answer.txt": "add token\n"}, setup=_GIT_SETUP),
    _new_task("term/34-initial-content",
              "Recover app.txt as it existed in the repository's root (initial) commit into initial.txt.",
              {}, "git show $(git rev-list --max-parents=0 HEAD):app.txt > initial.txt",
              "cat app.txt > initial.txt", "hard", "git",
              {"initial.txt": "alpha\n"}, setup=_GIT_SETUP),
    _new_task("term/35-app-commit-count",
              "Write the number of commits that changed app.txt to count.txt.",
              {}, "git rev-list --count HEAD -- app.txt > count.txt",
              "git rev-list --count HEAD > count.txt", "medium", "git",
              {"count.txt": "3\n"}, setup=_GIT_SETUP),
    _new_task("term/36-recover-deleted",
              "Recover the contents of token.txt from the commit that introduced it into recovered.txt. Do not restore it to the worktree.",
              {}, "git show $(git log --diff-filter=A --format=%H -- token.txt):token.txt > recovered.txt",
              "cat token.txt > recovered.txt", "hard", "git",
              verify=_v_exact({"recovered.txt": "secret\n"}, absent=("token.txt",)),
              setup=_GIT_SETUP),
    _new_task("term/37-token-commit-files",
              "Write the names of files changed by the commit whose subject is 'add token' to files.txt, one per line.",
              {}, "git show --pretty= --name-only $(git log --format=%H --grep='^add token$') > files.txt",
              "git show --pretty= --name-only HEAD > files.txt", "hard", "git",
              {"files.txt": "token.txt\n"}, setup=_GIT_SETUP),
])


TASKS.extend([
    _new_task("term/38-cron-quarter-hour",
              "Create backup.cron containing one cron entry that runs /usr/local/bin/backup.sh every 15 minutes. No comments.",
              {}, "printf '*/15 * * * * /usr/local/bin/backup.sh\\n' > backup.cron",
              "printf '15 * * * * /usr/local/bin/backup.sh\\n' > backup.cron", "easy", "schedule",
              {"backup.cron": "*/15 * * * * /usr/local/bin/backup.sh\n"}),
    _new_task("term/39-cron-environment",
              "Create metrics.cron with SHELL=/bin/bash, PATH=/usr/local/bin:/usr/bin:/bin, and a job at 02:05 daily that changes to /srv/app and runs /usr/bin/python3 collect.py. One setting or job per line, no comments.",
              {}, "printf 'SHELL=/bin/bash\\nPATH=/usr/local/bin:/usr/bin:/bin\\n5 2 * * * cd /srv/app && /usr/bin/python3 collect.py\\n' > metrics.cron",
              "printf '5 2 * * * python3 /srv/app/collect.py\\n' > metrics.cron", "medium", "schedule",
              {"metrics.cron": "SHELL=/bin/bash\nPATH=/usr/local/bin:/usr/bin:/bin\n5 2 * * * cd /srv/app && /usr/bin/python3 collect.py\n"}),
    _new_task("term/40-systemd-service",
              "Create worker.service for a simple service running /opt/worker/run.py with /usr/bin/python3, restarting on failure. Use [Unit] Description=Worker, [Service] ExecStart and Restart, and [Install] WantedBy=multi-user.target, in that order.",
              {}, "printf '[Unit]\\nDescription=Worker\\n[Service]\\nExecStart=/usr/bin/python3 /opt/worker/run.py\\nRestart=on-failure\\n[Install]\\nWantedBy=multi-user.target\\n' > worker.service",
              "printf '[Service]\\nExecStart=python3 /opt/worker/run.py\\n' > worker.service", "medium", "schedule",
              {"worker.service": "[Unit]\nDescription=Worker\n[Service]\nExecStart=/usr/bin/python3 /opt/worker/run.py\nRestart=on-failure\n[Install]\nWantedBy=multi-user.target\n"}),
    _new_task("term/41-systemd-timer",
              "Create cleanup.service with [Service] Type=oneshot and ExecStart=/usr/bin/find /tmp -type f -delete, and cleanup.timer with [Timer] OnCalendar=daily plus [Install] WantedBy=timers.target. Use the exact directive order given.",
              {}, ["printf '[Service]\\nType=oneshot\\nExecStart=/usr/bin/find /tmp -type f -delete\\n' > cleanup.service",
                   "printf '[Timer]\\nOnCalendar=daily\\n[Install]\\nWantedBy=timers.target\\n' > cleanup.timer"],
              "printf '[Timer]\\nOnCalendar=daily\\n' > cleanup.timer", "hard", "schedule",
              {"cleanup.service": "[Service]\nType=oneshot\nExecStart=/usr/bin/find /tmp -type f -delete\n",
               "cleanup.timer": "[Timer]\nOnCalendar=daily\n[Install]\nWantedBy=timers.target\n"}),
    _new_task("term/42-repair-cron",
              "jobs.cron has an invalid minute step */0. Change only that step so the cleanup job runs every 10 minutes; preserve the other job and comments exactly.",
              {"jobs.cron": "# maintenance\n0 3 * * * /usr/bin/backup\n*/0 * * * * /usr/bin/cleanup\n"},
              "sed -i 's#^\\*/0 \\* \\* \\* \\* /usr/bin/cleanup#*/10 * * * * /usr/bin/cleanup#' jobs.cron",
              "sed -i 's/\\*\\/0/*\\//g' jobs.cron", "easy", "schedule",
              {"jobs.cron": "# maintenance\n0 3 * * * /usr/bin/backup\n*/10 * * * * /usr/bin/cleanup\n"}),
    _new_task("term/43-service-hardening",
              "Add NoNewPrivileges=true and PrivateTmp=true to the [Service] section of api.service after Restart=always. Preserve all existing lines and order.",
              {"api.service": "[Unit]\nDescription=API\n[Service]\nExecStart=/usr/bin/api\nRestart=always\n[Install]\nWantedBy=multi-user.target\n"},
              "sed -i '/^Restart=always$/a NoNewPrivileges=true\\nPrivateTmp=true' api.service",
              "printf 'NoNewPrivileges=true\\nPrivateTmp=true\\n' >> api.service", "medium", "schedule",
              {"api.service": "[Unit]\nDescription=API\n[Service]\nExecStart=/usr/bin/api\nRestart=always\nNoNewPrivileges=true\nPrivateTmp=true\n[Install]\nWantedBy=multi-user.target\n"}),
])

_PARSE_CASES = [
    ("", {}), ("# comment", {}), ("a=1", {"a": "1"}),
    (" name = value ", {"name": "value"}),
    ("url=https://example.org?a=1", {"url": "https://example.org?a=1"}),
    ("broken", {}), ("=empty", {}),
]
_TEXT_CASES = [(("  Alpha   BETA ",), "alpha beta"), (("One\nTwo\tTHREE",), "one two three")]
_TEXT_CASES += [((s,), " ".join(s.lower().split()))
                for s in ["  MiXeD  case  ", "a  b  c", " single ", "\n"]]
_CLAMP_CASES = [((x, 2, 8), min(8, max(2, x))) for x in [-5, 0, 2, 5, 8, 20]]
_PRICE_CASES = [((prices, discount), round(sum(prices) * (1 - discount / 100), 2))
                for prices, discount in [([10, 20], 10), ([0.1, 0.2], 0),
                                         ([5.55, 4.45], 25), ([], 15)]]
_SUMMARY_CASES = [((values,), {"count": len(values), "total": sum(values),
                               "mean": (sum(values) / len(values) if values else 0)})
                  for values in [[], [1], [1, 2, 3], [-2, 4, 8]]]
_RECORDS = [{"id": " A1 ", "tags": ["Red", "blue", "red"]},
            {"id": "B2", "tags": []}, {"id": " C3 ", "tags": ["Z", "a", "A"]}]
_rng = random.Random(20261015)
_RECORDS += [{"id": f" R{i:02d} ",
              "tags": [_rng.choice(["Red", "BLUE", "z", "a"])
                       for _ in range(4)]} for i in range(5)]
_RECORD_CASES = [(rows, [{"id": x["id"].strip(),
                          "tags": sorted({t.lower() for t in x["tags"]})}
                         for x in rows]) for rows in [_RECORDS, _RECORDS[::-1], []]]

TASKS.extend([
    _new_task("term/44-normalize-refactor",
              "textutil.normalize(text) must lowercase text and collapse all whitespace runs to one space, with no leading/trailing space. Fix the implementation; public_test.py is only a smoke test.",
              {"textutil.py": "def normalize(text):\n    return text.strip()\n",
               "public_test.py": "from textutil import normalize\nassert normalize('  Alpha   BETA ') == 'alpha beta'\n"},
              "cat > textutil.py <<'PY'\ndef normalize(text):\n    return ' '.join(text.lower().split())\nPY",
              "printf 'def normalize(text):\\n    return \"alpha beta\"\\n' > textutil.py",
              "medium", "refactor", verify=_v_function_cases("textutil", "normalize", _TEXT_CASES),
              verification="seeded-invariant"),
    _new_task("term/45-clamp-refactor",
              "geometry.clamp(value, lo, hi) must return value bounded inclusively by lo and hi. Fix it without changing the public signature.",
              {"geometry.py": "def clamp(value, lo, hi):\n    return value\n",
               "public_test.py": "from geometry import clamp\nassert clamp(5, 2, 8) == 5\n"},
              "cat > geometry.py <<'PY'\ndef clamp(value, lo, hi):\n    return min(hi, max(lo, value))\nPY",
              "printf 'def clamp(value, lo, hi):\\n    return 5\\n' > geometry.py",
              "easy", "refactor", verify=_v_function_cases("geometry", "clamp", _CLAMP_CASES),
              verification="seeded-invariant"),
    _new_task("term/46-pricing-multifile",
              "pricing.apply(prices, discount) returns the rounded total after a percentage discount. api.total must delegate to pricing.apply; repair both modules if needed. Empty input totals 0.",
              {"pricing.py": "def apply(prices, discount):\n    return sum(prices)\n",
               "api.py": "def total(prices, discount):\n    return sum(prices)\n"},
              ["cat > pricing.py <<'PY'\ndef apply(prices, discount):\n    return round(sum(prices) * (1 - discount / 100), 2)\nPY",
               "cat > api.py <<'PY'\nfrom pricing import apply\n\ndef total(prices, discount):\n    return apply(prices, discount)\nPY"],
              "printf 'def total(prices, discount):\\n    return 27.0\\n' > api.py",
              "hard", "refactor", verify=_v_function_cases("api", "total", _PRICE_CASES),
              verification="seeded-invariant"),
    _new_task("term/47-parser-distribution",
              "parse.py reads a JSON list of strings from stdin and writes a JSON list of objects. Each non-comment string containing key=value becomes {key: value}; trim both, split only the first '=', ignore comments, malformed lines and empty keys.",
              {"parse.py": "import json,sys\nrows=json.load(sys.stdin)\njson.dump([dict([x.split('=')]) for x in rows],sys.stdout)\n"},
              "cat > parse.py <<'PY'\nimport json,sys\nout=[]\nfor line in json.load(sys.stdin):\n    line=line.strip()\n    if not line or line.startswith('#') or '=' not in line:\n        out.append({})\n        continue\n    k,v=line.split('=',1)\n    out.append({k.strip():v.strip()} if k.strip() else {})\njson.dump(out,sys.stdout)\nPY",
              "printf 'import json,sys\\njson.dump([{} for x in json.load(sys.stdin)],sys.stdout)\\n' > parse.py",
              "hard", "refactor", verify=_v_json_script("parse.py", _PARSE_CASES),
              verification="seeded-distribution"),
    _new_task("term/48-summary-multifile",
              "report.summarize(values) must return {count, total, mean}; empty input returns numeric zeros. Keep the calculation in metrics.py and the public wrapper in report.py.",
              {"metrics.py": "def calculate(values):\n    return {'count': len(values)}\n",
               "report.py": "from metrics import calculate\n\ndef summarize(values):\n    return calculate(values)\n"},
              "cat > metrics.py <<'PY'\ndef calculate(values):\n    total=sum(values)\n    return {'count':len(values),'total':total,'mean':total/len(values) if values else 0}\nPY",
              "printf 'def summarize(values):\\n    return {\"count\":len(values)}\\n' > report.py",
              "hard", "refactor", verify=_v_function_cases("report", "summarize", _SUMMARY_CASES),
              verification="seeded-invariant"),
    _new_task("term/49-record-normalization",
              "normalize.py reads a JSON list of {id,tags} records and writes the same-order list with whitespace-stripped ids and sorted unique lowercase tags. It must work for arbitrary lists, including empty ones.",
              {"normalize.py": "import json,sys\njson.dump(json.load(sys.stdin),sys.stdout)\n"},
              "cat > normalize.py <<'PY'\nimport json,sys\nrows=json.load(sys.stdin)\nout=[{'id':x['id'].strip(),'tags':sorted({t.lower() for t in x['tags']})} for x in rows]\njson.dump(out,sys.stdout)\nPY",
              "printf 'import json,sys\\njson.dump(json.load(sys.stdin),sys.stdout)\\n' > normalize.py",
              "hard", "refactor", verify=_v_json_script("normalize.py", _RECORD_CASES),
              verification="seeded-distribution"),
])


def load_tasks():
    ids = [t["id"] for t in TASKS]
    if len(set(ids)) != len(ids):
        raise TermBenchError("duplicate task id")
    if len(TASKS) != 50:
        raise TermBenchError(f"expected 50 tasks, got {len(TASKS)}")
    for t in TASKS:
        for key in ("id", "instruction", "files", "solution", "verify",
                    "trap", "tier", "family", "verification"):
            if key not in t:
                raise TermBenchError(f"{t.get('id', '?')}: missing {key!r}")
        if t["tier"] not in ("easy", "medium", "hard"):
            raise TermBenchError(f"{t['id']}: invalid tier")
        if t["verification"] not in ("exact", "seeded-invariant", "seeded-distribution"):
            raise TermBenchError(f"{t['id']}: invalid verification tier")
    return list(TASKS)


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
    for command in task.get("setup", ()):
        code, output = run_command(command, root)
        if code:
            raise TermBenchError(f"{task['id']}: fixture setup failed: {output}")


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
            "task_family": task.get("family"), "tier": task.get("tier"),
            "verification": task.get("verification"),
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
