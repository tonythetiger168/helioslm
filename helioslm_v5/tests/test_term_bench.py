"""T90 — roadmap Phase A: terminal-style sandbox tasks.

Pins:

1. every task starts UNSOLVED and its scripted reference solves it;
2. the verifiers reject plausible-but-wrong solutions (the traps each
   task is built around: a WARN line mentioning ERRORS, a 'northwest'
   region, a hard-coded calc.py, an archive of '.' instead of docs/ ...);
3. reply parsing: fenced command, DONE with/without confidence,
   out-of-range confidence dropped, '$ cmd' fallback, invalid reply;
4. the sandbox: cwd/HOME are the sandbox, timeouts, output truncation,
   file-size limit, and the verifiers never enter the sandbox (M1);
5. the API path end-to-end against a MOCK OpenAI-compatible server: a
   multi-turn episode runs, verbalized confidence is recorded, and the
   output scores on the calibration board.

Run from repo root: python3 helioslm_v5/tests/test_term_bench.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import term_bench as tb  # noqa: E402
from helioslm_v5.eval.calibration_board import score_artifact  # noqa: E402

_SCRIPT = _REPO / "examples" / "std_bench_run.py"
_TASKS = {t["id"]: t for t in tb.load_tasks()}

NAIVE = {
    "term/00-count-errors": "grep -c ERROR app.log > answer.txt",
    "term/02-unique-emails": "grep -oE '[a-zA-Z0-9.]+@[a-zA-Z.]+' "
                             "contacts.txt | sort -u > emails.txt",
    "term/03-fix-script": "printf 'def total(n):\\n    return 55\\n' > calc.py",
    "term/05-csv-sum": "awk -F, '$1 ~ /north/ {s+=$2} END {print s}' "
                       "sales.csv > total.txt",
    "term/07-replace-host": "sed -i 's/localhost/db.internal/g' config/*",
    "term/09-archive": "tar -czf backup.tar.gz .",
}


def _state_after(task, commands):
    with tempfile.TemporaryDirectory() as root:
        tb.seed_sandbox(task, root)
        for cmd in commands:
            tb.run_command(cmd, root)
        return task["verify"](root)


def test_tasks_start_unsolved_and_references_solve():
    assert len(_TASKS) == 10
    for tid, task in _TASKS.items():
        ok, detail = _state_after(task, [])
        assert not ok, f"{tid} is solved before any command: {detail}"
        ok, detail = _state_after(task, task["solution"])
        assert ok, f"{tid} reference failed: {detail}"
    print("PASS test_tasks_start_unsolved_and_references_solve")


def test_verifiers_reject_naive_solutions():
    for tid, cmd in NAIVE.items():
        ok, detail = _state_after(_TASKS[tid], [cmd])
        assert not ok, f"{tid} accepted a naive solution: {detail}"
    print("PASS test_verifiers_reject_naive_solutions")


def test_parse_reply():
    assert tb.parse_reply("ok\n```bash\nls -la\n```") == ("cmd", "ls -la")
    assert tb.parse_reply("```\nwc -l a\n```") == ("cmd", "wc -l a")
    assert tb.parse_reply("DONE confidence=0.75") == ("done", 0.75)
    assert tb.parse_reply("done") == ("done", None)
    assert tb.parse_reply("DONE confidence=7") == ("done", None)
    assert tb.parse_reply("$ cat app.log") == ("cmd", "cat app.log")
    assert tb.parse_reply("I would use awk.")[0] == "invalid"
    assert tb.parse_reply("")[0] == "invalid"
    print("PASS test_parse_reply")


def test_sandbox_limits():
    with tempfile.TemporaryDirectory() as root:
        real = str(Path(root).resolve())
        code, out = tb.run_command("pwd; echo $HOME", root)
        assert code == 0 and out.split() == [real, root], out
        assert tb.run_command("sleep 5", root, timeout=1)[0] == 124
        code, out = tb.run_command("yes | head -c 5000", root)
        assert len(out) < 2200 and "truncated" in out
        if os.name == "posix":
            code, _ = tb.run_command(
                "head -c 20000000 /dev/zero > big.bin", root)
            assert code != 0 and (Path(root) / "big.bin").stat().st_size \
                <= tb.CMD_FILE_BYTES
        # M1: nothing of the verifier is reachable from inside the sandbox
        tb.seed_sandbox(_TASKS["term/00-count-errors"], root)
        _, out = tb.run_command("grep -rl 'def _v_\\|verify' . || true", root)
        assert out.strip() == "", out
    print("PASS test_sandbox_limits")


class _Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        m = re.search(r"\[(term/[^,]+), step (\d+) of", prompt)
        tid, k = m.group(1), int(m.group(2)) - 1
        steps = _TASKS[tid]["solution"]
        if tid == "term/00-count-errors":      # a wrong-but-confident agent
            steps = [NAIVE[tid]]
        text = (f"```bash\n{steps[k]}\n```" if k < len(steps)
                else "DONE confidence=0.8")
        out = json.dumps({"choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 50,
                                    "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_api_path_with_mock():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [sys.executable, str(_SCRIPT), "--bench", "terminal",
                 "--base", f"http://127.0.0.1:{port}/v1",
                 "--model", "mock/agent", "--limit", "3", "--out-dir", td],
                cwd=_REPO, capture_output=True, text=True, timeout=300,
                env={**os.environ, "OPENAI_API_KEY": "mock"})
            assert r.returncode == 0, r.stdout + r.stderr
            path = Path(td) / "std_bench_terminal_agent_2026-10-15.json"
            out = json.loads(path.read_text())
            assert out["correct"] == "2/3", out["correct"]
            first = out["results"][0]
            assert not first["correct"] and first["ended"] == "done"
            assert all(row["conf"] == 0.8 and row["conf_type"] == "verbalized"
                       for row in out["results"])
            assert out["usage"]["calls"] == 2 + 2 + 2   # 1 cmd + DONE each
            row = score_artifact(path)
            assert row["overconfident_wrong"] == 0 and row["n_wrong_scored"] == 1
    finally:
        server.shutdown()
        server.server_close()
    print("PASS test_api_path_with_mock")


if __name__ == "__main__":
    test_tasks_start_unsolved_and_references_solve()
    test_verifiers_reject_naive_solutions()
    test_parse_reply()
    test_sandbox_limits()
    test_api_path_with_mock()
