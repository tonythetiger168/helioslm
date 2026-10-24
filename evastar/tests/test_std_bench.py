"""T89 — roadmap Phase A: GSM8K / HumanEval standard-benchmark harness.

Pins:

1. data provenance: the committed subsets match MANIFEST.json and are
   the first n upstream records with the expected schema;
2. GSM8K answer extraction and number normalization oracles;
3. HumanEval program assembly (body / fenced function) and the sandbox:
   canonical solutions pass, a stub fails, an infinite loop times out,
   and a memory bomb is stopped by the address-space limit;
4. the committed reference artifact rebuilds byte-for-byte, all eleven
   gates hold, and it is registered as oracle-tested;
5. the API path end-to-end against a MOCK OpenAI-compatible server
   (no network, no key): the output is in the probe schema and scores
   on the calibration board; an API error aborts without writing output.

Run from repo root: python3 evastar/tests/test_std_bench.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from evastar.eval.calibration_board import score_artifact  # noqa: E402
from evastar.eval.std_bench import (  # noqa: E402
    BenchSchemaError, extract_gsm8k_answer, extract_program,
    gsm8k_prompt, humaneval_prompt, load_gsm8k, load_humaneval,
    normalize_number, run_humaneval_check)

_DATA = _REPO / "data" / "std_bench"
_GSM = _DATA / "gsm8k_test_first50.jsonl"
_HE = _DATA / "humaneval_first20.jsonl"
_ARTIFACT = _REPO / "benchmarks" / "std_bench_reference_2026-10-15.json"
_SCRIPT = _REPO / "examples" / "std_bench_run.py"


def test_data_provenance():
    r = subprocess.run([sys.executable, str(_REPO / "scripts" /
                                            "fetch_std_bench.py"), "--verify"],
                       cwd=_REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    gsm, he = load_gsm8k(_GSM), load_humaneval(_HE)
    assert len(gsm) == 50 and len(he) == 20
    assert gsm[0]["answer"] == "18"            # Janet's ducks
    assert he[0]["task_id"] == "HumanEval/0"
    assert he[-1]["task_id"] == "HumanEval/19"
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.jsonl"
        bad.write_text(json.dumps({"question": "q", "answer": "no marker"}))
        try:
            load_gsm8k(bad)
        except BenchSchemaError as e:
            assert "####" in str(e)
        else:
            raise AssertionError("gsm8k record without #### accepted")
    print("PASS test_data_provenance")


def test_gsm8k_extraction():
    assert normalize_number("1,234") == "1234"
    assert normalize_number("$18.00") == "18"
    assert normalize_number("-0.50") == "-0.5"
    assert normalize_number("none") is None
    assert extract_gsm8k_answer("3 + 4 = 7\n#### 7") == "7"
    # the marker wins over a later number
    assert extract_gsm8k_answer("#### 12\nchecked with 99 eggs") == "12"
    assert extract_gsm8k_answer("first 5, then 10, so 15.") == "15"
    assert extract_gsm8k_answer("no digits here") is None
    print("PASS test_gsm8k_extraction")


def test_humaneval_sandbox():
    task = load_humaneval(_HE)[0]
    body = task["canonical_solution"]
    assert run_humaneval_check(task, extract_program(body, task)) == \
        (True, "passed")
    fenced = "```python\n" + task["prompt"] + body + "```"
    assert run_humaneval_check(task, extract_program(fenced, task))[0]
    ok, detail = run_humaneval_check(
        task, extract_program("    return False\n", task))
    assert not ok and "AssertionError" in detail
    assert run_humaneval_check(
        task, extract_program("    while True: pass\n", task),
        timeout=2) == (False, "timeout")
    if os.name == "posix":
        ok, detail = run_humaneval_check(
            task, extract_program("    x = bytearray(4 << 30)\n", task))
        assert not ok and "MemoryError" in detail, detail
    print("PASS test_humaneval_sandbox")


def test_reference_artifact():
    report = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert len(report["gates"]) == 15   # 4 gsm8k + 4 humaneval + 3 terminal + 4 swe
    for name, g in report["gates"].items():
        assert g["accuracy"] == g["required"], name
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_ARTIFACT.name, reg) == "oracle-tested"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--reference",
                        "--check"], cwd=_REPO, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    print("PASS test_reference_artifact")


class _Mock(BaseHTTPRequestHandler):
    answers = {}
    fail = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.fail:   # non-transient: must abort at once, no retries
            self.send_response(401)
            self.end_headers()
            return
        text = self.answers[body["messages"][0]["content"]]
        out = json.dumps({"choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 10,
                                    "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _run_cli(port, out_dir, *extra):
    env = {**os.environ, "OPENAI_API_KEY": "mock"}
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--base", f"http://127.0.0.1:{port}/v1",
         "--model", "mock/model-x", "--limit", "4", "--samples", "2",
         "--out-dir", out_dir, *extra],
        cwd=_REPO, capture_output=True, text=True, timeout=300, env=env)


def test_api_path_with_mock():
    gsm, he = load_gsm8k(_GSM), load_humaneval(_HE)
    answers = {gsm8k_prompt(t): f"work\n#### {t['answer']}" for t in gsm[:2]}
    answers.update({gsm8k_prompt(t): "#### -1" for t in gsm[2:4]})
    answers.update({humaneval_prompt(t): t["canonical_solution"]
                    for t in he[:4]})
    _Mock.answers, _Mock.fail = answers, False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        with tempfile.TemporaryDirectory() as td:
            r = _run_cli(port, td, "--bench", "gsm8k", "humaneval")
            assert r.returncode == 0, r.stdout + r.stderr
            g = json.loads((Path(td) / "std_bench_gsm8k_model-x_2026-10-15.json")
                           .read_text())
            assert g["correct"] == "2/4" and g["samples"] == 2
            assert g["usage"]["calls"] == 8
            assert all(row["conf_type"] == "self_consistency"
                       for row in g["results"])
            h = json.loads((Path(td) / "std_bench_humaneval_model-x_2026-10-15.json")
                           .read_text())
            assert h["correct"] == "4/4" and h["pass_at_1"] == 1.0
            # probe schema: the run can go straight onto the board
            row = score_artifact(Path(td) / "std_bench_gsm8k_model-x_2026-10-15.json")
            assert row["accuracy"] == 0.5 and row["n_scored"] == 4

        _Mock.fail = True
        with tempfile.TemporaryDirectory() as td:
            r = _run_cli(port, td, "--bench", "gsm8k")
            assert r.returncode != 0 and "aborting" in r.stderr, r.stderr
            assert not list(Path(td).iterdir()), "partial run was written"
    finally:
        server.shutdown()
        server.server_close()
    print("PASS test_api_path_with_mock")


if __name__ == "__main__":
    test_data_provenance()
    test_gsm8k_extraction()
    test_humaneval_sandbox()
    test_reference_artifact()
    test_api_path_with_mock()
