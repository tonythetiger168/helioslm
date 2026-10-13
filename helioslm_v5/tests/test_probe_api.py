"""T84 — M5: probe_api.py oracles against a MOCK OpenAI-compatible
endpoint (localhost http.server; no network, no key, no cost).

The 2026-10 refresh (v5.51) added named presets, token-usage cost
accounting, and the verification_tier field. The risky parts are the
parameter-degradation ladder and the honest-None confidence path —
both are exercised here against a live local HTTP double:

1. ladder: the mock 400s any request carrying `max_tokens`, forcing
   the probe down the degradation ladder to `max_completion_tokens`;
   logprobs in the response become a confidence; usage lands in the
   meter,
2. honesty: a mock that omits logprobs yields conf=None (recorded,
   never fabricated),
3. episode + CLI: a scripted mock plays a full tool-loop episode
   (calc -> finish) and a full main() run writes a summary JSON whose
   schema carries verification_tier / probe_generation / cost fields,
4. presets: the 2026-10 generation is present, and --list-models works
   against the mock.

Run from repo root: python3 helioslm_v5/tests/test_probe_api.py
"""
import json
import math
import os
import sys
import tempfile
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "examples"))
sys.path.insert(0, str(_REPO))

import probe_api  # noqa: E402

TOOL_CALC = ('@@tool@@{"calls":[{"name":"calc","args":{"expr":"2 + 3"}}]}'
             '@@end@@')
TOOL_FINISH_5 = ('@@tool@@{"calls":[{"name":"finish","args":{"answer":"5"}}]}'
                 '@@end@@')


class _MockHandler(BaseHTTPRequestHandler):
    """Behavior is driven by class attributes set by each test."""
    reject_max_tokens = False   # force the degradation ladder
    give_logprobs = True
    episode_script = False      # play calc-then-finish by transcript
    ladder_attempts = 0

    def log_message(self, *a):
        pass

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._reply({"data": [{"id": "gpt-6-astra"},
                                  {"id": "mock-mini"}]})
        else:
            self._reply({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(
            int(self.headers["Content-Length"])))
        if self.reject_max_tokens and "max_tokens" in body:
            type(self).ladder_attempts += 1
            self._reply({"error": {"message": "max_tokens unsupported",
                                   "code": "unsupported_parameter"}}, 400)
            return
        content = TOOL_FINISH_5
        if self.episode_script:
            msgs = body["messages"][0]["content"]
            # a REAL transcript step always sits on the prompt's LAST
            # line (the FEWSHOT example's "step 0: 5" is followed by
            # more text, so a naive substring check misfires)
            last_line = msgs.rstrip().split("\n")[-1]
            content = TOOL_FINISH_5 if last_line.startswith("step ") \
                else TOOL_CALC
        msg = {"role": "assistant", "content": content}
        choice = {"message": msg}
        if self.give_logprobs:
            choice["logprobs"] = {"content": [
                {"token": "@@", "logprob": math.log(0.9),
                 "top_logprobs": [{"token": "@@",
                                   "logprob": math.log(0.9)}]}]}
        self._reply({"choices": [choice],
                     "usage": {"prompt_tokens": 100,
                               "completion_tokens": 20}})


class _Mock:
    """Context manager: a live mock endpoint on 127.0.0.1."""

    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(_MockHandler, k, v)

    def __enter__(self):
        _MockHandler.ladder_attempts = 0
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()
        # restore defaults so tests never leak state into each other
        _MockHandler.reject_max_tokens = False
        _MockHandler.give_logprobs = True
        _MockHandler.episode_script = False


def test_ladder_logprobs_and_meter():
    with _Mock(reject_max_tokens=True) as base:
        meter = {}
        text, conf = probe_api.chat_once(base, "mock-key", "mock-model",
                                         "Task: test", max_tokens=50,
                                         meter=meter)
        assert text == TOOL_FINISH_5
        assert abs(conf - 0.9) < 1e-9, f"conf {conf} != 0.9"
        assert _MockHandler.ladder_attempts >= 1, \
            "ladder not exercised: mock saw no max_tokens rejection"
        assert meter == {"prompt_tokens": 100, "completion_tokens": 20,
                         "calls": 1}
    print("PASS test_ladder_logprobs_and_meter")


def test_no_logprobs_honest_none():
    with _Mock(give_logprobs=False) as base:
        text, conf = probe_api.chat_once(base, "mock-key", "mock-model",
                                         "Task: test", max_tokens=50)
        assert text == TOOL_FINISH_5
        assert conf is None, "logprob-less response must yield conf=None"
    print("PASS test_no_logprobs_honest_none")


def test_episode_against_mock():
    with _Mock(episode_script=True) as base:
        from types import SimpleNamespace
        from helioslm_v5.agent.envs import make_envs
        from helioslm_v5.agent.tools import build_default_registry
        args = SimpleNamespace(base=base, model="mock-model")
        reg, impls = build_default_registry()
        task = make_envs()[0].sample(__import__("random").Random(1))
        meter = {}
        final = probe_api.run_episode(args, "mock-key", task, reg, impls,
                                      meter=meter)
        assert final == "5", f"episode final {final!r}"
        assert meter["calls"] == 2  # calc step + finish step
    print("PASS test_episode_against_mock")


def test_cli_summary_schema():
    with _Mock(episode_script=True) as base:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe.json"
            os.environ["OPENAI_API_KEY"] = "mock-key"
            argv = sys.argv
            sys.argv = ["probe_api.py", "--preset", "astra",
                        "--base", base, "--tasks", "1", "--samples", "2",
                        "--out", str(out)]
            try:
                probe_api.main()
            finally:
                sys.argv = argv
            summary = json.loads(out.read_text())
    assert summary["verification_tier"] == "measured"
    assert summary["probe_generation"] == "2026-10"
    assert summary["prices_asof"] == "2026-10-13"
    assert summary["model"] == "gpt-6-astra"  # preset default kept
    assert summary["usage"]["calls"] > 0
    assert isinstance(summary["cost_usd"], float), \
        "usage present + priced preset => cost must be computed"
    # preset price math: astra = $10/$50 per 1M
    u = summary["usage"]
    want = (u["prompt_tokens"] * 10.0 + u["completion_tokens"] * 50.0) / 1e6
    assert abs(summary["cost_usd"] - round(want, 6)) < 1e-9
    assert summary["n"] == len(summary["results"]) > 0
    for r in summary["results"]:
        assert r["conf_type"] == "self_consistency"
        assert r["n_samples"] == 2
    print("PASS test_cli_summary_schema")


def test_presets_and_list_models():
    for name in ("astra", "sol", "opus-5.5", "sonnet-5.5", "argon",
                 "luna"):
        assert name in probe_api.PRESETS, f"preset {name} missing"
    assert probe_api.PRESETS["argon"]["price"] is None, \
        "unknown price must be None, never estimated"
    with _Mock() as base:
        ids = probe_api.list_models(base, "mock-key")
        assert "gpt-6-astra" in ids and "mock-mini" in ids
    print("PASS test_presets_and_list_models")


def main():
    test_ladder_logprobs_and_meter()
    test_no_logprobs_honest_none()
    test_episode_against_mock()
    test_cli_summary_schema()
    test_presets_and_list_models()
    print("T84 PASS: 5/5")


if __name__ == "__main__":
    main()
