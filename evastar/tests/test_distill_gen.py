"""D1 (v5.111): teacher data generation for distillation
(evastar/src/training/distill_gen.py).

Pins:
1. licence gate: the pinned DeepSeek-V4.1-Flash row passes; an unknown
   alias and a teacher whose hash is PENDING are refused;
2. request: vLLM thinking switch, served model name, keep_thinking drops
   or keeps earlier assistant reasoning;
3. run: one record per (prompt, sample) with reasoning, usage and
   finish_reason; resume skips done ids; transient errors retried with
   backoff, permanent ones logged to .errors.jsonl and retried next run;
   a torn last line from a killed run is ignored;
4. limits: output-token and cost limits stop submitting, in-flight work
   is still written; per-hour cost uses the clock;
5. token_stats: totals, percentiles, truncation, reasoning share, the
   character-share estimate when the server reports no reasoning tokens;
6. meta prompts -> extract_prompts round trip with duplicate removal;
7. estimate: per-token and per-hour arithmetic;
8. from_wire reads vLLM's "reasoning" field and both cache-hit spellings.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.eval import teacher_license_audit as tla  # noqa: E402
from evastar.hsh.llm import OpenAICompatAdapter  # noqa: E402
from evastar.src.training import distill_gen as dg  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fake_teacher(log=None, fail=None, clock=None, step=1.0):
    """complete(request) answering 'A:<question>' with reasoning;
    ``fail(question, call_index)`` may return an exception to raise."""
    calls = []

    def complete(req):
        q = req["messages"][-1]["content"]
        calls.append(req)
        if log is not None:
            log.append(q)
        if fail is not None:
            e = fail(q, len(calls))
            if e is not None:
                raise e
        if clock is not None:
            clock.t += step
        return {"content": f"A:{q}", "reasoning": "think " * 3,
                "finish_reason": "length" if q.startswith("long") else "stop",
                "model": "served",
                "usage": {"input_tokens": 10, "output_tokens": 100,
                          "reasoning_tokens": 60, "cached_input_tokens": 4}}

    complete.calls = calls
    return complete


def cfg(**kw):
    kw.setdefault("workers", 1)
    return dg.GenConfig(**kw)


def test_licence_gate(monkeypatch):
    assert dg.check_teacher("deepseek-v4.1-flash")["repo"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    with pytest.raises(dg.TeacherNotAllowed):
        dg.check_teacher("gpt-x")
    monkeypatch.setitem(dg.TEACHERS, "v4-flash", {"repo": "deepseek-ai/DeepSeek-V4-Flash",
                                                  "served_model": "x"})
    # pinned 2026-10-23 (v5.131): the gate lets it through
    assert dg.check_teacher("v4-flash")["repo"] == "deepseek-ai/DeepSeek-V4-Flash"
    # a row whose hash is PENDING is refused
    monkeypatch.setitem(tla.REVIEWS, "deepseek-ai/DeepSeek-V4-Flash",
                        {**tla.REVIEWS["deepseek-ai/DeepSeek-V4-Flash"],
                         "reviewed_sha256": tla.PENDING})
    with pytest.raises(dg.TeacherNotAllowed):
        dg.check_teacher("v4-flash")
    with pytest.raises(dg.TeacherNotAllowed):
        dg.Generator(cfg(teacher="v4-flash"), complete=fake_teacher())


def test_request_shape_and_keep_thinking():
    msgs = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning": "r1"},
            {"role": "user", "content": "q2"}]
    p = dg.normalize_prompt({"id": "p", "messages": msgs})
    g = dg.Generator(cfg(thinking=False), complete=fake_teacher())
    req = g.request(p)
    c = req["header"]["config"]
    assert c["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert c["chat_template_kwargs"] == {"thinking": False}
    assert "reasoning" not in req["messages"][1]
    assert msgs[1]["reasoning"] == "r1"                # caller's data untouched
    g_all = dg.Generator(cfg(keep_thinking="all"), complete=fake_teacher())
    assert g_all.request(p)["messages"][1]["reasoning"] == "r1"
    api = dg.Generator(cfg(endpoint="openai", model="deepseek-chat"), complete=fake_teacher())
    c = api.request(p)["header"]["config"]
    assert c["model"] == "deepseek-chat" and "chat_template_kwargs" not in c
    wire = OpenAICompatAdapter.to_wire(g.request(p))
    assert wire["chat_template_kwargs"] == {"thinking": False} and "provider" not in wire
    with pytest.raises(ValueError):
        dg.prompt_messages(msgs, "some")
    with pytest.raises(ValueError):
        dg.normalize_prompt({"messages": msgs[:2]})


def test_run_records_resume_and_errors(tmp_path):
    out = str(tmp_path / "gen.jsonl")
    prompts = [{"id": "a", "prompt": "qa", "domain": "math"},
               {"id": "b", "prompt": "qb"}, {"id": "c", "prompt": "bad"}]
    sleeps, tries = [], {}

    def fail(q, i):
        tries[q] = tries.get(q, 0) + 1
        if q == "qb" and tries[q] <= 2:              # b#0 fails twice, then works
            return urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO())
        if q == "bad":
            return urllib.error.HTTPError("u", 400, "bad request", {}, io.BytesIO())
        return None

    g = dg.Generator(cfg(samples=2, retries=3), complete=fake_teacher(fail=fail),
                     sleep=sleeps.append)
    s = g.run(prompts, out)
    assert s["written"] == 4 and s["errors"] == 2 and s["stop"] == "done"
    assert sleeps[:2] == [2.0, 4.0]                    # backoff doubles
    recs = dg.read_jsonl(out)
    assert sorted(r["id"] for r in recs) == ["a#0", "a#1", "b#0", "b#1"]
    r = next(r for r in recs if r["id"] == "a#0")
    assert r["content"] == "A:qa" and r["reasoning"] and r["finish_reason"] == "stop"
    assert r["domain"] == "math" and r["teacher"]["model"] == "served"
    assert r["usage"] == {"input_tokens": 10, "output_tokens": 100,
                          "reasoning_tokens": 60, "cached_input_tokens": 4}
    errs = dg.read_jsonl(out + ".errors.jsonl")
    assert {e["id"] for e in errs} == {"c#0", "c#1"} and errs[0]["attempts"] == 1

    with open(out, "a", encoding="utf-8") as fh:       # a killed run's torn line
        fh.write('{"id": "c#0", "conte')
    seen = []
    g2 = dg.Generator(cfg(samples=2), complete=fake_teacher(log=seen))
    s2 = g2.run(prompts, out)
    assert seen == ["bad", "bad"]                      # only the failed ones
    assert s2["skipped_done"] == 4


def test_limits_stop_submitting(tmp_path):
    prompts = [{"id": str(i), "prompt": f"q{i}"} for i in range(10)]
    g = dg.Generator(cfg(workers=2, max_output_tokens=250), complete=fake_teacher())
    s = g.run(prompts, str(tmp_path / "a.jsonl"))
    assert s["stop"] == "max_output_tokens"
    assert 3 <= s["written"] <= 4                      # in-flight still written
    assert s["written"] == len(dg.read_jsonl(str(tmp_path / "a.jsonl")))

    price = dg.Pricing(input_per_m=1e4, cached_input_per_m=0.0, output_per_m=1e4)
    assert price.token_cost({"input_tokens": 10, "cached_input_tokens": 4,
                             "output_tokens": 100}) == pytest.approx(1.06)
    g = dg.Generator(cfg(max_cost=2.0), complete=fake_teacher())
    s = g.run(prompts, str(tmp_path / "b.jsonl"), pricing=price)
    assert s["stop"] == "max_cost" and s["written"] == 2

    clock = Clock()
    hourly = dg.Pricing(per_hour=3600.0)               # 1 per second
    g = dg.Generator(cfg(max_cost=3.0), complete=fake_teacher(clock=clock), clock=clock)
    s = g.run(prompts, str(tmp_path / "c.jsonl"), pricing=hourly)
    assert s["stop"] == "max_cost" and s["written"] == 3 and s["cost"] == 3.0
    clock = Clock()
    g = dg.Generator(cfg(max_seconds=2.0), complete=fake_teacher(clock=clock), clock=clock)
    assert g.run(prompts, str(tmp_path / "d.jsonl"))["written"] == 2


def test_token_stats():
    def rec(dom, out, reas, fin, reasoning="", content=""):
        return {"domain": dom, "finish_reason": fin, "reasoning": reasoning,
                "content": content,
                "usage": {"input_tokens": 5, "output_tokens": out, "reasoning_tokens": reas}}
    recs = [rec("math", 100, 80, "stop"), rec("math", 300, 200, "length"),
            rec("chat", 50, 0, "stop", reasoning="x" * 30, content="y" * 70),
            rec("chat", 10, 0, "stop", content="hi")]
    st = dg.token_stats(recs)
    a = st["all"]
    assert a["n"] == 4 and a["output_tokens"] == 460 and a["input_tokens"] == 20
    assert a["reasoning_tokens"] == 80 + 200 + 15 and a["reasoning_estimated"] == 1
    assert a["output_p50"] == 50 and a["output_p95"] == 300 and a["output_max"] == 300
    assert a["truncated_rate"] == 0.25 and a["finish_reason"] == {"stop": 3, "length": 1}
    assert st["by_domain"]["math"]["reasoning_share"] == pytest.approx(280 / 400)
    assert dg.token_stats([])["all"]["n"] == 0


def test_meta_prompts_round_trip():
    seeds = [{"domain": "math", "topic": "台灣高中數學：機率", "n": 25}]
    metas = dg.meta_prompts(seeds, per_call=10)
    assert len(metas) == 3 and len({m["id"] for m in metas}) == 3
    assert "台灣繁體中文" in metas[0]["messages"][0]["content"]
    answers = [{"meta": metas[0]["meta"],
                "content": '好的：\n["擲兩顆骰子點數和為 7 的機率？", "短", 3, '
                           '"擲兩顆骰子點數和為 7 的機率？"]'},
               {"meta": metas[1]["meta"], "content": "抱歉，無法提供"}]
    prompts, st = dg.extract_prompts(answers)
    assert [p["domain"] for p in prompts] == ["math"]
    assert st == {"answers": 2, "unparsed": 1, "items": 4, "kept": 1,
                  "duplicates": 1, "bad_length": 2}
    assert dg.normalize_prompt(prompts[0])["id"] == prompts[0]["id"]


def test_estimate():
    api = dg.Pricing(currency="CNY", input_per_m=1.0, output_per_m=4.0)
    e = dg.estimate(20000, 300, 2000, api)
    assert e["cost"] == pytest.approx(6 + 160) and e["hours"] == 0
    pod = dg.Pricing(per_hour=15.0)
    e = dg.estimate(20000, 300, 2000, pod, output_tokens_per_s=4000, setup_hours=1.0)
    assert e["hours"] == pytest.approx(40e6 / 4000 / 3600 + 1, abs=0.01)
    assert e["cost"] == pytest.approx(15 * (40e6 / 4000 / 3600 + 1), abs=0.01)
    with pytest.raises(ValueError):
        dg.estimate(1, 1, 1, pod)


def test_from_wire_vllm_spellings():
    vllm = {"model": "m", "choices": [{"finish_reason": "length",
                                       "message": {"content": "x", "reasoning": "r"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": 5}}}
    out = OpenAICompatAdapter.from_wire(vllm)
    assert out["reasoning"] == "r" and out["finish_reason"] == "length"
    assert out["usage"]["cached_input_tokens"] == 5 and out["model"] == "m"
    ds = {"choices": [{"message": {"content": "x"}}],
          "usage": {"prompt_tokens": 9, "completion_tokens": 3,
                    "prompt_cache_hit_tokens": 7}}
    assert OpenAICompatAdapter.from_wire(ds)["usage"]["cached_input_tokens"] == 7


def test_keyless_adapter_sends_no_auth():
    seen = {}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def opener(req, timeout):
        seen["auth"] = req.get_header("Authorization")
        return R()

    ad = OpenAICompatAdapter("http://localhost:8000/v1", api_key_env=None, opener=opener)
    req = {"header": {"config": {"model": "m"}}, "messages": [{"role": "user", "content": "q"}]}
    assert ad.complete(req)["content"] == "ok" and seen["auth"] is None


def test_cli_against_a_local_openai_server(tmp_path, capsys):
    """seed -> generate (meta) -> extract -> generate -> stats through the
    real HTTP adapter, against a stand-in vLLM server on localhost."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import distill_generate as cli

    bodies = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            q = body["messages"][-1]["content"]
            if "JSON 字串陣列" in q:
                content = json.dumps([f"題目{len(bodies)}-{i}：請說明原因" for i in range(3)],
                                     ensure_ascii=False)
                reasoning = None
            else:
                content, reasoning = "答案", "先想一想"
            out = {"model": body["model"],
                   "choices": [{"finish_reason": "stop",
                                "message": {"content": content, "reasoning": reasoning}}],
                   "usage": {"prompt_tokens": 20, "completion_tokens": 8}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"_note": "x", "math": ["機率"], "code": ["排序"]},
                                ensure_ascii=False), encoding="utf-8")
    d = tmp_path / "run"
    try:
        cli.main(["seed", "--seeds", str(seeds), "--per-topic", "2", "--per-call", "2",
                  "--out", str(d / "meta.jsonl")])
        cli.main(["generate", "--prompts", str(d / "meta.jsonl"), "--out",
                  str(d / "meta_ans.jsonl"), "--base-url", url, "--no-thinking",
                  "--workers", "2"])
        assert all(b["chat_template_kwargs"] == {"thinking": False} for b in bodies)
        cli.main(["extract", "--answers", str(d / "meta_ans.jsonl"),
                  "--out", str(d / "prompts.jsonl")])
        prompts = dg.read_jsonl(str(d / "prompts.jsonl"))
        assert len(prompts) == 6 and {p["domain"] for p in prompts} == {"math", "code"}
        cli.main(["generate", "--prompts", str(d / "prompts.jsonl"), "--out",
                  str(d / "ans.jsonl"), "--base-url", url, "--workers", "3",
                  "--per-hour", "15"])
        capsys.readouterr()
        cli.main(["stats", "--answers", str(d / "ans.jsonl")])
        st = json.loads(capsys.readouterr().out)
        assert st["all"]["n"] == 6 and st["all"]["reasoning_estimated"] == 6
        assert bodies[-1]["chat_template_kwargs"] == {"thinking": True}
        assert bodies[-1]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    finally:
        srv.shutdown()
