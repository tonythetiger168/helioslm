"""hsh H4: the LLM seam and the turn/step driver (hsh/llm.py, hsh/loop.py).

Pins:
1. a tool-using turn logs the dsh event sequence, and every request the
   agent sent equals the request rebuilt from its log (model-visible
   means logged), across turns, retries, injected context and a system
   prompt change;
2. agent/pre-step can reject or rewrite input; agent/request rewrites
   the config (logged as a request/header change); agent/turn-stopping
   can continue the turn; max_steps stops a tool loop;
3. request errors: logged as assistant/attempt; retried only when a
   listener asks; otherwise the turn ends with reason "error";
4. scoped tools registered in the setup window are visible only to that
   agent; deferred tool contexts reach the next request;
5. the agents service waits for sessions, tools and llm (inject);
6. TextModelAdapter speaks the @@tool@@ protocol; OpenAICompatAdapter's
   wire mapping (reasoning_content, tool calls) round-trips without a
   network.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, LLMError,  # noqa: E402
                             OpenAICompatAdapter, ScriptedAdapter, Sessions,
                             TextModelAdapter, ToolDefinition, Tools,
                             rebuild_requests, render_transcript, request_hash)

CFG = {"provider": "fake", "model": "m"}
ADD = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
       "required": ["a", "b"], "additionalProperties": False}


def world(adapter):
    ctx = Context()
    agents = ctx.plugin(Agents)
    assert agents.state == "pending"  # waits for its services
    for svc in (Sessions, Tools, LLM):
        ctx.plugin(svc)
    assert agents.state == "active"
    ctx.llm.register("fake", adapter)
    ctx.tools.register(ToolDefinition("add", "Add.", ADD, lambda a, r: a["a"] + a["b"]))
    return ctx


def call(name, args, cid="c1"):
    return {"tool_calls": [{"id": cid, "name": name, "arguments": json.dumps(args)}]}


def assert_logged(agent):
    rebuilt = [request_hash(r) for _, r in
               rebuild_requests(agent.session.events, agent.keep_thinking)]
    assert agent.sent == rebuilt


def types(agent):
    return [e["type"] for e in agent.session.events]


def test_tool_turn_sequence_and_replay():
    ad = ScriptedAdapter([dict(call("add", {"a": 2, "b": 3}), reasoning="add them",
                               usage={"input_tokens": 10, "output_tokens": 3}),
                          {"content": "5", "usage": {"input_tokens": 12, "output_tokens": 1}},
                          "bye"])
    ctx = world(ad)
    agent = ctx.agents.create("s1", CFG, system_prompt="be brief")
    assert agent.send("2+3?") == "5"
    assert types(agent) == [
        "turn/start", "step/start", "system/message", "user/message", "request/header",
        "assistant/message", "tool/call", "tool/result", "step/end",
        "step/start", "assistant/message", "step/end", "turn/end"]
    tr = agent.session.last("tool/result")["data"]["message"]
    assert tr == {"role": "tool", "call_id": "c1", "content": "5"}
    # step 2 saw the tool result and (keep_thinking="turn") step 1's reasoning
    second = ad.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-2]["reasoning"] == "add them"
    assert ad.requests[0]["header"]["tools"][0]["name"] == "add"

    agent.system_prompt = "be very brief"
    agent.inject("note: user is in Taipei")
    assert agent.send("thanks") == "bye"
    third = ad.requests[2]["messages"]
    assert third[0] == {"role": "system", "content": "be very brief"}
    assert [m["content"] for m in third[-2:]] == ["note: user is in Taipei", "thanks"]
    assert all("reasoning" not in m for m in third)  # previous turn's reasoning dropped
    assert_logged(agent)


def test_pre_step_reject_and_rewrite():
    ad = ScriptedAdapter(["ok"])
    ctx = world(ad)
    agent = ctx.agents.create("s", CFG)

    def policy(a, d, nxt):
        if "forbidden" in d["messages"][-1]["content"]:
            return {"action": "reject"}
        d = {**d, "messages": [{**m, "content": m["content"].upper()} for m in d["messages"]]}
        return nxt(a, d)

    ctx.on("agent/pre-step", policy)
    assert agent.send("forbidden topic") is None
    assert types(agent) == ["turn/start", "turn/end"]
    assert agent.session.last("turn/end")["data"]["reason"] == "rejected"
    agent.send("hello")
    assert ad.requests[0]["messages"][-1]["content"] == "HELLO"
    assert_logged(agent)


def test_request_config_change_is_logged():
    ad = ScriptedAdapter(["a", "b"])
    ctx = world(ad)
    agent = ctx.agents.create("s", CFG)
    agent.send("one")
    ctx.on("agent/request", lambda a, c, nxt: nxt(a, {**c, "temperature": 0.2}))
    agent.send("two")
    headers = [e["data"] for e in agent.session.events if e["type"] == "request/header"]
    assert [h["reason"] for h in headers] == ["initial", "change"]
    assert ad.requests[1]["header"]["config"]["temperature"] == 0.2
    assert_logged(agent)


def test_turn_stopping_continues_and_max_steps():
    ad = ScriptedAdapter(["draft", "final"])
    ctx = world(ad)
    agent = ctx.agents.create("s", CFG)
    done = []

    def check(a, out):
        if not done:
            done.append(1)
            return "Double-check your answer."
        return None

    ctx.on("agent/turn-stopping", check)
    assert agent.send("q") == "final"
    users = [e["data"] for e in agent.session.events if e["type"] == "user/message"]
    assert users[-1] == {"content": "Double-check your answer.", "source": "turn-stopping"}

    looping = ScriptedAdapter(fn=lambda req, i: call("add", {"a": i, "b": 0}, cid=f"c{i}"))
    ctx2 = world(looping)
    a2 = ctx2.agents.create("s", CFG, max_steps=3)
    a2.send("loop forever")
    assert a2.session.last("turn/end")["data"]["reason"] == "max-steps"
    assert len(looping.requests) == 3
    assert_logged(agent)
    assert_logged(a2)


def test_request_error_retry_and_fail():
    ad = ScriptedAdapter([RuntimeError("503"), "recovered"])
    ctx = world(ad)
    agent = ctx.agents.create("s", CFG, max_retries=2)
    ctx.on("agent/request-error", lambda a, e, d, nxt: {"action": "retry"})
    assert agent.send("hi") == "recovered"
    att = agent.session.last("assistant/attempt")["data"]
    assert "503" in att["error"]
    assert len(agent.sent) == 2
    assert_logged(agent)

    ad2 = ScriptedAdapter([RuntimeError("boom")])
    ctx2 = world(ad2)
    a2 = ctx2.agents.create("s", CFG)  # no retry listener: fail
    assert a2.send("hi") is None
    assert a2.session.last("turn/end")["data"]["reason"] == "error"
    assert a2.session.open_turn() is None
    assert_logged(a2)


def test_scoped_tools_and_deferred_context():
    def body(args, run):
        run.defer_context("README.md changed on disk")
        return "read"

    ad = ScriptedAdapter([call("read", {}), "done", "other agent"])
    ctx = world(ad)

    def setup(agent):
        ctx.tools.register(ToolDefinition("read", "Read.", {"type": "object"}, body),
                           ctx=agent.ctx)

    a = ctx.agents.create("a", CFG, setup=setup)
    b = ctx.agents.create("b", CFG)
    a.send("go")
    second = ad.requests[1]["messages"]
    assert second[-1] == {"role": "user", "content": "README.md changed on disk"}
    b.send("hi")
    assert [t["name"] for t in ad.requests[2]["header"]["tools"]] == ["add"]
    assert {t["name"] for t in ad.requests[0]["header"]["tools"]} == {"add", "read"}
    assert_logged(a)
    assert_logged(b)


def test_unknown_provider_fails_turn():
    ctx = world(ScriptedAdapter([]))
    agent = ctx.agents.create("s", {"provider": "missing"})
    assert agent.send("hi") is None
    assert "no adapter" in agent.session.last("assistant/attempt")["data"]["error"]
    with pytest.raises(LLMError):
        ctx.llm.register("fake", ScriptedAdapter())


def test_llm_stream_waterfall_wraps_calls():
    ad = ScriptedAdapter(["raw"])
    ctx = world(ad)
    ctx.on("llm/stream", lambda req, nxt: {**nxt(req), "content": "wrapped"})
    agent = ctx.agents.create("s", CFG)
    assert agent.send("x") == "wrapped"


def test_text_model_adapter_protocol():
    replies = iter(['@@tool@@{"calls":[{"name":"add","args":{"a":1,"b":2}}]}@@end@@',
                    "3"])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(replies)

    ctx = world(TextModelAdapter(generate))
    agent = ctx.agents.create("s", CFG, system_prompt="sys")
    assert agent.send("1+2") == "3"
    assert agent.session.last("tool/result")["data"]["message"]["content"] == "3"
    assert prompts[1].splitlines()[-1] == "##assistant##"
    assert "##tool##3" in prompts[1] and "@@tool@@" in prompts[1]
    assert_logged(agent)


def test_text_model_malformed_block_is_text():
    ctx = world(TextModelAdapter(lambda p: "@@tool@@{broken"))
    agent = ctx.agents.create("s", CFG)
    assert agent.send("x") == "@@tool@@{broken"


def test_render_transcript_shape():
    req = {"header": {"config": CFG}, "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "name": "add", "arguments": '{"a":1,"b":2}'}]},
        {"role": "tool", "call_id": "c", "content": "3"}]}
    lines = render_transcript(req).splitlines()
    assert lines[0] == "##user##hi" and lines[1].startswith("##assistant##@@tool@@")
    assert lines[2] == "##tool##3" and lines[3] == "##assistant##"


def test_openai_wire_mapping_offline(monkeypatch):
    req = {"header": {"config": {"provider": "ds", "model": "deepseek-v4-flash"},
                      "tools": [{"name": "add", "description": "", "parameters": ADD}]},
           "messages": [{"role": "user", "content": "q"},
                        {"role": "assistant", "content": "", "reasoning": "think",
                         "tool_calls": [{"id": "c", "name": "add", "arguments": "{}"}]},
                        {"role": "tool", "call_id": "c", "content": "3"}]}
    wire = OpenAICompatAdapter.to_wire(req)
    assert "provider" not in wire and wire["model"] == "deepseek-v4-flash"
    assert wire["messages"][1]["reasoning_content"] == "think"
    assert wire["messages"][2] == {"role": "tool", "tool_call_id": "c", "content": "3"}
    assert wire["tools"][0]["type"] == "function"

    resp = {"choices": [{"message": {"content": None, "reasoning_content": "r",
                                     "tool_calls": [{"id": "x", "type": "function",
                                                     "function": {"name": "add",
                                                                  "arguments": '{"a":1}'}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 2}}}
    out = OpenAICompatAdapter.from_wire(resp)
    assert out["reasoning"] == "r" and out["tool_calls"][0]["name"] == "add"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 5, "reasoning_tokens": 2}

    class FakeResp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    seen = {}

    def opener(r, timeout):
        seen["auth"] = r.get_header("Authorization")
        seen["url"] = r.full_url
        return FakeResp(resp)

    monkeypatch.setenv("HSH_TEST_KEY", "sk-test")
    ad = OpenAICompatAdapter("https://example.invalid/v1/", api_key_env="HSH_TEST_KEY",
                             opener=opener)
    assert ad.complete(req)["usage"]["input_tokens"] == 7
    assert seen == {"auth": "Bearer sk-test",
                    "url": "https://example.invalid/v1/chat/completions"}
    monkeypatch.delenv("HSH_TEST_KEY")
    with pytest.raises(LLMError):
        ad.complete(req)
