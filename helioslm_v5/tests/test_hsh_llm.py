"""hsh: streaming, HTTP errors and request retry (hsh/llm.py).

Pins:
1. OpenAICompatAdapter(stream=True) folds SSE deltas (text, reasoning,
   tool-call fragments by index, usage incl. cache hits) into one message,
   passes each delta to on_chunk, and the loop re-emits them live as
   agent/assistant-stream while the log keeps the merged stream;
2. HTTP 429/5xx become transient LLMErrors with Retry-After, 400 does not;
3. request-retry retries transient failures in the same step with
   backoff, gives up after max_retries, never retries a permanent error,
   and every attempt is in the log so requests still rebuild.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, LLMError,  # noqa: E402
                         OpenAICompatAdapter, ScriptedAdapter, Sessions, Tools,
                         is_transient, rebuild_requests, request_hash,
                         request_retry, usage_summary)

CFG = {"provider": "fake", "model": "m"}


def sse(*events):
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return io.BytesIO(body.encode())


STREAM = [
    {"model": "deepseek-v4-flash", "choices": [{"delta": {"reasoning_content": "Let me "}}]},
    {"choices": [{"delta": {"reasoning_content": "think."}}]},
    {"choices": [{"delta": {"content": "Hello"}}]},
    {"choices": [{"delta": {"content": " world"}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "add", "arguments": '{"a": '}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                  "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4,
                              "prompt_cache_hit_tokens": 6,
                              "completion_tokens_details": {"reasoning_tokens": 2}}},
]


def test_sse_fold():
    seen = []
    out = OpenAICompatAdapter.from_sse(sse(*STREAM), seen.append)
    assert out["content"] == "Hello world" and out["reasoning"] == "Let me think."
    assert out["tool_calls"] == [{"id": "c1", "name": "add", "arguments": '{"a": 1}'}]
    assert out["usage"] == {"input_tokens": 9, "output_tokens": 4, "reasoning_tokens": 2,
                            "cached_input_tokens": 6}
    assert out["finish_reason"] == "tool_calls" and out["model"] == "deepseek-v4-flash"
    assert out["stream"] == [{"type": "reasoning", "text": "Let me think."},
                             {"type": "text", "text": "Hello world"}]
    assert [c["text"] for c in seen] == ["Let me ", "think.", "Hello", " world"]


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_streaming_through_the_loop(monkeypatch):
    monkeypatch.setenv("HSH_TEST_KEY", "k")
    bodies = []

    def opener(req, timeout):
        bodies.append(json.loads(req.data))
        return Resp(sse(*STREAM[:4], {"choices": [{"delta": {}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                        ).getvalue())

    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.llm.register("ds", OpenAICompatAdapter("https://x.invalid", "HSH_TEST_KEY",
                                               opener=opener, stream=True))
    live = []
    ctx.on("agent/assistant-stream", lambda agent, chunk: live.append(chunk["text"]))
    a = ctx.agents.create("s", {"provider": "ds", "model": "m"})
    assert a.send("hi") == "Hello world"
    assert bodies[0]["stream"] is True and bodies[0]["stream_options"] == {"include_usage": True}
    assert "".join(live) == "Let me think.Hello world"
    msg = [e for e in a.session.events if e["type"] == "assistant/message"][0]["data"]
    assert msg["stream"][1] == {"type": "text", "text": "Hello world"}
    assert msg["message"]["reasoning"] == "Let me think."
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]


def test_http_errors(monkeypatch):
    monkeypatch.setenv("HSH_TEST_KEY", "k")

    def failing(code, headers=None):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "x", headers or {},
                                         io.BytesIO(b'{"error": "busy"}'))
        return opener

    ad = OpenAICompatAdapter("https://x.invalid", "HSH_TEST_KEY",
                             opener=failing(429, {"Retry-After": "3"}))
    req = {"header": {"config": {"provider": "ds", "model": "m"}}, "messages": []}
    with pytest.raises(LLMError) as e:
        ad.complete(req)
    assert e.value.status == 429 and e.value.transient and e.value.retry_after == 3.0
    assert "busy" in str(e.value)
    ad.opener = failing(400)
    with pytest.raises(LLMError) as e:
        ad.complete(req)
    assert not e.value.transient
    ad.opener = failing(503)
    with pytest.raises(LLMError) as e:
        ad.complete(req)
    assert is_transient(e.value)

    def down(req, timeout):
        raise urllib.error.URLError("connection refused")

    ad.opener = down
    with pytest.raises(LLMError) as e:
        ad.complete(req)
    assert e.value.transient
    assert not is_transient(ValueError("x"))


def world(replies, **retry):
    ad = ScriptedAdapter(replies)
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.llm.register("fake", ad)
    sleeps = []
    ctx.plugin(request_retry, dict({"sleep": sleeps.append}, **retry))
    return ctx, ad, sleeps


def test_request_retry():
    busy = LLMError("HTTP 429", status=429, retry_after=5)
    ctx, ad, sleeps = world([busy, LLMError("HTTP 502", status=502), "fine"])
    a = ctx.agents.create("s", CFG)
    assert a.send("hi") == "fine"
    assert len(sleeps) == 2 and sleeps[0] >= 5 and 0.5 <= sleeps[1] <= 2
    kinds = [e["type"] for e in a.session.events if e["type"].startswith("assistant/")]
    assert kinds == ["assistant/attempt", "assistant/attempt", "assistant/message"]
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]
    assert usage_summary(a.session.events)["total"]["attempts"] == 2
    # gives up after max_retries; a permanent error is not retried
    ctx, ad, sleeps = world([busy] * 5, max_retries=2)
    a = ctx.agents.create("s", CFG)
    assert a.send("hi") is None and len(sleeps) == 2 and len(ad.requests) == 3
    ctx, ad, sleeps = world([LLMError("HTTP 400", status=400), "never"])
    a = ctx.agents.create("s", CFG)
    assert a.send("hi") is None and sleeps == [] and len(ad.requests) == 1
    end = [e for e in a.session.events if e["type"] == "turn/end"][-1]
    assert end["data"]["reason"] == "error"
