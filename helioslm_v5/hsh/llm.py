"""hsh H4: the model adapter seam (``ctx.llm``).

After dsh's ``llm/llm`` package: one service owns the message vocabulary
and the adapter registry; a provider is added by registering an adapter,
and every call passes the ``llm/stream`` waterfall so plugins can wrap it
(metrics, caching, fault injection) without touching the loop.

Request (built from the session log, see session.build_request)::

    {"header": {"config": {"provider": ..., "model": ..., ...},
                "tools": [schema, ...]},
     "messages": [{"role": "system" | "user" | "assistant" | "tool", ...}]}

Adapter output (one assistant message)::

    {"content": str, "reasoning": str | None,
     "tool_calls": [{"id", "name", "arguments": <raw JSON string>}],
     "usage": {"input_tokens", "output_tokens", "reasoning_tokens"},
     "stream": [chunk, ...]}            # what was streamed, for the log

Adapters here:

- ``ScriptedAdapter``: replies from a list or a function (tests, demos);
- ``TextModelAdapter``: any ``generate(prompt) -> str`` model, such as a
  HeliosLM checkpoint, speaking the repo's ``@@tool@@`` chat protocol
  (agent/schema.py);
- ``OpenAICompatAdapter``: an OpenAI-compatible HTTP endpoint (DeepSeek,
  vLLM, SGLang, a local server). The key comes from an environment
  variable and is never logged.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request

from .kernel import Service


class LLMError(RuntimeError):
    """A failed model call. ``status`` is the HTTP status when there was
    one; ``transient`` marks failures worth retrying (rate limits,
    overload, dropped connections); ``retry_after`` is the server's hint
    in seconds."""

    def __init__(self, message, status=None, transient=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.transient = (status in (408, 409, 425, 429) or (status or 0) >= 500
                          if transient is None else transient)
        self.retry_after = retry_after


def _normalize(out):
    if not isinstance(out, dict):
        raise LLMError("adapter must return a dict")
    calls = []
    for i, c in enumerate(out.get("tool_calls") or []):
        args = c.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        calls.append({"id": c.get("id") or f"call_{i}", "name": c["name"],
                      "arguments": args})
    usage = {k: int((out.get("usage") or {}).get(k, 0))
             for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
    cached = int((out.get("usage") or {}).get("cached_input_tokens") or 0)
    if cached:
        usage["cached_input_tokens"] = cached
    res = {"content": out.get("content") or "", "tool_calls": calls,
           "usage": usage, "stream": list(out.get("stream") or [])}
    if out.get("reasoning"):
        res["reasoning"] = out["reasoning"]
    if out.get("finish_reason"):
        res["finish_reason"] = str(out["finish_reason"])
    return json.loads(json.dumps(res, allow_nan=False))


class LLM(Service):
    key = "llm"

    def start(self):
        self.ctx.declare("llm/stream", "waterfall")
        self._adapters = {}

    def register(self, provider, adapter, ctx=None):
        ctx = ctx or self.ctx
        if provider in self._adapters:
            raise LLMError(f"provider {provider!r} already registered")
        self._adapters[provider] = adapter

        def dispose():
            if self._adapters.get(provider) is adapter:
                del self._adapters[provider]

        return ctx.effect(dispose)

    def providers(self):
        return sorted(self._adapters)

    def call(self, request, scope=None, on_chunk=None):
        """Run ``request`` through ``llm/stream`` and the adapter.

        ``on_chunk({"type": "text" | "reasoning", "text": ...})`` sees live
        output when the adapter streams (its ``complete`` takes an
        ``on_chunk`` keyword); other adapters just return the message."""
        provider = request["header"]["config"].get("provider")
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise LLMError(f"no adapter for provider {provider!r}", transient=False)
        frozen = copy.deepcopy(request)
        streams = on_chunk is not None and _takes_on_chunk(adapter.complete)

        def final(req):
            if streams:
                return adapter.complete(req, on_chunk=on_chunk)
            return adapter.complete(req)

        out = self.ctx.scoped(scope).waterfall("llm/stream", frozen, final=final)
        return _normalize(out)


def _takes_on_chunk(fn):
    try:
        return "on_chunk" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class ScriptedAdapter:
    """Replies in order from ``replies`` (dicts, or strings = plain text),
    or from ``fn(request, index)``. Records every request it received."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = fn
        self.requests = []

    def complete(self, request):
        i = len(self.requests)
        self.requests.append(copy.deepcopy(request))
        if self.fn is not None:
            out = self.fn(request, i)
        else:
            if i >= len(self.replies):
                raise LLMError("scripted adapter ran out of replies")
            out = self.replies[i]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, str):
            out = {"content": out}
        return out


ROLE_TAG = {"system": "##system##", "user": "##user##",
            "assistant": "##assistant##", "tool": "##tool##"}


def assistant_text(message):
    """What a text model writes for an assistant message: a ``@@tool@@``
    block when it calls tools, else its content. Argument keys are sorted,
    as ``TextModelAdapter`` stores them, so a transcript reads the same
    whether a call came from an API model or a text model (distillation
    trains a text model on an API model's transcripts)."""
    if message.get("tool_calls"):
        calls = [{"name": c["name"], "args": json.loads(c["arguments"] or "{}")}
                 for c in message["tool_calls"]]
        return "@@tool@@" + json.dumps({"calls": calls}, ensure_ascii=False,
                                       sort_keys=True) + "@@end@@"
    return message.get("content", "")


def render_transcript(request):
    """Flatten a request into the ##role## transcript a text model reads."""
    tools = request["header"].get("tools") or []
    lines = []
    if tools:
        lines.append("##tools##" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
    for m in request["messages"]:
        body = m.get("content", "")
        if m["role"] == "assistant":
            body = assistant_text(m)
        lines.append(ROLE_TAG[m["role"]] + body)
    lines.append(ROLE_TAG["assistant"])
    return "\n".join(lines)


class TextModelAdapter:
    """Wrap ``generate(prompt) -> str``. A ``@@tool@@`` block becomes tool
    calls; anything else is a text reply. A malformed block is returned as
    text so the loop logs it and the model sees its own mistake (the repo's
    "never swallowed silently" rule), instead of crashing the turn."""

    def __init__(self, generate, count_tokens=len):
        self.generate = generate
        self.count_tokens = count_tokens

    def complete(self, request):
        from ..agent.schema import ToolCallError, parse_chat_turn
        prompt = render_transcript(request)
        text = self.generate(prompt)
        usage = {"input_tokens": self.count_tokens(prompt),
                 "output_tokens": self.count_tokens(text)}
        try:
            kind, parsed = parse_chat_turn(text)
        except ToolCallError:
            return {"content": text, "usage": usage, "stream": [text]}
        if kind == "text":
            return {"content": parsed, "usage": usage, "stream": [text]}
        calls = [{"id": f"call_{i}", "name": c.name,
                  "arguments": json.dumps(c.args, ensure_ascii=False, sort_keys=True)}
                 for i, c in enumerate(parsed)]
        return {"content": "", "tool_calls": calls, "usage": usage, "stream": [text]}


class OpenAICompatAdapter:
    """POST ``{base_url}/chat/completions``. ``api_key_env`` names the
    environment variable holding the key; None sends no key (a self-hosted
    vLLM / SGLang server). ``reasoning_content`` (DeepSeek) is mapped to
    ``reasoning`` both ways. ``stream=True`` reads server-sent events and
    passes each text or reasoning delta to ``on_chunk`` as it arrives;
    the logged ``stream`` keeps the deltas merged by kind. HTTP failures
    raise :class:`LLMError` with the status (429 and 5xx are transient)."""

    def __init__(self, base_url, api_key_env="OPENAI_API_KEY", timeout=120,
                 opener=None, stream=False):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.stream = stream

    @staticmethod
    def to_wire(request):
        cfg = dict(request["header"]["config"])
        cfg.pop("provider", None)
        msgs = []
        for m in request["messages"]:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["call_id"],
                             "content": m["content"]})
                continue
            w = {"role": m["role"], "content": m.get("content", "")}
            if m.get("reasoning"):
                w["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                w["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"],
                                                 "arguments": c["arguments"]}}
                                   for c in m["tool_calls"]]
            msgs.append(w)
        body = dict(cfg, messages=msgs)
        tools = request["header"].get("tools")
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        return body

    @staticmethod
    def from_wire(resp):
        choice = resp["choices"][0]["message"]
        calls = [{"id": c["id"], "name": c["function"]["name"],
                  "arguments": c["function"].get("arguments") or "{}"}
                 for c in choice.get("tool_calls") or []]
        u = resp.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        return {"content": choice.get("content") or "",
                # vLLM >= 0.10 names the field "reasoning"
                "reasoning": choice.get("reasoning_content") or choice.get("reasoning"),
                "tool_calls": calls,
                "usage": {"input_tokens": u.get("prompt_tokens", 0),
                          "output_tokens": u.get("completion_tokens", 0),
                          "reasoning_tokens": details.get("reasoning_tokens", 0),
                          # DeepSeek API / vLLM spellings of cache hits
                          "cached_input_tokens": u.get("prompt_cache_hit_tokens") or
                          (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0},
                "finish_reason": resp["choices"][0].get("finish_reason"),
                "model": resp.get("model"),
                "stream": []}

    def complete(self, request, on_chunk=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMError(f"environment variable {self.api_key_env} is not set",
                               transient=False)
            headers["Authorization"] = f"Bearer {key}"
        body = self.to_wire(request)
        if self.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(),
            method="POST", headers=headers)
        try:
            with self.opener(req, timeout=self.timeout) as fh:
                if self.stream:
                    return self.from_sse(fh, on_chunk)
                return self.from_wire(json.loads(fh.read().decode()))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = ""
            ra = exc.headers.get("Retry-After") if exc.headers else None
            try:
                ra = float(ra) if ra is not None else None
            except ValueError:
                ra = None
            raise LLMError(f"HTTP {exc.code} from {self.base_url}: {detail}".rstrip(": "),
                           status=exc.code, retry_after=ra) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise LLMError(f"cannot reach {self.base_url}: {exc}", transient=True) from None

    @staticmethod
    def from_sse(lines, on_chunk=None):
        """Fold an OpenAI-style SSE stream into one adapter message."""
        content, reasoning, calls, stream = [], [], {}, []
        usage, finish, model = {}, None, None

        def emit(kind, text):
            if not text:
                return
            if stream and stream[-1]["type"] == kind:
                stream[-1]["text"] += text
            else:
                stream.append({"type": kind, "text": text})
            if on_chunk is not None:
                on_chunk({"type": kind, "text": text})

        for raw in lines:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("error"):
                raise LLMError(f"stream error: {ev['error']}", transient=True)
            model = ev.get("model") or model
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                r = d.get("reasoning_content") or d.get("reasoning")
                if r:
                    reasoning.append(r)
                    emit("reasoning", r)
                if d.get("content"):
                    content.append(d["content"])
                    emit("text", d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "",
                                                                 "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = ch.get("finish_reason") or finish
        details = usage.get("completion_tokens_details") or {}
        return {"content": "".join(content), "reasoning": "".join(reasoning) or None,
                "tool_calls": [{"id": c["id"] or f"call_{i}", "name": c["name"],
                                "arguments": c["arguments"] or "{}"}
                               for i, c in sorted(calls.items())],
                "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                          "output_tokens": usage.get("completion_tokens", 0),
                          "reasoning_tokens": details.get("reasoning_tokens", 0),
                          "cached_input_tokens": usage.get("prompt_cache_hit_tokens") or
                          (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0},
                "finish_reason": finish, "model": model, "stream": stream}


# ----- retrying transient failures --------------------------------------------

def is_transient(exc):
    if isinstance(exc, LLMError):
        return exc.transient
    return isinstance(exc, (ConnectionError, TimeoutError, socket.timeout,
                            urllib.error.URLError))


def request_retry(ctx, config=None):
    """Preset row ``request-retry``: retry transient model failures in the
    same step with exponential backoff and jitter (dsh's request-error
    recovery). Config: ``max_retries`` (4), ``base_s`` (1), ``max_s``
    (30); a server's ``Retry-After`` wins when it is longer. Every failed
    attempt is logged as ``assistant/attempt`` by the loop."""
    cfg = config or {}
    limit = cfg.get("max_retries", 4)
    base, cap = cfg.get("base_s", 1.0), cfg.get("max_s", 30.0)
    sleep = cfg.get("sleep", time.sleep)
    counts = {}

    def on_error(agent, exc, decision, nxt):
        decision = nxt(agent, exc, decision)
        if decision.get("action") == "retry" or not is_transient(exc):
            return decision
        step = next((e["seq"] for e in reversed(agent.session.events)
                     if e["type"] == "step/start"), -1)
        key = (id(agent), step)
        n = counts.get(key, 0)
        if n >= limit:
            return decision
        if len(counts) > 256:
            counts.clear()
        counts[key] = n + 1
        wait = min(cap, base * (2 ** n)) * (0.5 + random.random() / 2)
        if getattr(exc, "retry_after", None):
            wait = max(wait, min(float(exc.retry_after), cap * 2))
        sleep(wait)
        return {"action": "retry", "max_retries": limit}

    ctx.on("agent/request-error", on_error)


request_retry.inject = ["agents"]
