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
import json
import os
import urllib.request

from .kernel import Service


class LLMError(RuntimeError):
    pass


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
    res = {"content": out.get("content") or "", "tool_calls": calls,
           "usage": usage, "stream": list(out.get("stream") or [])}
    if out.get("reasoning"):
        res["reasoning"] = out["reasoning"]
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

    def call(self, request, scope=None):
        provider = request["header"]["config"].get("provider")
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise LLMError(f"no adapter for provider {provider!r}")
        frozen = copy.deepcopy(request)
        out = self.ctx.scoped(scope).waterfall(
            "llm/stream", frozen, final=lambda req: adapter.complete(req))
        return _normalize(out)


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


def render_transcript(request):
    """Flatten a request into the ##role## transcript a text model reads."""
    tools = request["header"].get("tools") or []
    lines = []
    if tools:
        lines.append("##tools##" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
    for m in request["messages"]:
        body = m.get("content", "")
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{"name": c["name"], "args": json.loads(c["arguments"] or "{}")}
                     for c in m["tool_calls"]]
            body = "@@tool@@" + json.dumps({"calls": calls}, ensure_ascii=False) + "@@end@@"
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
    ``reasoning`` both ways."""

    def __init__(self, base_url, api_key_env="OPENAI_API_KEY", timeout=120,
                 opener=None):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen

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

    def complete(self, request):
        headers = {"Content-Type": "application/json"}
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise LLMError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(self.to_wire(request)).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=data, method="POST",
            headers=headers)
        with self.opener(req, timeout=self.timeout) as fh:
            return self.from_wire(json.loads(fh.read().decode()))
