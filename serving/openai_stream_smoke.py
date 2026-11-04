#!/usr/bin/env python3
"""POST one streaming chat completion and print a short summary.

Locks the client side of a later serving phase. An external gateway is
reached at a base URL that ends with ``/openai/v1`` or ``/v1``. This
client POSTs ``{base}/chat/completions`` with ``model`` (default
``helioslm``), ``stream: true``, and a short user prompt. The API key
is read from the ``SERVING_API_KEY`` environment variable and sent as
``Authorization: Bearer <key>``. The key is never printed or logged;
if it would appear in an error or in the printed text, it is replaced
with ``[redacted]``.

The response must be ``text/event-stream``: ``data: {...}`` lines ended
by ``data: [DONE]``. Text is the concatenation of
``choices[0].delta.content``. The process exits non-zero on HTTP
errors, a non-SSE content type, a malformed chunk, a missing ``[DONE]``
terminator, a chunk whose ``model`` is not the requested model, empty
output, or a timeout.

Run from the repository root (no extra dependencies)::

    python serving/openai_stream_smoke.py --base-url http://127.0.0.1:8000/openai/v1

``SERVING_API_KEY`` must already be set. This module does not contact a
host unless ``--base-url`` is passed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_KEY_ENV = "SERVING_API_KEY"
DEFAULT_MODEL = "helioslm"
DEFAULT_PROMPT = "Reply with one short sentence."
DEFAULT_TIMEOUT_S = 30.0
_SSE_MEDIA_TYPE = "text/event-stream"
_REDACTED = "[redacted]"


class StreamSmokeError(Exception):
    """Contract or transport failure. The message is safe to print."""


class StreamSummary:
    """Assembled stream plus the figures the CLI prints."""

    __slots__ = ("text", "chunks", "chars", "ttft_s", "total_s")

    def __init__(self, text, chunks, ttft_s, total_s):
        self.text = text
        self.chunks = chunks
        self.chars = len(text)
        self.ttft_s = ttft_s
        self.total_s = total_s


def _redact(text, key):
    if key and key in text:
        return text.replace(key, _REDACTED)
    return text


def _error(msg, key):
    return StreamSmokeError(_redact(msg, key))


def normalize_base_url(base_url):
    """Return ``base_url`` without a trailing slash, or raise.

    The value must be an ``http`` or ``https`` URL whose path ends with
    ``/openai/v1`` or ``/v1``. A trailing slash is stripped first.
    Userinfo (``user:password@``) is rejected and is not echoed.
    """
    if not isinstance(base_url, str):
        raise StreamSmokeError(
            "base URL must be an http(s) URL ending with "
            f"/openai/v1 or /v1 (got {base_url!r})"
        )
    shown = base_url
    url = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password:
        raise StreamSmokeError(
            "base URL must be an http(s) URL ending with "
            "/openai/v1 or /v1 and must not include userinfo"
        )
    suffix_ok = url.endswith("/openai/v1") or url.endswith("/v1")
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not suffix_ok:
        raise StreamSmokeError(
            "base URL must be an http(s) URL ending with "
            f"/openai/v1 or /v1 (got {shown!r})"
        )
    return url


def _delta_content(event):
    choices = event.get("choices", [])
    if not isinstance(choices, list):
        raise StreamSmokeError("malformed chunk: choices is not a list")
    if not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        raise StreamSmokeError("malformed chunk: choices[0] is not an object")
    delta = choice.get("delta", {})
    if delta is None:
        return ""
    if not isinstance(delta, dict):
        raise StreamSmokeError("malformed chunk: delta is not an object")
    if "content" not in delta or delta["content"] is None:
        return ""
    content = delta["content"]
    if not isinstance(content, str):
        raise StreamSmokeError("malformed chunk: delta.content is not a string")
    return content


def _check_model(event, model):
    if "model" not in event:
        got = "missing"
    elif event["model"] != model:
        got = repr(event["model"])
    else:
        return
    raise StreamSmokeError(
        f"wrong model in chunk: expected {model!r}, got {got}"
    )


def _read_sse(response, model, key, started):
    parts = []
    chunks = 0
    ttft = None
    saw_done = False
    while True:
        raw = response.readline()
        if raw == b"":
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            saw_done = True
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            snippet = data[:120]
            raise _error(f"malformed chunk: {snippet!r}", key) from None
        if not isinstance(event, dict):
            raise _error("malformed chunk: JSON value is not an object", key)
        try:
            _check_model(event, model)
            piece = _delta_content(event)
        except StreamSmokeError as exc:
            raise _error(str(exc), key) from None
        chunks += 1
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - started
            parts.append(piece)
    if not saw_done:
        raise _error("stream ended without data: [DONE]", key)
    return "".join(parts), chunks, ttft


def _timeout_error(timeout, key):
    return _error(f"timeout after {timeout:g}s waiting for the stream", key)


def _request_failed(exc, key):
    detail = f"{type(exc).__name__}: {exc}"
    return _error(f"request failed: {detail}", key)


def stream_chat(base_url, model=DEFAULT_MODEL, prompt=DEFAULT_PROMPT,
                timeout=DEFAULT_TIMEOUT_S):
    """POST one streaming chat completion and return a :class:`StreamSummary`.

    Reads ``SERVING_API_KEY`` from the environment. Raises
    :class:`StreamSmokeError` for every contract and transport failure
    listed in the module docstring. Does not print.
    """
    key = os.environ.get(API_KEY_ENV) or ""
    try:
        base = normalize_base_url(base_url)
    except StreamSmokeError as exc:
        raise _error(str(exc), key) from None
    if not key:
        raise StreamSmokeError(f"{API_KEY_ENV} is not set")
    if not isinstance(model, str) or not model:
        raise _error("model must be a non-empty string", key)
    if not isinstance(prompt, str):
        raise _error("prompt must be a string", key)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or timeout <= 0):
        raise _error(
            f"timeout must be a positive number of seconds (got {timeout!r})",
            key,
        )

    body = json.dumps({
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": _SSE_MEDIA_TYPE,
            "Authorization": "Bearer " + key,
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ctype = response.headers.get("Content-Type")
            media = (ctype or "").split(";", 1)[0].strip().lower()
            if media != _SSE_MEDIA_TYPE:
                shown = ctype if ctype else "none"
                raise _error(
                    "expected content type text/event-stream, got "
                    f"{shown}",
                    key,
                )
            text, chunks, ttft = _read_sse(response, model, key, started)
    except StreamSmokeError:
        raise
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except (OSError, http.client.HTTPException):
            detail = ""
        if len(detail) > 200:
            detail = detail[:200] + "..."
        reason = exc.reason if isinstance(exc.reason, str) else ""
        msg = f"HTTP {exc.code}"
        if reason:
            msg = f"{msg} {reason}"
        if detail:
            msg = f"{msg}: {detail}"
        raise _error(msg, key) from None
    except (TimeoutError, socket.timeout):
        raise _timeout_error(timeout, key) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise _timeout_error(timeout, key) from None
        raise _request_failed(reason, key) from None
    except (http.client.HTTPException, OSError) as exc:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            raise _timeout_error(timeout, key) from None
        raise _request_failed(exc, key) from None

    total = time.perf_counter() - started
    if text == "" or ttft is None:
        raise _error("empty output: assembled delta content is empty", key)
    return StreamSummary(text, chunks, ttft, total)


def format_summary(summary):
    """One-line summary: chunks, chars, time to first token, total time."""
    return (
        f"chunks={summary.chunks} chars={summary.chars} "
        f"time_to_first_token_s={summary.ttft_s:.4f} "
        f"total_time_s={summary.total_s:.4f}"
    )


def main(argv=None):
    """CLI entry. Returns 0 on success and 1 on a contract or transport error."""
    parser = argparse.ArgumentParser(
        prog="openai_stream_smoke",
        description=(
            "POST one streaming chat completion to an OpenAI-compatible "
            "base URL and print a short summary."
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="endpoint base ending with /openai/v1 or /v1",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="socket timeout in seconds (default %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        summary = stream_chat(
            args.base_url,
            model=args.model,
            prompt=args.prompt,
            timeout=args.timeout,
        )
    except StreamSmokeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    key = os.environ.get(API_KEY_ENV) or ""
    print(_redact(summary.text, key))
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
