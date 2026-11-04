"""CPU contract tests for the OpenAI-compatible stream smoke client.

A stdlib HTTP server on 127.0.0.1 (ephemeral port) plays the SSE
response. Nothing here contacts an external host.

This sidecar binds a port, so it is listed in EXEMPT_SIDECARS for the
same reason as test_hsh_web and is not folded into the in-process daily
TESTS list. CI collects it with ``pytest helioslm_v5/tests/``.
"""

import contextlib
import io
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from serving.openai_stream_smoke import (  # noqa: E402
    API_KEY_ENV,
    DEFAULT_PROMPT,
    main,
    normalize_base_url,
    stream_chat,
)

_KEY = "smoke-test-key"


@contextlib.contextmanager
def _key(value=_KEY):
    old = os.environ.get(API_KEY_ENV)
    if value is None:
        os.environ.pop(API_KEY_ENV, None)
    else:
        os.environ[API_KEY_ENV] = value
    try:
        yield value
    finally:
        if old is None:
            os.environ.pop(API_KEY_ENV, None)
        else:
            os.environ[API_KEY_ENV] = old


def _chunk(model, delta, finish=None):
    event = {
        "id": "chatcmpl-smoke",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"


def _ok_body(model="helioslm", parts=("Hello", " stream")):
    lines = [_chunk(model, {"role": "assistant"})]
    for part in parts:
        lines.append(_chunk(model, {"content": part}))
    lines.append(_chunk(model, {}, finish="stop"))
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _write(handler, status, content_type, body):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


@contextlib.contextmanager
def _server(respond):
    captured = {}
    lock = threading.Lock()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(n)
            with lock:
                captured["path"] = self.path.split("?", 1)[0]
                captured["authorization"] = self.headers.get("Authorization")
                captured["content_type"] = self.headers.get("Content-Type")
                captured["body"] = json.loads(raw.decode("utf-8"))
            respond(self, release)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}", captured, lock
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


def _summary(out):
    line = out.strip().splitlines()[-1]
    return dict(part.split("=", 1) for part in line.split())


def _assert_request(captured, lock, path="/openai/v1/chat/completions"):
    with lock:
        assert captured["path"] == path
        assert captured["authorization"] == "Bearer " + _KEY
        assert captured["body"]["model"] == "helioslm"
        assert captured["body"]["stream"] is True
        assert _KEY not in json.dumps(captured["body"])


def test_local_stub_contract():
    def respond(handler, release):
        _write(handler, 200, "text/event-stream; charset=utf-8", _ok_body())

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1/",
            "--timeout", "2",
        ])
    assert code == 0, err
    assert err == ""
    assert _KEY not in out
    lines = out.splitlines()
    assert lines[0] == "Hello stream"
    fields = _summary(out)
    assert fields["chunks"] == "4"
    assert fields["chars"] == "12"
    ttft = float(fields["time_to_first_token_s"])
    total = float(fields["total_time_s"])
    assert total >= ttft >= 0.0
    _assert_request(captured, lock)
    with lock:
        assert captured["content_type"] == "application/json"
        assert captured["body"]["messages"] == [
            {"role": "user", "content": DEFAULT_PROMPT},
        ]


def test_v1_base_path():
    def respond(handler, release):
        _write(handler, 200, "text/event-stream", _ok_body())

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/v1",
            "--timeout", "2",
        ])
    assert code == 0, err
    assert out.splitlines()[0] == "Hello stream"
    _assert_request(captured, lock, path="/v1/chat/completions")


def test_printed_text_redacts_api_key():
    def respond(handler, release):
        _write(handler, 200, "text/event-stream", _ok_body(parts=(_KEY,)))

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 0, err
    assert _KEY not in out
    assert _KEY not in err
    assert out.splitlines()[0] == "[redacted]"
    _assert_request(captured, lock)


def test_missing_done():
    body = _chunk("helioslm", {"content": "Hi"}).encode("utf-8")

    def respond(handler, release):
        _write(handler, 200, "text/event-stream", body)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "data: [DONE]" in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_malformed_chunk():
    raw = (
        _chunk("helioslm", {"content": "Hi"})
        + "data: {not-json}\n\n"
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    def respond(handler, release):
        _write(handler, 200, "text/event-stream", raw)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "malformed chunk" in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_http_401():
    def respond(handler, release):
        payload = json.dumps({"error": "denied " + _KEY}).encode("utf-8")
        _write(handler, 401, "application/json", payload)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "HTTP 401" in err
    assert _KEY not in err
    assert _KEY not in out
    _assert_request(captured, lock)


def test_non_sse_content_type():
    def respond(handler, release):
        _write(
            handler, 200, "application/json",
            b'{"choices":[{"message":{"content":"hi-from-json"}}]}',
        )

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "text/event-stream" in err
    assert "application/json" in err
    assert "hi-from-json" not in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_empty_output():
    raw = (
        _chunk("helioslm", {"role": "assistant"})
        + _chunk("helioslm", {}, finish="stop")
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    def respond(handler, release):
        _write(handler, 200, "text/event-stream", raw)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "empty output" in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_wrong_model():
    raw = (
        _chunk("helioslm", {"content": "Hi"})
        + _chunk("other-model", {"content": "!"})
        + "data: [DONE]\n\n"
    ).encode("utf-8")

    def respond(handler, release):
        _write(handler, 200, "text/event-stream", raw)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert out == ""
    assert "wrong model" in err
    assert "other-model" in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_chunk_missing_model():
    event = {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}
    raw = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode("utf-8")

    def respond(handler, release):
        _write(handler, 200, "text/event-stream", raw)

    with _key(), _server(respond) as (base, captured, lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "2",
        ])
    assert code == 1
    assert "wrong model" in err
    assert "missing" in err
    assert _KEY not in err
    _assert_request(captured, lock)


def test_timeout():
    def respond(handler, release):
        release.wait(3)

    with _key(), _server(respond) as (base, _captured, _lock):
        code, out, err = _run([
            "--base-url", base + "/openai/v1",
            "--timeout", "0.4",
        ])
    assert code == 1, err
    assert out == ""
    assert "timeout" in err
    assert _KEY not in err


def test_missing_api_key_does_not_call_out():
    with _key(None):
        code, out, err = _run(["--base-url", "http://127.0.0.1:1/openai/v1"])
    assert code == 1
    assert out == ""
    assert f"{API_KEY_ENV} is not set" in err


def test_wrong_base_url_suffix():
    bad = [
        "http://127.0.0.1:9/v1/chat/completions",
        "http://127.0.0.1:9/openai",
        "http://127.0.0.1:9/api",
        "http://127.0.0.1:9/openai/v1extra",
        "http://127.0.0.1:9/V1",
        "/openai/v1",
    ]
    with _key():
        for url in bad:
            code, out, err = _run(["--base-url", url])
            assert code == 1, url
            assert out == "", url
            assert "ending with /openai/v1 or /v1" in err, (url, err)
            assert _KEY not in err


def test_bad_base_url_redacts_api_key():
    with _key():
        code, out, err = _run(["--base-url", "http://127.0.0.1:9/" + _KEY])
    assert code == 1
    assert out == ""
    assert "ending with /openai/v1 or /v1" in err
    assert _KEY not in err


def test_userinfo_rejected_without_echo():
    code, out, err = _run(["--base-url", "http://user:s3cret@127.0.0.1:9/v1"])
    assert code == 1
    assert out == ""
    assert "userinfo" in err
    assert "s3cret" not in err
    assert "user:s3cret" not in err


def test_base_url_suffixes_normalize():
    assert normalize_base_url("http://127.0.0.1:9/openai/v1") == \
        "http://127.0.0.1:9/openai/v1"
    assert normalize_base_url("http://127.0.0.1:9/openai/v1/") == \
        "http://127.0.0.1:9/openai/v1"
    assert normalize_base_url("https://127.0.0.1:9/v1") == \
        "https://127.0.0.1:9/v1"
    assert normalize_base_url("https://127.0.0.1:9/v1/") == \
        "https://127.0.0.1:9/v1"


def test_timeout_flag_must_be_positive():
    with _key():
        code, out, err = _run([
            "--base-url", "http://127.0.0.1:9/openai/v1",
            "--timeout", "0",
        ])
    assert code == 1
    assert out == ""
    assert "timeout" in err
    assert _KEY not in err


def test_ttft_is_measured_at_first_content():
    """Same-thread pause after the first content line.

    The sleep sits inside ``readline`` of the later chunk, after time to
    first token is recorded, so the gap does not depend on scheduling.
    """
    lines = [
        _chunk("helioslm", {"content": "Hello"}).encode("utf-8"),
        _chunk("helioslm", {"content": " stream"}).encode("utf-8"),
        b"data: [DONE]\n",
    ]

    class _Resp:
        def __init__(self):
            self.i = 0

        def readline(self):
            if self.i == 1:
                time.sleep(0.15)
            if self.i >= len(lines):
                return b""
            line = lines[self.i]
            self.i += 1
            return line

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        class _Headers:
            def get(self, name, default=None):
                if name.lower() == "content-type":
                    return "text/event-stream"
                return default

        headers = _Headers()

    def fake_urlopen(request, timeout=None):
        assert request.full_url.endswith("/openai/v1/chat/completions")
        assert request.get_header("Authorization") == "Bearer " + _KEY
        sent = json.loads(request.data.decode("utf-8"))
        assert sent["model"] == "helioslm"
        assert sent["stream"] is True
        return _Resp()

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        with _key():
            summary = stream_chat("http://127.0.0.1:9/openai/v1", timeout=2)
    finally:
        urllib.request.urlopen = original
    assert summary.text == "Hello stream"
    assert summary.chunks == 2
    assert summary.chars == 12
    assert summary.total_s >= summary.ttft_s + 0.1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"{len(fns)}/{len(fns)} openai stream smoke tests passed")
