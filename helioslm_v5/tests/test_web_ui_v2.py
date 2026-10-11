"""T75 - web UI v2: path containment, session store, SSE stream, headers.

Covers the v1.36 fixes that are server-side testable without a browser:
- _safe_media_path blocks ../ escapes (the v1.36 P0 path traversal)
- SessionStore: id validation, atomic JSON persistence, auto-title,
  param clamping, delete
- HTTP layer: security headers on every response, media traversal blocked
  over the wire, full session lifecycle, SSE chat stream ends by EOF
  (Connection: close) and reports confidence only when the backend does
  (never fabricated — the v1.36 P0).
"""
import json
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = tempfile.TemporaryDirectory()
# module creates ~/.helios_ui at import — redirect HOME just for the import,
# then restore so co-resident pytest tests see the real environment
_old_home = os.environ.get("HOME")
os.environ["HOME"] = _TMP.name
os.environ.pop("OPENAI_BASE", None)     # no API backend in tests
os.environ.pop("OPENAI_KEY", None)

from http.server import ThreadingHTTPServer  # noqa: E402

from harness import web_ui_v2           # noqa: E402

if _old_home is not None:
    os.environ["HOME"] = _old_home


def test_safe_media_path():
    with tempfile.TemporaryDirectory() as root:
        ok = web_ui_v2._safe_media_path(root, "a/b.png")
        real = os.path.realpath(root)
        assert ok and ok.startswith(real + os.sep)
        for bad in ("../x.png", "../../etc/passwd", "/etc/passwd", "",
                    "a\x00b"):
            assert web_ui_v2._safe_media_path(root, bad) is None, bad
    print("PASS test_safe_media_path")


def test_session_store():
    with tempfile.TemporaryDirectory() as d:
        st = web_ui_v2.SessionStore(d)
        s = st.create()
        assert re.fullmatch(r"[A-Za-z0-9_-]{8,64}", s["id"])
        # append: two messages land together; auto-title from first exchange
        st.append_messages(s["id"], {"role": "user", "content": "hello v2"},
                           {"role": "assistant", "content": "hi"})
        got = st.get(s["id"])
        assert got["title"] == "hello v2" and len(got["messages"]) == 2
        # params are clamped to sane ranges
        st.update(s["id"], params={"temperature": 9.9, "max_tokens": -5})
        p = st.get(s["id"])["params"]
        assert p["temperature"] == 2.0 and p["max_tokens"] == 1
        # id validation: traversal / garbage ids never touch the disk
        assert st._path("../x") is None and st.get("../x") is None
        assert st._path("ab") is None              # too short
        assert st._path("a" * 65) is None          # too long
        # listing survives a fresh store (plain-JSON persistence)
        st2 = web_ui_v2.SessionStore(d)
        assert len(st2.list()) == 1 and st2.list()[0]["n_messages"] == 2
        assert st.delete(s["id"]) is True and st.get(s["id"]) is None
        assert st.list() == []
    print("PASS test_session_store")


def _boot():
    web_ui_v2.BACKEND = web_ui_v2.SmokeBackend()   # deterministic
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui_v2.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _post(url, obj, method="POST"):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method=method)
    return urllib.request.urlopen(req, timeout=15)


def test_http_endpoints():
    server, port = _boot()
    base = f"http://127.0.0.1:{port}"
    try:
        # health + security headers on every response
        r = urllib.request.urlopen(base + "/api/health", timeout=5)
        h = json.loads(r.read())
        assert h["ok"] and h["backend"] == "smoke-echo"
        assert "default-src 'self'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        # chat page + retro dashboard both served
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "<textarea" in html or "chat" in html.lower()
        dash = urllib.request.urlopen(base + "/dashboard", timeout=5).read()
        assert len(dash) > 100
        # media path traversal blocked at the HTTP layer too
        for bad in ("/media/..%2F..%2Fetc%2Fpasswd",
                    "/media/../harness/web_ui_v2.py",
                    "/media/%2e%2e/%2e%2e/etc/passwd"):
            try:
                urllib.request.urlopen(base + bad, timeout=5)
                raise AssertionError("traversal not blocked: " + bad)
            except urllib.error.HTTPError as e:
                assert e.code in (403, 404), (bad, e.code)
        # session lifecycle over HTTP
        sid = json.loads(_post(base + "/api/sessions", {}).read())["session"]["id"]
        raw = _post(f"{base}/api/sessions/{sid}/chat/stream",
                    {"message": "ping v2"}).read().decode()
        # SSE frames: deltas arrive as 3-char chunks — reassemble them
        events = [json.loads(ln[5:]) for ln in raw.splitlines()
                  if ln.startswith("data:")]
        full = "".join(e.get("delta", "") for e in events)
        assert "[smoke] echo: ping v2" in full
        done = [e for e in events if e.get("done")]
        assert done and done[0]["tok_s"] > 0
        # honest confidence: smoke backend reports none -> null, not 0.95
        assert done[0]["confidence"] is None
        s = json.loads(urllib.request.urlopen(
            f"{base}/api/sessions/{sid}", timeout=5).read())["session"]
        assert len(s["messages"]) == 2 and s["title"] == "ping v2"
        _post(f"{base}/api/sessions/{sid}", {"title": "renamed"}, "PATCH")
        s = json.loads(urllib.request.urlopen(
            f"{base}/api/sessions/{sid}", timeout=5).read())["session"]
        assert s["title"] == "renamed"
        req = urllib.request.Request(f"{base}/api/sessions/{sid}",
                                     method="DELETE")
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["deleted"]
        # deleted session is a clean 404, not a traceback
        try:
            _post(f"{base}/api/sessions/{sid}/chat/stream", {"message": "x"})
            raise AssertionError("deleted session should 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
        server.server_close()
    print("PASS test_http_endpoints")


if __name__ == "__main__":
    test_safe_media_path()
    test_session_store()
    test_http_endpoints()
    print("\n3/3 web UI v2 tests passed")
