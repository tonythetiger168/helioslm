"""T74 - v1.32: 3-pane UI structure."""
import sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _boot_context, HarnessHandler
from http.server import HTTPServer


def test_three_pane_html():
    ctx = _boot_context("full")
    HarnessHandler.ctx = ctx
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/chat", timeout=5).read().decode()
    assert "sidebar" in html
    assert "effects audit" in html.lower()
    assert "helios-harness" in html
    server.shutdown()
    print("PASS test_three_pane_html")


def test_v1_rejects_foreign_host_and_origin():
    """v1 shares v2's loopback Host / Origin checks (it used to bind
    0.0.0.0 with no checks on POST /chat)."""
    import urllib.error
    ctx = _boot_context("full")
    HarnessHandler.ctx = ctx
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        for req in (
            urllib.request.Request(base + "/chat",
                                   headers={"Host": "evil.example"}),
            urllib.request.Request(
                base + "/chat", data=b'{"message": "hi"}', method="POST",
                headers={"Content-Type": "text/plain",
                         "Origin": "https://evil.example"}),
        ):
            try:
                urllib.request.urlopen(req, timeout=5)
                raise AssertionError(f"accepted {req.get_method()} {req.headers}")
            except urllib.error.HTTPError as e:
                assert e.code == 403, e.code
    finally:
        server.shutdown()
        server.server_close()
    print("PASS test_v1_rejects_foreign_host_and_origin")


if __name__ == "__main__":
    test_three_pane_html()
    test_v1_rejects_foreign_host_and_origin()
    print("UI v2 test done")
