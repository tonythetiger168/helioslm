"""T67 - v1.25: chat endpoint."""
import sys, threading, time, urllib.request, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _boot_context, HarnessHandler
from http.server import HTTPServer


def test_chat_endpoint():
    ctx = _boot_context("full")
    HarnessHandler.ctx = ctx
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    # dashboard has chat link
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
    assert "/chat" in html
    # chat page
    chat_html = urllib.request.urlopen(f"http://127.0.0.1:{port}/chat", timeout=5).read().decode()
    assert "helios-chat" in chat_html
    # POST a message
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/chat",
        data=json.dumps({"sid": "test1", "message": "hello"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    r = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert "response" in r and "confidence" in r
    assert r["effects"] >= 0
    server.shutdown()
    print("PASS test_chat_endpoint")


if __name__ == "__main__":
    test_chat_endpoint()
    print("chat test done")
