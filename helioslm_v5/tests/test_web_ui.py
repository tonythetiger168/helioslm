"""T65 - v1.23: Web UI."""
import sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import PresetPlugin


def test_web_ui_endpoints():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.effect("test", {"hello": "world"})
    from harness.web_ui import HarnessHandler, main
    HarnessHandler.ctx = ctx
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    r = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/effects", timeout=5).read())
    assert r["n"] == 1
    r2 = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/services", timeout=5).read())
    assert "tools.registry" in r2["services"]
    server.shutdown()
    print("PASS test_web_ui_endpoints")


if __name__ == "__main__":
    import json
    test_web_ui_endpoints()
    print("web UI test done")
