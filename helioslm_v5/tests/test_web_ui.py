"""T66 - v1.24: web UI with preset + smoke workflow."""
import sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _boot_context, HarnessHandler
from http.server import HTTPServer
import json


def test_boot_full_preset():
    ctx = _boot_context("full")
    assert len(ctx.effects) >= 4   # boot + 3 smoke
    assert "tools.registry" in ctx._services
    print(f"PASS test_boot_full_preset ({len(ctx.effects)} effects)")


def test_html_dashboard():
    ctx = _boot_context("full")
    HarnessHandler.ctx = ctx
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
    assert "helios-harness" in html
    assert "Recent effects" in html
    r = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/effects", timeout=5).read())
    assert r["n"] >= 4
    assert all("verified" in e for e in r["effects"])
    server.shutdown()
    print("PASS test_html_dashboard")


if __name__ == "__main__":
    test_boot_full_preset()
    test_html_dashboard()
    print("web UI v2 tests done")
