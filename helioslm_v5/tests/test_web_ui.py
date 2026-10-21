"""T74 - v1.32: 3-pane UI structure (v1 UI, archived in #54 behind the
harness/web_ui.py shim), plus the v1 media path-traversal regression."""
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


def test_media_routes_stay_inside_media_dir(tmp_path):
    """v1.36 joined the raw URL onto the media dir: /video/../x served any
    file the process could read (and the v1 server binds 0.0.0.0)."""
    import harness.web_ui as ui
    media = tmp_path / "media"
    media.mkdir()
    (media / "ok.mp4").write_bytes(b"mp4")
    (tmp_path / "secret.txt").write_text("secret")
    old = ui._MEDIA_DIR
    ui._MEDIA_DIR = str(media)
    HarnessHandler.ctx = _boot_context("full")
    server = HTTPServer(("127.0.0.1", 0), HarnessHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{port}"
        assert urllib.request.urlopen(url + "/video/ok.mp4", timeout=5).read() == b"mp4"
        import http.client
        for path in ("/video/../secret.txt", "/audio/../secret.txt",
                     "/video/" + str(tmp_path / "secret.txt")):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", path)          # sent as-is, no normalisation
            resp = conn.getresponse()
            assert resp.status == 404 and b"secret" not in resp.read(), path
            conn.close()
    finally:
        server.shutdown()
        ui._MEDIA_DIR = old


if __name__ == "__main__":
    test_three_pane_html()
    print("UI v2 test done")
