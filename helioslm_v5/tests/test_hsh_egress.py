"""T158 — hsh H13: network egress audit for tool subprocesses (v5.181).

Pins:
1. without the service, bash children get no proxy or hook (unchanged);
2. deny mode: curl through the proxy to a host off the allow list gets
   403, Python in a bash child is refused before DNS, run_code too; each
   attempt comes back as ``egress`` records and a ``[network blocked``
   line, and report() sums them per host;
3. audit mode records the same attempt and lets it through;
4. the proxy forwards plain HTTP to an allowed host;
5. allow patterns use fnmatch; loopback is always allowed;
6. isolate runs bash under ``unshare -rn`` (skipped where unavailable).

7. the web UI mounts it with --egress / --egress-allow.

No test leaves the machine: blocked hosts are refused before any lookup.
"""
import http.client
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (Context, Egress, Tools, Workspace,  # noqa: E402
                             bash_tool, run_code_tool)

PY_CONNECT = ("python3 -c \"import socket\ntry:\n socket.create_connection("
              "('blocked.invalid', 80), timeout=2)\nexcept PermissionError as e:\n"
              " print('refused', e)\nexcept OSError as e:\n print('oserror', e)\"")


def world(tmp_path, egress=None):
    ctx = Context()
    ctx.plugin(Tools)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    if egress is not None:
        ctx.plugin(Egress, egress)
    ctx.plugin(bash_tool)
    ctx.plugin(run_code_tool)
    return ctx


def bash(ctx, command):
    r = ctx.tools.execute("c1", "bash", {"description": "t", "command": command})
    assert not r.is_error, r.content
    return r


def test_no_service_no_proxy(tmp_path):
    r = bash(world(tmp_path), "echo \"[$HTTPS_PROXY][$HSH_EGRESS_LOG]\"")
    assert "[][]" in r.content and "egress" not in r.value


def test_deny_mode(tmp_path):
    ctx = world(tmp_path, {"allow": ["*.example.org"]})
    if shutil.which("curl"):
        r = bash(ctx, "curl -s -o /dev/null -w '%{http_code}' https://blocked.invalid/x")
        assert r.value["output"].strip() in ("000", "403")
        assert r.value["egress"] == [{"layer": "proxy", "host": "blocked.invalid",
                                      "port": 443, "allowed": False}]
        assert "[network blocked by the harness: blocked.invalid:443]" in r.content
    r = bash(ctx, PY_CONNECT)
    assert "refused" in r.value["output"]
    assert {"layer": "python", "host": "blocked.invalid", "port": 80,
            "allowed": False} in r.value["egress"]
    r = ctx.tools.execute("c2", "run_code", {"program": (
        "import urllib.request\n"
        "urllib.request.urlopen('http://blocked2.invalid/', timeout=2)")})
    assert r.is_error and "blocked2.invalid" in r.content
    rep = ctx.egress.report()
    assert rep["blocked.invalid"]["blocked"] >= 1 and rep["blocked2.invalid"]["blocked"] >= 1
    assert ctx.egress.drain() == []


def test_audit_mode_lets_through(tmp_path):
    ctx = world(tmp_path, {"mode": "audit"})
    r = bash(ctx, PY_CONNECT)
    assert "refused" not in r.value["output"]        # DNS fails instead
    assert r.value["egress"][0]["host"] == "blocked.invalid"
    assert "recorded (not on the allow list)" in r.content


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_proxy_forwards_allowed(tmp_path):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Hello)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ctx = world(tmp_path, {})
    try:
        conn = http.client.HTTPConnection("127.0.0.1", int(
            ctx.egress.env()["HSH_EGRESS_PROXY_PORT"]), timeout=5)
        conn.request("GET", f"http://localhost:{srv.server_port}/a?b=1")
        assert json.loads(conn.getresponse().read()) == {"path": "/a?b=1"}
        assert ctx.egress.drain()[-1] == {"layer": "proxy", "host": "localhost",
                                          "port": srv.server_port, "allowed": True}
    finally:
        srv.shutdown()


def test_allow_patterns(tmp_path):
    eg = world(tmp_path, {"allow": ["*.Example.org", "pypi.org"]}).egress
    assert eg.allowed("cdn.example.org") and eg.allowed("PyPI.org")
    assert not eg.allowed("example.org.evil.test") and not eg.allowed("files.pypi.org")
    assert eg.allowed("localhost") and eg.allowed("127.0.0.5")
    with pytest.raises(ValueError):
        Egress(Context(), {"mode": "open"})


def _unshare_ok():
    return shutil.which("unshare") and subprocess.run(
        ["unshare", "-rn", "true"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _unshare_ok(), reason="needs unprivileged user namespaces")
def test_isolate(tmp_path):
    ctx = world(tmp_path, {"isolate": True})
    r = bash(ctx, "tail -n +3 /proc/net/dev | cut -d: -f1")
    assert r.value["output"].split() == ["lo"]


def test_web_flag(tmp_path):
    from helioslm_v5.hsh.web import HarnessApp
    app = HarnessApp(str(tmp_path), str(tmp_path / "data"), demo=True,
                     egress={"mode": "deny", "allow": ["pypi.org"]})
    assert app.ctx.egress.allow == ["pypi.org"]
    assert HarnessApp(str(tmp_path), str(tmp_path / "d2"), demo=True).ctx.get("egress") is None
