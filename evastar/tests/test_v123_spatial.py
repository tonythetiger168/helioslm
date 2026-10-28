"""T149 - v1.23: spatial plugin (pano2world client) against a stub server.

The stub speaks the pano2world API (POST /api/generate, GET /api/jobs/<id>,
GET /api/worlds/<id>/scene, POST /api/worlds/<id>/query) with a fixed scene,
so the test needs no GPU and no pano2world checkout."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import SpatialPlugin

SCENE = {"objects": [{"id": "sofa_0", "label": "sofa", "distance": 2.0,
                      "bearing": -30.0, "elevation": -10.0,
                      "center": [-1.0, -0.35, 1.7], "size": [1.8, 0.8],
                      "score": 0.6}],
         "floor_y": -1.6, "walk_radius": 0.7, "clearance": [3.0] * 360}


class _Stub(BaseHTTPRequestHandler):
    calls = []

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Stub.calls.append(("GET", self.path))
        if self.path == "/api/jobs/w1":
            self._send({"id": "w1", "status": "done", "radius": 0.7,
                        "file": "/worlds/w1/world.ply", "error": None})
        elif self.path == "/api/worlds/w1/scene":
            self._send(SCENE)
        else:
            self._send({"detail": "Unknown world"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Stub.calls.append(("POST", self.path, body))
        if self.path == "/api/generate":
            self._send({"id": "w1"})
        elif self.path == "/api/worlds/w1/query":
            if body["op"] == "distance":
                self._send({"from": "camera", "to": body["args"][0],
                            "distance": 2.0})
            else:
                self._send({"text": "1 objects. sofa (sofa_0) 2.0 units"})
        else:
            self._send({"detail": "no"}, 404)


class _Route:
    def __init__(self, value):
        self.value = value


class _Gate:
    """Stands in for TrustGate (which needs torch): records calls, fixed route."""

    def __init__(self, value="DIRECT"):
        self.value, self.calls = value, []

    def decide(self, call, meta):
        self.calls.append(call)
        return _Route(self.value)


def _ctx(url, gate="DIRECT", **kw):
    ctx = Context()
    if gate:
        ctx.register("decision.trust", _Gate(gate))
    ctx.use(SpatialPlugin(base_url=url, **kw))
    return ctx


def _server():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_spatial_generate_scene_query():
    srv, url = _server()
    ctx = _ctx(url, poll_s=0.01)
    w = ctx.spatial.generate("a cozy living room", backend="sdxl360")
    assert w == {"world": "w1", "file": "/worlds/w1/world.ply",
                 "walk_radius": 0.7}, w
    sent = [c for c in _Stub.calls if c[:2] == ("POST", "/api/generate")][0][2]
    assert sent["backend"] == "sdxl360" and sent["prompt"].startswith("a cozy")
    sc = ctx.spatial.scene("w1")
    assert sc["objects"][0]["id"] == "sofa_0" and sc["floor_y"] == -1.6
    d = ctx.spatial.query("w1", "distance", "sofa_0")
    assert d["distance"] == 2.0
    assert "sofa" in ctx.spatial.describe("w1")["text"]
    kinds = [e.kind for e in ctx.effects]
    assert {"spatial_generate", "spatial_scene", "spatial_query"} <= set(kinds)
    srv.shutdown()
    print("PASS test_spatial_generate_scene_query")


def test_spatial_errors_are_dicts():
    srv, url = _server()
    r = _ctx(url).spatial.scene("missing")
    assert "error" in r and "404" in r["error"], r
    srv.shutdown()
    assert "error" in _ctx("http://127.0.0.1:9").spatial.scene("w1")   # nothing listens
    print("PASS test_spatial_errors_are_dicts")


def test_spatial_generate_needs_trustgate():
    r = _ctx("http://127.0.0.1:9", gate=None).spatial.generate("anything")
    assert "no TrustGate" in r.get("error", ""), r          # fail closed
    r = _ctx("http://127.0.0.1:9", gate="ESCALATE").spatial.generate("x")
    assert "blocked by TrustGate (ESCALATE)" in r.get("error", ""), r
    print("PASS test_spatial_generate_needs_trustgate")


if __name__ == "__main__":
    test_spatial_generate_scene_query()
    test_spatial_errors_are_dicts()
    test_spatial_generate_needs_trustgate()
    print("\nspatial plugin tests done")
