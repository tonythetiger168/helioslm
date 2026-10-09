"""HeliosLM Web UI -- Mission Control style (v1.23).

A minimal web interface for the harness: view effects, trigger plugins,
monitor agents. stdlib-only (http.server) for zero dependencies.

Run:  python -m helioslm_v5.harness.web_ui
Open: http://localhost:8990
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from harness.harness_core import Context


class HarnessHandler(BaseHTTPRequestHandler):
    ctx = None   # set by main()

    def _json(self, data, code=200):
        body = json.dumps(data, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._json({"name": "helios-harness", "version": "1.23",
                        "endpoints": ["/effects", "/services", "/plugins",
                                       "/state"]})
        elif self.path == "/effects":
            effects = [{"id": e.id, "kind": e.kind,
                        "payload": str(e.payload)[:80]}
                       for e in self.ctx.effects]
            self._json({"effects": effects, "n": len(effects)})
        elif self.path == "/services":
            self._json({"services": sorted(self.ctx._services.keys())})
        elif self.path == "/state":
            self._json({"state": {k: str(v)[:100]
                                  for k, v in self.ctx.__dict__.items()
                                  if k == "state"}})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        pass   # quiet


def main(ctx=None, port=8990):
    if ctx is None:
        ctx = Context()
    HarnessHandler.ctx = ctx
    server = HTTPServer(("0.0.0.0", port), HarnessHandler)
    print(f"helios-harness UI on http://localhost:{port}")
    print(f"  /effects  -- audit trail")
    print(f"  /services -- registered services")
    server.serve_forever()


if __name__ == "__main__":
    main()
