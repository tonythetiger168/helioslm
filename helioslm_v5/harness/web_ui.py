"""HeliosLM Web UI -- Mission Control style (v1.24).

Run:  python -m helioslm_v5.harness.web_ui [--preset full|minimal|agent]
Open: http://localhost:8990

The UI boots with a smoke workflow so /effects is never empty.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# ensure harness + agent packages are importable
_HERE = __file__.rsplit("/", 2)[0]          # helioslm_v5/
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE + "/agent")


def _boot_context(preset="full"):
    """Boot a Context with the requested preset + a smoke workflow."""
    from harness.harness_core import Context
    from harness.harness_plugins import (AgentWorkflowPlugin, ModelPlugin,
                                         PresetPlugin, ToolPlugin)
    ctx = Context()
    ctx.use(PresetPlugin(preset, model_fn=None))
    ctx.use(ToolPlugin())
    ctx.effect("boot", {"preset": preset, "msg": "harness started"},
               replay_fn=lambda p: True)
    # smoke workflow: 3 fake tasks so /effects has content
    def fake(prompt, seed, step):
        return "42"
    ctx.use(ModelPlugin("smoke", fake))
    ctx.use(AgentWorkflowPlugin())
    from envs import make_envs
    import random
    rng = random.Random(42)
    env = make_envs()[0]
    for i in range(3):
        task = env.sample(rng)
        wf = ctx.agent.run(task.text, fake, max_steps=2)
        ctx.effect(f"smoke_{i}",
                   {"task": task.text[:40], "final": str(wf["final"])[:20]},
                   replay_fn=lambda p: True)
    return ctx


class HarnessHandler(BaseHTTPRequestHandler):
    ctx = None

    def _json(self, data, code=200):
        body = json.dumps(data, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", ""):
            # simple dashboard
            n_effects = len(self.ctx.effects)
            services = sorted(self.ctx._services.keys())
            rows = "".join(
                f"<tr><td>{e.id}</td><td>{e.kind}</td>"
                f"<td>{str(e.payload)[:60]}</td></tr>"
                for e in self.ctx.effects[-20:])
            page = f"""<!DOCTYPE html><html><head>
<title>helios-harness</title>
<style>body{{font-family:monospace;background:#111;color:#0f0;padding:2em}}
table{{border-collapse:collapse}}td,th{{border:1px solid#0f0;padding:4px 8px}}
h1{{color:#0ff}}</style></head><body>
<h1>helios-harness</h1>
<p>effects: {n_effects} | services: {len(services)}</p>
<p>plugins: {len(self.ctx._plugins)}</p>
<h2>Recent effects</h2>
<table><tr><th>id</th><th>kind</th><th>payload</th></tr>{rows}</table>
<p><a href="/effects" style="color:#0ff">/effects (JSON)</a> |
<a href="/services" style="color:#0ff">/services</a></p>
</body></html>"""
            self._html(page)
        elif self.path == "/effects":
            effects = [{"id": e.id, "kind": e.kind,
                        "payload": str(e.payload)[:80],
                        "verified": e.verify()}
                       for e in self.ctx.effects]
            self._json({"effects": effects, "n": len(effects)})
        elif self.path == "/services":
            self._json({"services": sorted(self.ctx._services.keys())})
        elif self.path == "/plugins":
            self._json({"plugins": [type(p).__name__ for p in self.ctx._plugins]})
        elif self.path == "/state":
            self._json({"state_keys": list(getattr(self.ctx, "state", {}).keys())
                        if hasattr(self.ctx, "state") else []})
        else:
            self._json({"error": "not found", "endpoints": ["/", "/effects",
                        "/services", "/plugins", "/state"]}, 404)

    def log_message(self, format, *args):
        pass


def main(ctx=None, port=8990, preset="full"):
    if ctx is None:
        ctx = _boot_context(preset)
        print(f"booted preset={preset}, effects={len(ctx.effects)}")
    HarnessHandler.ctx = ctx
    server = HTTPServer(("0.0.0.0", port), HarnessHandler)
    print(f"helios-harness UI on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--preset", default="full",
                    choices=["minimal", "full", "agent"])
    args = ap.parse_args()
    main(port=args.port, preset=args.preset)
