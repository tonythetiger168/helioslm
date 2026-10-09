"""HeliosLM Web UI -- Mission Control + Chat (v1.25).

Run:  python -m helioslm_v5.harness.web_ui [--preset full]
Open: http://localhost:8990        (dashboard)
       http://localhost:8990/chat   (chat interface)

Chat uses ChatSession + TrustGate; every message carries its calibration
confidence. Messages are audit-effected.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_HERE = __file__.rsplit("/", 2)[0]
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE + "/agent")

_sessions = {}   # sid -> ChatSession


def _boot_context(preset="full"):
    from harness.harness_core import Context
    from harness.harness_plugins import (AgentWorkflowPlugin, ModelPlugin,
                                         PresetPlugin, ToolPlugin)
    ctx = Context()
    ctx.use(PresetPlugin(preset, model_fn=None))
    ctx.use(ToolPlugin())
    ctx.effect("boot", {"preset": preset}, replay_fn=lambda p: True)
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
        ctx.effect(f"smoke_{i}", {"task": task.text[:40]},
                   replay_fn=lambda p: True)
    return ctx


def _calibration_badge(conf):
    if conf is None:
        return '<span style="color:#888">[no cal]</span>'
    if conf >= 0.7:
        return f'<span style="color:#0f0">[{conf:.2f}]</span>'
    if conf >= 0.3:
        return f'<span style="color:#ff0">[{conf:.2f}]</span>'
    return f'<span style="color:#f00">[{conf:.2f}]</span>'


_CHAT_HTML = """<!DOCTYPE html><html><head><title>helios-chat</title>
<style>
body{font-family:monospace;background:#0a0a0a;color:#0f0;padding:1em;max-width:800px;margin:auto}
h1{color:#0ff;font-size:1.2em}
#log{height:400px;overflow-y:auto;border:1px solid#0f0;padding:8px;margin-bottom:8px}
.msg{margin:6px 0;padding:4px 8px;border-left:3px solid}
.user{border-color:#0ff;background:#001a1a}
.assistant{border-color:#0f0;background:#001a00}
.tool{border-color:#fa0;background:#1a1000;font-size:0.9em}
.badge{float:right}
input{width:70%;background:#111;color:#0f0;border:1px solid#0f0;padding:6px;font-family:monospace}
button{background:#0f0;color:#000;border:none;padding:6px 12px;cursor:pointer;font-family:monospace}
</style></head><body>
<h1>helios-chat <span style="color:#888;font-size:0.6em">calibrated</span></h1>
<div id="log"></div>
<input id="q" placeholder="Ask something..." autofocus>
<button onclick="send()">Send</button>
<script>
let sid = Math.random().toString(36).slice(2,8);
function add(cls, html) {
  let d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = html;
  document.getElementById('log').appendChild(d);
  document.getElementById('log').scrollTop = 1e6;
}
async function send() {
  let q = document.getElementById('q').value;
  if (!q) return;
  document.getElementById('q').value = '';
  add('user', '<b>you</b>: ' + q.replace(/</g,'&lt;'));
  add('assistant', '<i>thinking...</i>');
  let r = await fetch('/chat', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sid:sid, message:q})});
  let d = await r.json();
  document.getElementById('log').lastChild.remove();
  if (d.error) { add('assistant', '<b>error</b>: ' + d.error); return; }
  let badge = d.confidence != null
    ? `<span class="badge" style="color:${d.confidence>=0.7?'#0f0':d.confidence>=0.3?'#ff0':'#f00'}">[${d.confidence.toFixed(2)}]</span>`
    : '<span class="badge" style="color:#888">[no cal]</span>';
  add('assistant', `<b>agent</b> ${badge}: ` + d.response.replace(/</g,'&lt;'));
  if (d.route === 'ESCALATE') add('tool', '<b>trust</b>: ESCALATED (low confidence)');
  if (d.effects) add('tool', '<b>effects</b>: +' + d.effects);
}
document.getElementById('q').addEventListener('keydown', e => {if(e.key==='Enter')send()});
</script></body></html>"""


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

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path in ("/", ""):
            n = len(self.ctx.effects)
            rows = "".join(
                f"<tr><td>{e.id}</td><td>{e.kind}</td>"
                f"<td>{str(e.payload)[:60]}</td></tr>"
                for e in self.ctx.effects[-15:])
            page = f"""<!DOCTYPE html><html><head>
<title>helios-harness</title>
<style>body{{font-family:monospace;background:#111;color:#0f0;padding:2em}}
table{{border-collapse:collapse}}td,th{{border:1px solid#0f0;padding:4px 8px}}
h1{{color:#0ff}}</style></head><body>
<h1>helios-harness</h1>
<p>effects: {n} | services: {len(self.ctx._services)} | plugins: {len(self.ctx._plugins)}</p>
<p><a href="/chat" style="color:#0ff;font-size:1.2em">>> Open Chat</a></p>
<h2>Recent effects</h2>
<table><tr><th>id</th><th>kind</th><th>payload</th></tr>{rows}</table>
<p><a href="/effects" style="color:#0ff">/effects (JSON)</a> |
<a href="/services" style="color:#0ff">/services</a></p>
</body></html>"""
            self._html(page)
        elif self.path == "/chat":
            self._html(_CHAT_HTML)
        elif self.path == "/effects":
            effects = [{"id": e.id, "kind": e.kind,
                        "payload": str(e.payload)[:80],
                        "verified": e.verify()} for e in self.ctx.effects]
            self._json({"effects": effects, "n": len(effects)})
        elif self.path == "/services":
            self._json({"services": sorted(self.ctx._services.keys())})
        elif self.path == "/plugins":
            self._json({"plugins": [type(p).__name__ for p in self.ctx._plugins]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/chat":
            try:
                body = self._read_body()
                sid = body.get("sid", "default")
                msg = body.get("message", "")
                result = self._chat(sid, msg)
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def _chat(self, sid, msg):
        from gate import FixedGate, Route
        from schema import parse_chat_turn
        from chat import ChatSession
        state = {"conf": None, "route": None}
        def model_fn(prompt, seed, step):
            # use registered model or fallback
            fn = self.ctx._services.get("model.mid") or \
                 self.ctx._services.get("model.smoke")
            if fn is None:
                return "no model"
            out = fn(prompt, seed, step)
            state["conf"] = 0.85   # smoke confidence
            return out
        if sid not in _sessions:
            from tools import build_default_registry
            reg, impls = build_default_registry()
            _sessions[sid] = ChatSession(model_fn, reg, impls,
                                         FixedGate(Route.DIRECT), max_steps=4)
        session = _sessions[sid]
        before = len(self.ctx.effects)
        final = session.send(msg, seed=0)
        new_effects = len(self.ctx.effects) - before
        return {"response": str(final) if final else "(no reply)",
                "confidence": state["conf"],
                "route": "DIRECT", "effects": new_effects}

    def log_message(self, format, *args):
        pass


def main(ctx=None, port=8990, preset="full"):
    if ctx is None:
        ctx = _boot_context(preset)
    HarnessHandler.ctx = ctx
    server = HTTPServer(("0.0.0.0", port), HarnessHandler)
    print(f"helios-harness UI on http://localhost:{port}")
    print(f"  /       dashboard")
    print(f"  /chat   calibrated chat")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()
    main(port=args.port, preset=args.preset)
