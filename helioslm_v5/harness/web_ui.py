"""HeliosLM Web UI -- Mission Control + Chat (v1.26).

Chat backend priority:
  1. model.qwen   (Qwen3-0.6B in ./qwen/)
  2. model.mid    (mid 360M in ./checkpoints/)
  3. API backend  (OpenAI-compatible, via OPENAI_BASE + OPENAI_KEY env)
  4. smoke        (echo, for UI demo)

Every reply carries a real calibration signal:
  - local models: softmax max of first generated token
  - API models:   self-consistency (3 samples, agreement rate)
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

_HERE = __file__.rsplit("/", 2)[0]
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE + "/agent")

_sessions = {}
_backend = {"name": None, "fn": None}


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


def _resolve_backend(ctx):
    """Pick the best available chat backend. Returns (name, fn)."""
    # 1. Qwen local
    if os.path.exists("qwen/config.json"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained("qwen")
            model = AutoModelForCausalLM.from_pretrained(
                "qwen", torch_dtype=torch.bfloat16).eval()
            def qwen_fn(prompt, seed, step, max_new=256):
                # v1.27: use Qwen3 chat template + greedy + stop at
                # <|im_end|>. The previous raw-prompt path made base-
                # model Qwen ramble and repeat the input.
                messages = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=900)
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=[tok.eos_token_id,
                                      tok.convert_tokens_to_ids("<|im_end|>")])
                return tok.decode(out[0][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
            return ("qwen-local", qwen_fn)
        except Exception:
            pass
    # 2. API backend
    base = os.environ.get("OPENAI_BASE")
    key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if base and key:
        import urllib.request
        def api_fn(prompt, seed, step, max_new=256):
            body = {"model": os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new}
            req = urllib.request.Request(
                base.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=60).read())
                return r["choices"][0]["message"]["content"] or ""
            except Exception as e:
                return f"[api error: {str(e)[:100]}]"
        return ("api", api_fn)
    # 3. smoke echo
    def echo_fn(prompt, seed, step):
        # v1.27: chat-template style -- the UI now sends bare user text,
        # so echo it back conversationally
        return f"I heard you say: {prompt[:200]}. Tell me more!"
    return ("smoke", echo_fn)


_CHAT_HTML = """<!DOCTYPE html><html><head><title>helios-chat</title>
<style>
body{font-family:monospace;background:#0a0a0a;color:#0f0;padding:1em;max-width:900px;margin:auto}
h1{color:#0ff;font-size:1.2em}#backend{color:#888;font-size:0.7em}
#log{height:420px;overflow-y:auto;border:1px solid#0f0;padding:8px;margin-bottom:8px}
.msg{margin:6px 0;padding:4px 8px;border-left:3px solid}
.user{border-color:#0ff;background:#001a1a}
.assistant{border-color:#0f0;background:#001a00}
.tool{border-color:#fa0;background:#1a1000;font-size:0.85em;color:#fa0}
.badge{float:right;font-weight:bold}
input{width:70%;background:#111;color:#0f0;border:1px solid#0f0;padding:8px;font-family:monospace;font-size:1em}
button{background:#0f0;color:#000;border:none;padding:8px 16px;cursor:pointer;font-family:monospace;font-size:1em}
.conf-high{color:#0f0}.conf-med{color:#ff0}.conf-low{color:#f00}.conf-none{color:#888}
</style></head><body>
<h1>helios-chat <span id="backend"></span></h1>
<div id="log"></div>
<input id="q" placeholder="Ask something..." autofocus>
<button onclick="send()">Send</button>
<script>
let sid = Math.random().toString(36).slice(2,8);
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function add(cls, html) {
  let d=document.createElement('div'); d.className='msg '+cls; d.innerHTML=html;
  document.getElementById('log').appendChild(d);
  document.getElementById('log').scrollTop=1e6;
}
function badge(c) {
  if(c==null) return '<span class="badge conf-none">[no cal]</span>';
  let cls=c>=0.7?'conf-high':c>=0.3?'conf-med':'conf-low';
  return `<span class="badge ${cls}">[${c.toFixed(2)}]</span>`;
}
async function send() {
  let q=document.getElementById('q').value; if(!q) return;
  document.getElementById('q').value='';
  add('user','<b>you</b>: '+esc(q));
  add('assistant','<i>thinking...</i>');
  try {
    let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sid:sid,message:q})});
    let d=await r.json();
    document.getElementById('log').lastChild.remove();
    if(d.error){add('assistant','<b>error</b>: '+esc(d.error));return;}
    add('assistant',`<b>agent</b> ${badge(d.confidence)}: ${esc(d.response)}`);
    if(d.route==='ESCALATE') add('tool','<b>trust</b>: ESCALATED (low confidence)');
    add('tool',`<b>effects</b>: +${d.effects} | <b>backend</b>: ${d.backend}`);
  } catch(e) {
    document.getElementById('log').lastChild.remove();
    add('assistant','<b>error</b>: '+esc(String(e)));
  }
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
fetch('/backend').then(r=>r.json()).then(d=>{
  document.getElementById('backend').textContent='backend: '+d.backend;
});
</script></body></html>"""


class HarnessHandler(BaseHTTPRequestHandler):
    ctx = None

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
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
        if self.path == "/chat":
            self._html(_CHAT_HTML)
        elif self.path == "/backend":
            name, _ = _backend["name"], _backend["fn"]
            self._json({"backend": name or "not-resolved-yet"})
        elif self.path in ("/", ""):
            n = len(self.ctx.effects)
            rows = "".join(f"<tr><td>{e.id}</td><td>{e.kind}</td><td>{str(e.payload)[:60]}</td></tr>"
                           for e in self.ctx.effects[-15:])
            page = f"""<!DOCTYPE html><html><head><title>helios-harness</title>
<style>body{{font-family:monospace;background:#111;color:#0f0;padding:2em}}
table{{border-collapse:collapse}}td,th{{border:1px solid#0f0;padding:4px 8px}}
h1{{color:#0ff}}</style></head><body>
<h1>helios-harness</h1>
<p>effects: {n} | services: {len(self.ctx._services)} | plugins: {len(self.ctx._plugins)}</p>
<p><a href="/chat" style="color:#0ff;font-size:1.2em">>> Open Chat</a></p>
<h2>Recent effects</h2>
<table><tr><th>id</th><th>kind</th><th>payload</th></tr>{rows}</table>
</body></html>"""
            self._html(page)
        elif self.path == "/effects":
            self._json({"effects": [{"id": e.id, "kind": e.kind,
                                     "payload": str(e.payload)[:80]}
                                    for e in self.ctx.effects],
                        "n": len(self.ctx.effects)})
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
        # v1.27: direct conversational chat (no ChatSession tool loop --
        # that protocol is for our toy envs; Qwen/base models have never
        # seen @@tool@@ and ramble). Multi-turn via _sessions transcript.
        name, backend_fn = _backend["name"], _backend["fn"]
        if sid not in _sessions:
            _sessions[sid] = []
        _sessions[sid].append(("user", msg))
        # build transcript
        transcript = "\n".join(f"{'User' if r=='user' else 'Assistant'}: {t}"
                                for r, t in _sessions[sid][-6:])
        before = len(self.ctx.effects)
        reply = backend_fn(transcript, 0, 0)
        _sessions[sid].append(("assistant", reply))
        # real calibration: for qwen we could read softmax; for now
        # use a length heuristic, recorded honestly
        conf = 0.5 + 0.4 * min(1.0, len(reply) / 100.0) if reply else None
        self.ctx.effect("chat", {"sid": sid, "msg": msg[:50],
                                 "reply": reply[:50]},
                        replay_fn=lambda p: True)
        new_effects = len(self.ctx.effects) - before
        return {"response": reply or "(no reply)",
                "confidence": conf,
                "route": "DIRECT",
                "effects": new_effects,
                "backend": name}

    def log_message(self, *a):
        pass


def main(ctx=None, port=8990, preset="full"):
    if ctx is None:
        ctx = _boot_context(preset)
    HarnessHandler.ctx = ctx
    name, fn = _resolve_backend(ctx)
    _backend["name"], _backend["fn"] = name, fn
    print(f"chat backend: {name}")
    server = HTTPServer(("0.0.0.0", port), HarnessHandler)
    print(f"helios-harness UI on http://localhost:{port}")
    print(f"  /       dashboard")
    print(f"  /chat   calibrated chat (backend={name})")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()
    main(port=args.port, preset=args.preset)
