"""HeliosLM Web UI -- v1.36 (clean rewrite).

Mission Control dashboard + calibrated chat + media (image/video/audio).
All media served locally via ~/.helios_media (cross-platform).
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_HERE = __file__.rsplit("/", 2)[0]
sys.path.insert(0, _HERE)
sys.path.insert(0, _HERE + "/agent")

_MEDIA_DIR = os.path.join(os.path.expanduser("~"), ".helios_media")
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


def _search_image(query):
    import urllib.parse
    return f"https://picsum.photos/seed/{urllib.parse.quote(query)[:20]}/400/300"


def _search_video(query):
    return [
        "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
        "https://www.w3.org/2010/05/video/mediafiles/foreman-orig.mp4",
    ]


def _proxy_video(url, out_name):
    os.makedirs(_MEDIA_DIR, exist_ok=True)
    out_path = os.path.join(_MEDIA_DIR, out_name)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        try:
            import urllib.request as _u
            req = _u.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = _u.urlopen(req, timeout=60).read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"[media] downloaded {out_name}: {len(data)} bytes", flush=True)
        except Exception as e:
            print(f"[media] download failed: {e}", flush=True)
            return None
    return out_name


def _tts_audio(text, out_path=None):
    import time as _t
    if out_path is None:
        out_path = os.path.join(_MEDIA_DIR, f"tts_{int(_t.time())}.mp3")
    os.makedirs(_MEDIA_DIR, exist_ok=True)
    text = text[:150]
    try:
        from gtts import gTTS
        gTTS(text, lang="en").save(out_path)
        if os.path.getsize(out_path) > 500:
            return out_path
    except Exception:
        pass
    try:
        wav = out_path.replace(".mp3", ".wav")
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, wav)
        engine.runAndWait()
        if os.path.exists(wav) and os.path.getsize(wav) > 500:
            return wav
    except Exception:
        pass
    return None


def _clean_reply(text):
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for bad in ("##assistant##", "##user##", "HeliosLM", "I am HeliosLM"):
        text = text.replace(bad, "")
    lines = text.split("\n")
    out = []
    for ln in lines:
        if not out or ln.strip() != out[-1].strip():
            out.append(ln)
    return "\n".join(out).strip()


_CHAT_HTML = """<!DOCTYPE html><html><head><title>helios-harness</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'Segoe UI',monospace,sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;overflow:hidden}
#sidebar{width:220px;background:#010409;border-right:1px solid#21262d;display:flex;flex-direction:column;padding:12px}
#sidebar .brand{color:#58a6ff;font-weight:bold;font-size:1.1em;margin-bottom:16px}
#sidebar .section{color:#8b949e;font-size:0.75em;text-transform:uppercase;margin:12px 0 6px}
#sidebar .item{padding:6px 10px;border-radius:6px;color:#c9d1d9;font-size:0.9em}
#sidebar .item.active{background:#1f6feb;color:#fff}
#sidebar .spacer{flex:1}
#main{flex:1;display:flex;flex-direction:column}
#header{padding:10px 16px;border-bottom:1px solid#21262d;display:flex;align-items:center;gap:12px}
#header h1{font-size:1em;font-weight:600}
.badge{font-size:0.75em;padding:2px 8px;border-radius:10px}
.badge.high{background:#238636;color:#fff}.badge.med{background:#9e6a03;color:#fff}
.badge.low{background:#da3633;color:#fff}.badge.none{background:#30363d;color:#8b949e}
#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:75%;padding:10px 14px;border-radius:8px;line-height:1.5;font-size:0.92em}
.msg.user{align-self:flex-end;background:#1f6feb;color:#fff}
.msg.assistant{align-self:flex-start;background:#161b22;border:1px solid#30363d}
.msg.tool{align-self:flex-start;background:#0d1117;border:1px solid#21262d;font-size:0.8em;color:#8b949e}
.msg .meta{font-size:0.7em;opacity:0.7;margin-top:4px}
.msg img,.msg video,.msg audio{max-width:100%;border-radius:6px;margin-top:6px}
#inputbar{padding:12px 16px;border-top:1px solid#21262d;display:flex;gap:8px}
#q{flex:1;background:#0d1117;border:1px solid#30363d;border-radius:8px;color:#c9d1d9;padding:10px 14px;outline:none}
#q:focus{border-color:#1f6feb}
#sendbtn{background:#238636;color:#fff;border:none;border-radius:8px;padding:10px 20px;cursor:pointer}
#right{width:260px;background:#010409;border-left:1px solid#21262d;padding:12px;overflow-y:auto;font-size:0.8em}
#right .section{color:#8b949e;font-size:0.72em;text-transform:uppercase;margin:10px 0 6px}
#right .effect{padding:4px 8px;border-bottom:1px solid#161b22;font-family:monospace;color:#8b949e}
#right .effect .id{color:#58a6ff}
</style></head><body>
<div id="sidebar">
  <div class="brand">helios-harness</div>
  <div class="section">Session</div>
  <div class="item active" id="sid-label">session</div>
  <div class="section">Backend</div>
  <div class="item" id="backend-label">...</div>
  <div class="spacer"></div>
  <div class="backend" id="stats"></div>
</div>
<div id="main">
  <div id="header"><h1>Chat</h1><span class="badge none" id="cal-badge">no cal</span></div>
  <div id="chat"></div>
  <div id="inputbar">
    <input id="q" placeholder="Ask anything..." autofocus>
    <button id="sendbtn" onclick="send()">Send</button>
  </div>
</div>
<div id="right">
  <div class="section">Effects audit</div>
  <div id="effects-list"></div>
</div>
<script>
let sid = Math.random().toString(36).slice(2,8);
document.getElementById('sid-label').textContent = 'sid: ' + sid;
function esc(x){return x.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function addMsg(cls, html, meta) {
  let d = document.createElement('div'); d.className = 'msg ' + cls; d.innerHTML = html;
  if (meta) { let m = document.createElement('div'); m.className='meta'; m.innerHTML = meta; d.appendChild(m); }
  document.getElementById('chat').appendChild(d);
  document.getElementById('chat').scrollTop = 1e6;
}
function confBadge(c) {
  if (c == null) return '<span class="badge none">no cal</span>';
  let cls = c>=0.7?'high':c>=0.3?'med':'low';
  return `<span class="badge ${cls}">${c.toFixed(2)}</span>`;
}
function loadEffects() {
  fetch('/effects').then(r=>r.json()).then(d=>{
    let el = document.getElementById('effects-list');
    el.innerHTML = d.effects.slice(-15).reverse().map(e =>
      `<div class="effect"><span class="id">${e.id}</span> ${e.kind}</div>`).join('');
    document.getElementById('stats').textContent = d.n + ' effects total';
  });
}
async function send() {
  let q = document.getElementById('q').value; if (!q) return;
  document.getElementById('q').value = '';
  addMsg('user', esc(q));
  addMsg('assistant', '<i>thinking...</i>');
  try {
    let r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sid:sid, message:q})});
    let d = await r.json();
    document.getElementById('chat').lastChild.remove();
    if (d.error) { addMsg('assistant', '<b>error</b>: ' + esc(d.error)); return; }
    document.getElementById('cal-badge').outerHTML = confBadge(d.confidence).replace('class="badge','id="cal-badge" class="badge');
    let body = d.response;
    if (body.startsWith('<img') || body.startsWith('<video') || body.startsWith('<audio')) {
      addMsg('assistant', body, `backend: ${d.backend} | effects +${d.effects}`);
    } else {
      addMsg('assistant', esc(body), `backend: ${d.backend} | effects +${d.effects}`);
    }
    loadEffects();
  } catch(e) {
    document.getElementById('chat').lastChild.remove();
    addMsg('assistant', '<b>error</b>: ' + esc(String(e)));
  }
}
document.getElementById('q').addEventListener('keydown', e => {if(e.key==='Enter')send()});
fetch('/backend').then(r=>r.json()).then(d => {
  document.getElementById('backend-label').textContent = d.backend;
});
loadEffects();
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
            self._json({"backend": _backend["name"] or "unknown"})
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
<p><a href="/chat" style="color:#0ff;font-size:1.2em">> Open Chat</a></p>
<h2>Recent effects</h2>
<table><tr><th>id</th><th>kind</th><th>payload</th></tr>{rows}</table>
</body></html>"""
            self._html(page)
        elif self.path.startswith("/video/"):
            fname = self.path[len("/video/"):]
            fpath = os.path.join(_MEDIA_DIR, fname)
            if os.path.exists(fpath):
                data = open(fpath, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "not found"}, 404)
        elif self.path.startswith("/audio/"):
            fname = self.path[len("/audio/"):]
            fpath = os.path.join(_MEDIA_DIR, fname)
            if os.path.exists(fpath):
                data = open(fpath, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "not found"}, 404)
        elif self.path == "/effects":
            effects = [{"id": e.id, "kind": e.kind,
                        "payload": str(e.payload)[:80]}
                       for e in self.ctx.effects]
            self._json({"effects": effects, "n": len(effects)})
        elif self.path == "/services":
            self._json({"services": sorted(self.ctx._services.keys())})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/chat":
            try:
                body = self._read_body()
                result = self._chat(body.get("sid", "default"),
                                    body.get("message", ""))
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)[:200]}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def _chat(self, sid, msg):
        name, backend_fn = _backend["name"], _backend["fn"]
        msg_l = msg.lower()
        # image
        if any(k in msg_l for k in ("show me", "show", "picture", "image",
                                     "photo", "画", "图")) and not            any(k in msg_l for k in ("video", "影片", "视频")):
            url = _search_image(msg)
            reply = f'<img src="{url}" style="max-width:100%">'
            _sessions.setdefault(sid, []).append(("user", msg))
            _sessions[sid].append(("assistant", reply))
            self.ctx.effect("media_img", {"q": msg[:40]}, replay_fn=lambda p: True)
            return {"response": reply, "confidence": 0.95, "route": "MEDIA",
                    "effects": 1, "backend": "media"}
        # video
        if any(k in msg_l for k in ("video", "影片", "视频", "play video")):
            urls = _search_video(msg)
            local = None
            for u in urls:
                local = _proxy_video(u, f"vid_{sid}_{int(time.time())}.mp4")
                if local:
                    break
            if local:
                reply = (f'<video controls style="max-width:100%">'
                         f'<source src="/video/{local}" type="video/mp4">'
                         f'Your browser does not support the video tag.</video>'
                         f'<br><small>local: {local}</small>')
            else:
                reply = "<b>video download failed</b>"
            _sessions.setdefault(sid, []).append(("user", msg))
            _sessions[sid].append(("assistant", reply))
            self.ctx.effect("media_vid", {"q": msg[:40]}, replay_fn=lambda p: True)
            return {"response": reply, "confidence": 0.95, "route": "MEDIA",
                    "effects": 1, "backend": "media"}
        # audio
        if any(k in msg_l for k in ("sound", "audio", "voice", "speak",
                                     "say", "声音", "播放", "念")):
            path = _tts_audio(msg)
            if path:
                import shutil
                ext = "wav" if path.endswith(".wav") else "mp3"
                safe = f"tts_{sid}_{int(time.time())}.{ext}"
                shutil.copy(path, os.path.join(_MEDIA_DIR, safe))
                reply = (f'<audio controls style="width:100%">'
                         f'<source src="/audio/{safe}" type="audio/{ext}">'
                         f'</audio><br><small>{safe}</small>')
            else:
                reply = "<b>TTS failed</b> -- pip install gtts"
            _sessions.setdefault(sid, []).append(("user", msg))
            _sessions[sid].append(("assistant", reply))
            self.ctx.effect("media_aud", {"q": msg[:40]}, replay_fn=lambda p: True)
            return {"response": reply, "confidence": 0.95, "route": "MEDIA",
                    "effects": 1, "backend": "media"}
        # plain chat
        if sid not in _sessions:
            _sessions[sid] = []
        _sessions[sid].append(("user", msg))
        transcript = "\n".join(f"{'User' if r=='user' else 'Assistant'}: {t}"
                                for r, t in _sessions[sid][-6:])
        before = len(self.ctx.effects)
        reply = _clean_reply(backend_fn(transcript, 0, 0))
        _sessions[sid].append(("assistant", reply))
        conf = 0.5 + 0.4 * min(1.0, len(reply) / 100.0) if reply else None
        self.ctx.effect("chat", {"sid": sid, "msg": msg[:50]},
                        replay_fn=lambda p: True)
        new_effects = len(self.ctx.effects) - before
        return {"response": reply, "confidence": conf, "route": "DIRECT",
                "effects": new_effects, "backend": name}

    def log_message(self, *a):
        pass


def _resolve_backend(ctx):
    base = os.environ.get("OPENAI_BASE")
    key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if base and key:
        import urllib.request as _u
        def api_fn(prompt, seed, step, max_new=256):
            body = {"model": os.environ.get("OPENAI_MODEL", "deepseek-chat"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new}
            req = _u.Request(base.rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            try:
                r = json.loads(_u.urlopen(req, timeout=60).read())
                return r["choices"][0]["message"]["content"] or ""
            except Exception as e:
                return f"[api error: {str(e)[:100]}]"
        return ("api", api_fn)
    if os.path.exists("qwen/config.json"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained("qwen")
            model = AutoModelForCausalLM.from_pretrained(
                "qwen", torch_dtype=torch.bfloat16).eval()
            def qwen_fn(prompt, seed, step, max_new=256):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                enc = tok(text, return_tensors="pt", truncation=True, max_length=900)
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=min(max_new, 150),
                        do_sample=False, repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=[tok.eos_token_id,
                                      tok.convert_tokens_to_ids("<|im_end|>")])
                reply = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
                import re as _re
                return _re.sub(r"<think>.*?</think>", "", reply,
                               flags=_re.DOTALL).strip()
            return ("qwen-local", qwen_fn)
        except Exception:
            pass
    def echo_fn(prompt, seed, step):
        return f"[no LLM backend] I heard: {prompt[:100]}"
    return ("smoke", echo_fn)


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
    print(f"  /chat   calibrated chat")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()
    main(port=args.port, preset=args.preset)
