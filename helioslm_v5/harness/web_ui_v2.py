#!/usr/bin/env python3
"""web_ui_v2.py — HeliosLM Harness Web UI v2.0.

Drop-in replacement for helioslm_v5/harness/web_ui.py (v1.36).

Constraints honoured (CONTRIBUTING.md): pure stdlib, CPU-runnable, no new
dependencies. Optional media helpers lazily import PIL/ffmpeg/edge-tts/gTTS/
pyttsx3 only when those endpoints are actually called.

Fixes vs v1.36:
  * P0 path traversal in /video/ + /audio/ -> unified /media/ handler with
    realpath containment check (_safe_media_path).
  * P0 fabricated confidence (0.5 + 0.4*len(reply)/100, hardcoded 0.95) ->
    confidence is shown ONLY when the backend actually returns one; otherwise
    the badge honestly reads "uncalibrated".
  * P1 single-threaded HTTPServer -> ThreadingHTTPServer.
  * P1 no streaming -> SSE endpoint /api/sessions/<id>/chat/stream with live
    tokens/s; OpenAI-compatible backends are streamed through, others are
    chunk-replayed so the UX is uniform.
  * P1 no persistence -> Jan-style plain-JSON sessions under
    ~/.helios_ui/sessions/, with list/rename/delete/history APIs.
  * P1 UX: markdown rendering (escape-first, XSS-safe), code copy buttons,
    stop button (AbortController), textarea with Shift+Enter, per-session
    temperature/max_tokens/system-prompt panel.
  * P2: security headers (CSP, X-Content-Type-Options, X-Frame-Options,
    Referrer-Policy), backend detection relative to __file__ instead of CWD,
    unified dark design (retro dashboard moved to /dashboard), request size
    limits, JSON error envelope.
  * MTP acceptance-rate telemetry surfaced when the backend reports it.
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- config ---

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.expanduser("~/.helios_ui")
_MEDIA_DIR = os.path.join(_DATA_DIR, "media")
_SESSIONS_DIR = os.path.join(_DATA_DIR, "sessions")
os.makedirs(_MEDIA_DIR, exist_ok=True)
os.makedirs(_SESSIONS_DIR, exist_ok=True)

_MAX_BODY = 1 << 20          # 1 MiB request cap
_HISTORY_TURNS = 12          # messages of context sent to the backend
_STREAM_CHUNK = 24           # chars per replay chunk for non-stream backends

# ------------------------------------------------------- path containment --

def _safe_media_path(root, name):
    """Resolve ``name`` under ``root``; return None if it escapes."""
    if not name or "\x00" in name:
        return None
    root_real = os.path.realpath(root)
    cand = os.path.realpath(os.path.join(root_real, name))
    if cand == root_real or cand.startswith(root_real + os.sep):
        return cand
    return None

# ------------------------------------------------------------ session store

class SessionStore:
    """Jan-style plain-JSON session persistence. Thread-safe."""

    def __init__(self, directory):
        self.dir = directory
        self.lock = threading.Lock()

    def _path(self, sid):
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid or ""):
            return None
        return _safe_media_path(self.dir, sid + ".json")

    def list(self):
        out = []
        with self.lock:
            for fn in sorted(os.listdir(self.dir)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(self.dir, fn), "r", encoding="utf-8") as f:
                        s = json.load(f)
                    out.append({
                        "id": s["id"],
                        "title": s.get("title", "untitled"),
                        "updated": s.get("updated", 0),
                        "n_messages": len(s.get("messages", [])),
                    })
                except (OSError, ValueError, KeyError):
                    continue
        out.sort(key=lambda s: s.get("updated", 0), reverse=True)
        return out

    def create(self, title="new chat"):
        sid = uuid.uuid4().hex[:16]
        now = time.time()
        s = {"id": sid, "title": title, "created": now, "updated": now,
             "params": {"temperature": 0.7, "max_tokens": 512, "system": ""},
             "messages": []}
        self._save(s)
        return s

    def get(self, sid):
        p = self._path(sid)
        if not p or not os.path.exists(p):
            return None
        with self.lock:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None

    def _save(self, s):
        p = self._path(s["id"])
        if not p:
            return
        s["updated"] = time.time()
        tmp = p + ".tmp"
        with self.lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=1)
            os.replace(tmp, p)

    def update(self, sid, title=None, params=None):
        s = self.get(sid)
        if not s:
            return None
        if isinstance(title, str) and title.strip():
            s["title"] = title.strip()[:120]
        if isinstance(params, dict):
            merged = dict(s.get("params", {}))
            for k in ("temperature", "max_tokens", "system"):
                if k in params:
                    merged[k] = params[k]
            try:
                merged["temperature"] = max(0.0, min(2.0, float(merged.get("temperature", 0.7))))
                merged["max_tokens"] = max(1, min(8192, int(merged.get("max_tokens", 512))))
                merged["system"] = str(merged.get("system", ""))[:4000]
            except (TypeError, ValueError):
                pass
            s["params"] = merged
        self._save(s)
        return s

    def append_messages(self, sid, *msgs):
        s = self.get(sid)
        if not s:
            return None
        first_exchange = not s["messages"]
        s["messages"].extend(msgs)
        if first_exchange and s.get("title") in ("new chat", ""):
            first = msgs[0].get("content", "")
            s["title"] = (first[:48] + ("…" if len(first) > 48 else "")) or "new chat"
        self._save(s)
        return s

    def delete(self, sid):
        p = self._path(sid)
        if p and os.path.exists(p):
            with self.lock:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return True
        return False

SESSIONS = SessionStore(_SESSIONS_DIR)

# ---------------------------------------------------------------- backends --

class BackendResult:
    """Normalized generation result. confidence/mtp_acceptance stay None
    unless the backend genuinely reports them — never fabricated."""
    def __init__(self, text, confidence=None, mtp_acceptance=None):
        self.text = text
        self.confidence = confidence
        self.mtp_acceptance = mtp_acceptance


class SmokeBackend:
    """Zero-dependency echo backend; always available."""
    name = "smoke-echo"

    def stream(self, messages, params):
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        reply = ("[smoke] echo: " + user +
                 "\n\n(Configure OPENAI_BASE/OPENAI_KEY or a local checkpoint "
                 "for a real model.)")
        for i in range(0, len(reply), 3):
            yield reply[i:i + 3]
            time.sleep(0.005)


class OpenAIBackend:
    """OpenAI-compatible endpoint (OPENAI_BASE/OPENAI_KEY), streaming."""
    name = "api"

    def __init__(self, base, key, model):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model

    def stream(self, messages, params):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 512),
            "stream": True,
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = (chunk.get("choices", [{}])[0]
                                 .get("delta", {}).get("content"))
                    except ValueError:
                        continue
                    if delta:
                        yield delta
        except (urllib.error.URLError, OSError, ValueError) as e:
            yield "\n[backend error: %s]" % e


class LocalHeliosBackend:
    """Optional local checkpoint via the repo's own inference stack.

    Detected relative to this file (not the CWD). Lazy torch import keeps
    `python web_ui_v2.py` instant on machines without a checkpoint.
    """
    name = "local-helios"

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self._engine = None
        self._tried = False

    def _load(self):
        if self._tried:
            return self._engine
        self._tried = True
        try:
            import torch  # noqa: F401
            src = os.path.join(_SCRIPT_DIR, "..", "src")
            if os.path.isdir(src):
                import sys
                sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "..")))
            from helioslm_v5.src.model_v5 import HeliosLMv5, HeliosLMv5Config  # type: ignore
            cfg = HeliosLMv5Config()
            model = HeliosLMv5(cfg)
            state = torch.load(self.ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("model", state), strict=False)
            model.eval()
            self._engine = model
        except Exception:
            self._engine = None
        return self._engine

    def stream(self, messages, params):
        model = self._load()
        if model is None:
            yield "[local backend unavailable — checkpoint/config mismatch]"
            return
        # Tokenizer wiring is checkpoint-specific; until a BPE ships (repo
        # issue #5), replay the generated text in chunks for uniform UX.
        user = next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")
        text = "[local] checkpoint loaded; generation requires tokenizer (see issue #5). prompt: " + user[:80]
        for i in range(0, len(text), _STREAM_CHUNK):
            yield text[i:i + _STREAM_CHUNK]
            time.sleep(0.01)


def _resolve_backend():
    base = os.environ.get("OPENAI_BASE", "").strip()
    key = os.environ.get("OPENAI_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "helioslm").strip()
    if base and key:
        return OpenAIBackend(base, key, model)
    for cand in (os.path.join(_SCRIPT_DIR, "..", "checkpoints", "toy_v5.13.pt"),
                 os.path.join(_SCRIPT_DIR, "..", "..", "checkpoints",
                              "toy_v5.13.pt"),  # repo-root layout
                 os.path.join(_SCRIPT_DIR, "qwen", "config.json")):
        if os.path.exists(os.path.abspath(cand)):
            b = LocalHeliosBackend(os.path.abspath(cand))
            return b
    return SmokeBackend()

BACKEND = _resolve_backend()

# ------------------------------------------------------------------- media --

def _search_image(prompt):
    """Deterministic placeholder image (picsum), seeded by the prompt."""
    import hashlib
    seed = int(hashlib.blake2b(prompt.encode("utf-8"), digest_size=4).hexdigest(), 16)
    return "https://picsum.photos/seed/%d/768/512" % seed


_BLENDER_MOVIES = {
    "bunny": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
    "sintel": "https://download.blender.org/durian/movies/Sintel.2010.480p.mp4",
    "tears": "https://download.blender.org/mango/ToS/ToS-4k-1920.mov",
}


def _search_video(prompt):
    low = prompt.lower()
    for k, url in _BLENDER_MOVIES.items():
        if k in low:
            return url
    return _BLENDER_MOVIES["bunny"]


def _gen_local_video(prompt):
    """Optional PIL+ffmpeg clip generation; lazily imported."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        return None
    out = os.path.join(_MEDIA_DIR, "gen_%s.mp4" % uuid.uuid4().hex[:10])
    frames_dir = os.path.join(_MEDIA_DIR, "frames_" + uuid.uuid4().hex[:8])
    os.makedirs(frames_dir, exist_ok=True)
    try:
        for i in range(24):
            img = Image.new("RGB", (320, 180), (10 + i * 8 % 200, 20, 40 + i * 4 % 160))
            d = ImageDraw.Draw(img)
            d.text((10, 80), prompt[:38], fill=(230, 230, 230))
            img.save(os.path.join(frames_dir, "f%03d.png" % i))
        r = subprocess.run(
            ["ffmpeg", "-y", "-framerate", "12", "-i",
             os.path.join(frames_dir, "f%03d.png"),
             "-pix_fmt", "yuv420p", out],
            capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        return "/media/" + os.path.basename(out)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        for fn in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, fn))
            except OSError:
                pass
        try:
            os.rmdir(frames_dir)
        except OSError:
            pass


def _tts_audio(text, voice=""):
    """TTS chain edge-tts > gTTS > pyttsx3; lazily imported."""
    out = os.path.join(_MEDIA_DIR, "tts_%s.mp3" % uuid.uuid4().hex[:10])
    try:
        import asyncio
        import edge_tts  # type: ignore
        async def _go():
            v = voice or "zh-CN-XiaoxiaoNeural"
            await edge_tts.Communicate(text, v).save(out)
        asyncio.run(_go())
        if os.path.exists(out):
            return "/media/" + os.path.basename(out)
    except Exception:
        pass
    try:
        from gtts import gTTS  # type: ignore
        gTTS(text=text[:500], lang="zh").save(out)
        return "/media/" + os.path.basename(out)
    except Exception:
        pass
    try:
        import pyttsx3  # type: ignore
        eng = pyttsx3.init()
        eng.save_to_file(text[:500], out)
        eng.runAndWait()
        if os.path.exists(out):
            return "/media/" + os.path.basename(out)
    except Exception:
        pass
    return None


_MEDIA_KEYWORDS = {
    "image": ("畫", "圖片", "照片", "image", "picture", "photo", "draw"),
    "video": ("影片", "視頻", "video", "movie", "clip"),
    "tts": ("唸", "念", "朗讀", "讀出", "say", "read aloud", "tts", "speak"),
}


def _classify_media(text):
    low = text.lower()
    for kind, kws in _MEDIA_KEYWORDS.items():
        if any(k in low for k in kws):
            return kind
    return None

# -------------------------------------------------------------- HTTP layer --

def _json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "HeliosUI/2.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------
    def log_message(self, fmt, *args):  # quieter logs
        pass

    def _headers(self, status=200, ctype="application/json; charset=utf-8",
                 extra=None, length=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' https://picsum.photos data:; "
                         "media-src 'self' https://download.blender.org; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _send_json(self, obj, status=200):
        body = _json_bytes(obj)
        self._headers(status, length=len(body))
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode("utf-8")
        self._headers(status, "text/html; charset=utf-8", length=len(body))
        self.wfile.write(body)

    def _error(self, status, msg):
        self._send_json({"error": msg}, status)

    def _read_body(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if n > _MAX_BODY:
            return "too_large"
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send_html(_CHAT_HTML)
        if path == "/dashboard":
            return self._send_html(_DASHBOARD_HTML)
        if path == "/effects":
            return self._send_html(_effects_html())
        if path == "/api/health":
            return self._send_json({
                "ok": True, "backend": BACKEND.name,
                "confidence": "backend-reported only",
                "version": self.server_version})
        if path == "/api/sessions":
            return self._send_json({"sessions": SESSIONS.list()})
        m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)", path)
        if m:
            s = SESSIONS.get(m.group(1))
            return self._send_json({"session": s}) if s else self._error(404, "not found")
        if path.startswith("/media/"):
            return self._serve_media(path[len("/media/"):])
        return self._error(404, "not found")

    def _serve_media(self, name):
        p = _safe_media_path(_MEDIA_DIR, name)
        if not p or not os.path.isfile(p):
            return self._error(403 if p else 404, "forbidden" if p else "not found")
        ext = os.path.splitext(p)[1].lower()
        ctype = {".mp4": "video/mp4", ".mov": "video/quicktime",
                 ".mp3": "audio/mpeg", ".png": "image/png",
                 ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
        try:
            size = os.path.getsize(p)
            self._headers(200, ctype, length=size)
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except OSError:
            return self._error(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/sessions":
            body = self._read_body()
            if body in (None, "too_large"):
                return self._error(400 if body is None else 413, "bad body")
            s = SESSIONS.create((body or {}).get("title", "new chat"))
            return self._send_json({"session": s}, 201)
        m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)/chat/stream", path)
        if m:
            return self._chat_stream(m.group(1))
        if path == "/api/media/image":
            return self._media_endpoint("image")
        if path == "/api/media/video":
            return self._media_endpoint("video")
        if path == "/api/media/tts":
            return self._media_endpoint("tts")
        return self._error(404, "not found")

    def do_PATCH(self):
        m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)", self.path.split("?", 1)[0])
        if not m:
            return self._error(404, "not found")
        body = self._read_body()
        if body in (None, "too_large"):
            return self._error(400 if body is None else 413, "bad body")
        s = SESSIONS.update(m.group(1), title=body.get("title"), params=body.get("params"))
        return self._send_json({"session": s}) if s else self._error(404, "not found")

    def do_DELETE(self):
        m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)", self.path.split("?", 1)[0])
        if not m:
            return self._error(404, "not found")
        return self._send_json({"deleted": SESSIONS.delete(m.group(1))})

    # -- chat streaming ------------------------------------------------------
    def _chat_stream(self, sid):
        body = self._read_body()
        if body in (None, "too_large"):
            return self._error(400 if body is None else 413, "bad body")
        msg = str((body or {}).get("message", "")).strip()
        if not msg:
            return self._error(400, "empty message")
        s = SESSIONS.get(sid)
        if not s:
            return self._error(404, "session not found")

        media_kind = _classify_media(msg)
        if media_kind:
            return self._chat_media(sid, s, msg, media_kind)

        params = s.get("params", {})
        history = []
        if params.get("system"):
            history.append({"role": "system", "content": params["system"]})
        for m in s.get("messages", [])[-_HISTORY_TURNS:]:
            history.append({"role": m["role"], "content": m["content"]})
        history.append({"role": "user", "content": msg})

        self._headers(200, "text/event-stream; charset=utf-8",
                      {"Cache-Control": "no-cache", "Connection": "close",
                       "X-Accel-Buffering": "no"})
        self.close_connection = True  # SSE ends by EOF, not Content-Length
        t0 = time.time()
        pieces = []

        def emit(obj):
            try:
                self.wfile.write(b"data: " + _json_bytes(obj) + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        try:
            for delta in BACKEND.stream(history, params):
                pieces.append(delta)
                emit({"delta": delta})
        except (BrokenPipeError, ConnectionResetError):
            return  # client hit stop; thread ends quietly
        except Exception as e:
            try:
                emit({"error": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        full = "".join(pieces)
        elapsed = max(time.time() - t0, 1e-6)
        approx_tokens = max(1, len(full) // 4)
        stats = {"done": True, "elapsed_s": round(elapsed, 2),
                 "tokens": approx_tokens,
                 "tok_s": round(approx_tokens / elapsed, 1),
                 "confidence": getattr(BACKEND, "last_confidence", None),
                 "mtp_acceptance": getattr(BACKEND, "last_mtp_acceptance", None)}
        SESSIONS.append_messages(sid,
                                 {"role": "user", "content": msg},
                                 {"role": "assistant", "content": full,
                                  "stats": stats})
        try:
            emit(stats)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _chat_media(self, sid, session, msg, kind):
        """Media requests: non-streamed JSON with honest (absent) confidence."""
        prompt = re.sub(r"(畫|圖片|照片|image|picture|photo|draw|影片|視頻|video|"
                        r"movie|clip|唸|念|朗讀|讀出|say|read aloud|tts|speak)",
                        "", msg, flags=re.I).strip() or msg
        if kind == "image":
            url = _search_image(prompt)
        elif kind == "video":
            url = _search_video(prompt) or _gen_local_video(prompt)
        else:
            url = _tts_audio(prompt)
        if not url:
            return self._error(503, "media backend unavailable")
        content = "%s: %s" % (kind, url)
        stats = {"confidence": None}  # media responses carry no confidence
        SESSIONS.append_messages(sid,
                                 {"role": "user", "content": msg},
                                 {"role": "assistant", "content": content,
                                  "media": {"kind": kind, "url": url},
                                  "stats": stats})
        return self._send_json({"media": {"kind": kind, "url": url},
                                "stats": stats})

    def _media_endpoint(self, kind):
        body = self._read_body()
        if body in (None, "too_large"):
            return self._error(400 if body is None else 413, "bad body")
        text = str((body or {}).get("prompt") or (body or {}).get("text") or "")
        if not text.strip():
            return self._error(400, "empty prompt")
        if kind == "image":
            url = _search_image(text)
        elif kind == "video":
            url = _search_video(text) or _gen_local_video(text)
        else:
            url = _tts_audio(text, str((body or {}).get("voice", "")))
        return self._send_json({"url": url}) if url else self._error(503, "unavailable")

# ---------------------------------------------------------------- frontend --

_CHAT_HTML = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HeliosLM Harness</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--dim:#8b949e;
--acc:#58a6ff;--user:#1f6feb;--ok:#3fb950;--warn:#d29922}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Segoe UI",
"Noto Sans TC","PingFang TC",sans-serif;height:100vh;display:flex;overflow:hidden}
#side{width:250px;background:var(--panel);border-right:1px solid var(--border);
display:flex;flex-direction:column;flex-shrink:0}
#side h1{font-size:15px;padding:14px 14px 8px;letter-spacing:.5px}
#newBtn{margin:6px 14px 10px;padding:8px;background:var(--user);color:#fff;
border:0;border-radius:6px;cursor:pointer;font-size:13px}
#newBtn:hover{filter:brightness(1.15)}
#sessList{flex:1;overflow-y:auto;padding:0 8px}
.sess{padding:8px 10px;border-radius:6px;cursor:pointer;color:var(--dim);
display:flex;justify-content:space-between;gap:6px;align-items:center}
.sess:hover{background:#21262d;color:var(--fg)}
.sess.on{background:#21262d;color:var(--fg);outline:1px solid var(--border)}
.sess .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.sess .x{opacity:0;color:var(--dim);border:0;background:none;cursor:pointer;font-size:14px}
.sess:hover .x{opacity:.8}
#sideFooter{padding:10px 14px;border-top:1px solid var(--border);font-size:12px;
color:var(--dim);display:flex;flex-direction:column;gap:4px}
#sideFooter a{color:var(--acc);text-decoration:none}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#topbar{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;
align-items:center;gap:10px;background:var(--panel)}
#sessTitle{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;border:1px solid var(--border);
color:var(--dim)}
.badge.on{color:var(--ok);border-color:var(--ok)}
button.ghost{background:none;border:1px solid var(--border);color:var(--dim);
border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px}
button.ghost:hover{color:var(--fg);border-color:var(--dim)}
#chat{flex:1;overflow-y:auto;padding:18px 16px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:78%;padding:10px 14px;border-radius:12px;white-space:normal;
word-wrap:break-word}
.msg.user{align-self:flex-end;background:var(--user);color:#fff;
border-bottom-right-radius:4px}
.msg.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--border);
border-bottom-left-radius:4px}
.meta{font-size:11px;color:var(--dim);margin-top:6px;display:flex;gap:10px;flex-wrap:wrap}
.msg pre{background:#0a0e14;border:1px solid var(--border);border-radius:8px;
padding:10px;overflow-x:auto;margin:8px 0;position:relative}
.msg code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.msg p code,.msg li code{background:#0a0e14;padding:1px 5px;border-radius:4px}
.copyBtn{position:absolute;top:6px;right:6px;font-size:11px;background:#21262d;
border:1px solid var(--border);color:var(--dim);border-radius:5px;padding:2px 8px;
cursor:pointer}
.copyBtn:hover{color:var(--fg)}
.msg img,.msg video{max-width:100%;border-radius:8px;margin-top:8px}
.msg audio{margin-top:8px;width:260px}
.msg h1,.msg h2,.msg h3{margin:10px 0 4px;font-size:16px}
.msg ul,.msg ol{margin:6px 0 6px 22px}
.msg a{color:var(--acc)}
.typing::after{content:"▌";animation:blink 1s infinite}
@keyframes blink{50%{opacity:0}}
#composer{border-top:1px solid var(--border);padding:12px 16px;background:var(--panel)}
#inputRow{display:flex;gap:8px;align-items:flex-end}
#inp{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;
color:var(--fg);padding:10px 12px;font:inherit;resize:none;max-height:160px}
#inp:focus{outline:1px solid var(--acc)}
#sendBtn,#stopBtn{padding:10px 18px;border-radius:8px;border:0;cursor:pointer;font-size:14px}
#sendBtn{background:var(--user);color:#fff}
#sendBtn:disabled{opacity:.5;cursor:default}
#stopBtn{background:#da3633;color:#fff;display:none}
#hint{font-size:11px;color:var(--dim);margin-top:6px}
#drawer{position:fixed;top:0;right:0;width:290px;height:100%;background:var(--panel);
border-left:1px solid var(--border);padding:18px;transform:translateX(100%);
transition:transform .2s;z-index:10;overflow-y:auto}
#drawer.open{transform:none}
#drawer h2{font-size:14px;margin-bottom:14px}
#drawer label{display:block;font-size:12px;color:var(--dim);margin:12px 0 4px}
#drawer input,#drawer textarea{width:100%;background:var(--bg);border:1px solid var(--border);
border-radius:6px;color:var(--fg);padding:7px 9px;font:inherit}
#drawer textarea{resize:vertical;min-height:70px}
#saveParams{margin-top:16px;width:100%;padding:9px;background:var(--user);color:#fff;
border:0;border-radius:6px;cursor:pointer}
@media(max-width:760px){#side{display:none}}
</style></head>
<body>
<aside id="side">
  <h1>⚡ HeliosLM Harness</h1>
  <button id="newBtn">＋ New chat</button>
  <div id="sessList"></div>
  <div id="sideFooter">
    <span id="backendBadge" class="badge">backend: …</span>
    <a href="/dashboard" target="_blank">retro dashboard ↗</a>
    <a href="/effects" target="_blank">effects audit ↗</a>
  </div>
</aside>
<div id="main">
  <div id="topbar">
    <span id="sessTitle">—</span>
    <span id="confBadge" class="badge" title="shown only when the backend reports one">uncalibrated</span>
    <button class="ghost" id="paramsBtn">⚙ params</button>
  </div>
  <div id="chat"></div>
  <div id="composer">
    <div id="inputRow">
      <textarea id="inp" rows="1" placeholder="Message… (Enter to send, Shift+Enter for newline)"></textarea>
      <button id="sendBtn">Send</button>
      <button id="stopBtn">Stop</button>
    </div>
    <div id="hint">media: try “畫一張山” / “video of bunny” / “朗讀這段” · confidence 只在後端真實回傳時顯示</div>
  </div>
</div>
<div id="drawer">
  <h2>Session parameters</h2>
  <label>temperature <span id="tempVal"></span></label>
  <input type="range" id="pTemp" min="0" max="2" step="0.05">
  <label>max_tokens</label>
  <input type="number" id="pMaxTok" min="1" max="8192">
  <label>system prompt</label>
  <textarea id="pSys"></textarea>
  <button id="saveParams">Save</button>
</div>
<script>
"use strict";
const $ = id => document.getElementById(id);
let sessionId = null, abortCtl = null, sessions = [];

/* ---------- mini markdown: escape FIRST, then transform (XSS-safe) ------ */
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")
  .replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function md(src){
  let s = esc(src);
  const blocks = [];
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
    blocks.push('<pre><button class="copyBtn">copy</button><code>' + code + '</code></pre>');
    return "\x00" + (blocks.length - 1) + "\x00";
  });
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>")
       .replace(/^### (.*)$/gm, "<h3>$1</h3>")
       .replace(/^## (.*)$/gm, "<h2>$1</h2>")
       .replace(/^# (.*)$/gm, "<h1>$1</h1>")
       .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
       .replace(/\*([^*\n]+)\*/g, "<i>$1</i>")
       .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/(?:^|\n)((?:[-*] .*(?:\n|$))+)/g, m => {
    const items = m.trim().split("\n").map(l => "<li>" + l.replace(/^[-*] /, "") + "</li>").join("");
    return "\n<ul>" + items + "</ul>";
  });
  s = s.replace(/\n{2,}/g, "<br><br>").replace(/\n/g, "<br>");
  s = s.replace(/\x00(\d+)\x00/g, (m, i) => blocks[+i]);
  return s;
}
document.addEventListener("click", e => {
  if (e.target.classList && e.target.classList.contains("copyBtn")) {
    const code = e.target.parentElement.querySelector("code").innerText;
    navigator.clipboard.writeText(code).then(() => { e.target.textContent = "copied";
      setTimeout(() => e.target.textContent = "copy", 1200); });
  }
});

/* ------------------------------- sessions ------------------------------- */
async function api(path, opts){
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(()=>({}))).error || r.status);
  return r.json();
}
async function loadSessions(){
  sessions = (await api("/api/sessions")).sessions;
  const el = $("sessList"); el.innerHTML = "";
  for (const s of sessions) {
    const d = document.createElement("div");
    d.className = "sess" + (s.id === sessionId ? " on" : "");
    const t = document.createElement("span"); t.className = "t"; t.textContent = s.title;
    const x = document.createElement("button"); x.className = "x"; x.textContent = "×";
    x.title = "delete";
    x.onclick = async ev => { ev.stopPropagation();
      await api("/api/sessions/" + s.id, {method: "DELETE"});
      if (s.id === sessionId) sessionId = null;
      await loadSessions(); if (!sessionId && sessions[0]) openSession(sessions[0].id);
      else if (!sessionId) await newSession(); };
    d.append(t, x);
    d.onclick = () => openSession(s.id);
    el.appendChild(d);
  }
}
async function newSession(){
  const s = (await api("/api/sessions", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: "{}"})).session;
  await loadSessions(); openSession(s.id);
}
async function openSession(id){
  sessionId = id;
  const s = (await api("/api/sessions/" + id)).session;
  $("sessTitle").textContent = s.title;
  $("chat").innerHTML = "";
  for (const m of s.messages) addMsg(m.role, m.content, m);
  fillParams(s.params || {});
  await loadSessions();
  $("chat").scrollTop = $("chat").scrollHeight;
}

/* -------------------------------- render -------------------------------- */
function addMsg(role, content, m){
  const d = document.createElement("div");
  d.className = "msg " + (role === "user" ? "user" : "bot");
  const body = document.createElement("div");
  if (m && m.media) {
    body.textContent = content;
    const url = m.media.url;
    let node;
    if (m.media.kind === "image") { node = document.createElement("img"); node.src = url; }
    else if (m.media.kind === "video") { node = document.createElement("video"); node.src = url; node.controls = true; }
    else { node = document.createElement("audio"); node.src = url; node.controls = true; }
    body.appendChild(document.createElement("br")); body.appendChild(node);
  } else if (role === "user") {
    body.textContent = content;
  } else {
    body.innerHTML = md(content);
  }
  d.appendChild(body);
  if (m && m.stats && role !== "user") {
    const meta = document.createElement("div"); meta.className = "meta";
    if (m.stats.tok_s) meta.appendChild(span(m.stats.tokens + " tok · " + m.stats.tok_s + " tok/s · " + m.stats.elapsed_s + "s"));
    if (m.stats.mtp_acceptance != null) meta.appendChild(span("MTP accept " + (m.stats.mtp_acceptance*100).toFixed(1) + "%"));
    meta.appendChild(span(m.stats.confidence != null ? "conf " + m.stats.confidence.toFixed(2) : "uncalibrated"));
    d.appendChild(meta);
  }
  $("chat").appendChild(d);
  $("chat").scrollTop = $("chat").scrollHeight;
  return d;
}
function span(t){ const s = document.createElement("span"); s.textContent = t; return s; }

/* --------------------------------- send --------------------------------- */
async function send(){
  const inp = $("inp"), text = inp.value.trim();
  if (!text || !sessionId || abortCtl) return;
  inp.value = ""; autogrow();
  addMsg("user", text);
  const bot = addMsg("assistant", "");
  const body = bot.firstChild; body.classList.add("typing");
  abortCtl = new AbortController();
  $("sendBtn").disabled = true; $("stopBtn").style.display = "";
  let acc = "";
  try {
    const r = await fetch("/api/sessions/" + sessionId + "/chat/stream", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text}), signal: abortCtl.signal});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!line.startsWith("data: ")) continue;
        const obj = JSON.parse(line.slice(6));
        if (obj.delta) { acc += obj.delta; body.innerHTML = md(acc); }
        if (obj.error) { acc += "\n[error] " + obj.error; body.innerHTML = md(acc); }
        if (obj.done) {
          body.classList.remove("typing");
          const meta = document.createElement("div"); meta.className = "meta";
          meta.appendChild(span(obj.tokens + " tok · " + obj.tok_s + " tok/s · " + obj.elapsed_s + "s"));
          if (obj.mtp_acceptance != null) meta.appendChild(span("MTP accept " + (obj.mtp_acceptance*100).toFixed(1) + "%"));
          const conf = obj.confidence != null ? "conf " + obj.confidence.toFixed(2) : "uncalibrated";
          meta.appendChild(span(conf));
          $("confBadge").textContent = conf;
          $("confBadge").className = "badge" + (obj.confidence != null ? " on" : "");
          bot.appendChild(meta);
        }
        $("chat").scrollTop = $("chat").scrollHeight;
      }
    }
  } catch (e) {
    body.classList.remove("typing");
    if (e.name !== "AbortError") { acc += "\n[aborted: " + e.message + "]"; body.innerHTML = md(acc); }
  } finally {
    abortCtl = null;
    $("sendBtn").disabled = false; $("stopBtn").style.display = "none";
    loadSessions();
    const s = await api("/api/sessions/" + sessionId).catch(() => null);
    if (s) $("sessTitle").textContent = s.session.title;
  }
}
$("sendBtn").onclick = send;
$("stopBtn").onclick = () => { if (abortCtl) abortCtl.abort(); };
$("inp").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }});
function autogrow(){ const i = $("inp"); i.style.height = "auto";
  i.style.height = Math.min(i.scrollHeight, 160) + "px"; }
$("inp").addEventListener("input", autogrow);
$("newBtn").onclick = newSession;

/* -------------------------------- params -------------------------------- */
function fillParams(p){
  $("pTemp").value = p.temperature ?? 0.7;
  $("tempVal").textContent = $("pTemp").value;
  $("pMaxTok").value = p.max_tokens ?? 512;
  $("pSys").value = p.system ?? "";
}
$("pTemp").oninput = () => $("tempVal").textContent = $("pTemp").value;
$("paramsBtn").onclick = () => $("drawer").classList.toggle("open");
$("saveParams").onclick = async () => {
  await api("/api/sessions/" + sessionId, {method: "PATCH",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({params: {temperature: +$("pTemp").value,
      max_tokens: +$("pMaxTok").value, system: $("pSys").value}})});
  $("drawer").classList.remove("open");
};

/* --------------------------------- boot --------------------------------- */
(async () => {
  const h = await api("/api/health").catch(() => null);
  $("backendBadge").textContent = "backend: " + (h ? h.backend : "offline");
  await loadSessions();
  if (sessions[0]) openSession(sessions[0].id); else await newSession();
})();
</script></body></html>"""

_DASHBOARD_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>HeliosLM retro dashboard</title>
<style>body{background:#0a0f0a;color:#33ff66;font:14px/1.6 ui-monospace,Menlo,monospace;
padding:32px;max-width:860px;margin:auto}
h1{color:#66ff99;border-bottom:1px dashed #33ff66;padding-bottom:8px}
a{color:#66ff99}pre{background:#061206;padding:12px;border:1px solid #1c3a1c;
overflow:auto}</style></head><body>
<h1>HELIOSLM // RETRO DASHBOARD</h1>
<pre id="out">loading…</pre>
<p><a href="/">← back to chat</a> · <a href="/effects">effects audit</a></p>
<script>
fetch("/api/health").then(r=>r.json()).then(h=>{
  document.getElementById("out").textContent =
    "backend     : " + h.backend + "\n" +
    "ui version  : " + h.version + "\n" +
    "confidence  : " + h.confidence + "\n" +
    "status      : OK\n\n" +
    "The retro panel lives on; the main chat moved to a unified dark design.";
}).catch(e=>{document.getElementById("out").textContent="health check failed: "+e;});
</script></body></html>"""


def _effects_html():
    """Effects-audit panel: the repo's signature transparency surface."""
    rows = [
        ("chat.stream", "model generate", "none (in-memory + session JSON)"),
        ("media.image", "HTTP GET picsum.photos", "network egress"),
        ("media.video", "HTTP GET download.blender.org / local ffmpeg", "network egress / subprocess"),
        ("media.tts", "edge-tts / gTTS / pyttsx3", "network egress or local audio"),
        ("sessions", "read/write ~/.helios_ui/sessions/*.json", "local filesystem"),
    ]
    body = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % t for t in rows)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>effects audit</title><style>body{background:#0d1117;color:#e6edf3;"
            "font:14px sans-serif;padding:32px}table{border-collapse:collapse}"
            "td,th{border:1px solid #30363d;padding:8px 12px;text-align:left}"
            "th{color:#58a6ff}</style></head><body>"
            "<h1>Effects audit</h1><p>Every externally-visible effect this UI can trigger.</p>"
            "<table><tr><th>surface</th><th>effect</th><th>class</th></tr>" +
            body + "</table></body></html>")


# -------------------------------------------------------------------- main --

def main(port=8990):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("HeliosLM Harness UI v2.0  →  http://127.0.0.1:%d" % port)
    print("backend: %s   sessions: %s" % (BACKEND.name, _SESSIONS_DIR))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8990)
