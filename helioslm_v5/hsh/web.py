"""hsh web UI: a local browser front end for the hsh harness.

Modelled on the DeepSeek Harness web app (sidebar of sessions grouped by
workspace, a centred composer, streamed replies with collapsible thinking
and tool cards, a settings dialog), written fresh in the standard library
plus one static page (``webui/``: HTML, CSS and JS, no build step and no
third-party code).

Run it::

    export DEEPSEEK_API_KEY=...          # or use --demo / --openai-base
    python -m helioslm_v5.hsh.web --workspace /path/to/repo --open

What the server does:

- one workspace per server; sessions are hsh agents created from a preset
  (``standard`` by default) and saved as JSONL session logs under
  ``--data-dir`` (default ``~/.hsh/web``), so they survive a restart and
  resume where they stopped;
- each turn runs in a background thread; the browser follows a session
  with server-sent events: every logged event, live text and thinking
  deltas, status changes, and questions;
- ``ask_user_question``, ``exit_plan_mode`` and tool approvals appear as
  cards in the page (``ctx.questions`` / ``ctx.approval``); the
  permission chip picks ``auto`` (run tools), ``ask`` (approve each
  write, command and MCP call) or ``read-only`` (refuse them);
- API keys are read from environment variables only; the page shows
  whether each one is set and never accepts or stores a key.

Security: binds to 127.0.0.1 by default; requests whose ``Host`` is not
loopback (DNS rebinding) and state-changing requests whose ``Origin`` is
not this server are refused; a strict Content-Security-Policy allows only
this server's own script and style.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .builtin import Workspace
from .egress import Egress
from .interaction import plan_state, set_plan_mode
from .kernel import Context
from .llm import LLM, OpenAICompatAdapter, ScriptedAdapter
from .loop import Agents
from .presets import PRESETS, compose, create_agent
from .session import Session, Sessions, derive_messages, usage_summary
from .skills import FilesystemSkills, default_roots
from .tools import Tools

STATIC = Path(__file__).with_name("webui")
MAX_BODY = 1_000_000
FRAME_KEEP = 4000
WRITE_TOOLS = {"bash", "write", "edit", "str_replace_editor", "run_code", "job_kill"}

DEFAULT_PROVIDERS = {
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com",
                 "api_key_env": "DEEPSEEK_API_KEY", "stream": True,
                 "models": ["deepseek-v4-flash", "deepseek-v4-pro"]},
}


# ----- demo model --------------------------------------------------------------

def demo_reply(request, index):
    """A keyless stand-in model so the UI can be tried and tested: it lists
    the workspace with ``glob`` and answers from what it saw; "ask me"
    makes it ask a question first."""
    msgs = request["messages"]
    last = msgs[-1] if msgs else {}
    if last.get("role") == "tool":
        prev = next((m for m in reversed(msgs) if m.get("role") == "assistant"), {})
        called = [c["name"] for c in prev.get("tool_calls") or []]
        if "ask_user_question" in called:
            return {"content": f"Thanks. You answered:\n\n> {last['content']}"}
        files = [x for x in last["content"].splitlines() if x.strip()]
        listing = "\n".join(f"- `{f}`" for f in files[:20]) or "- (nothing)"
        noun = "file" if len(files) == 1 else "files"
        return {"content": f"The workspace has **{len(files)}** {noun} at the top level:\n\n"
                           f"{listing}\n\nThis is the demo model. Set `DEEPSEEK_API_KEY` "
                           "(or start with `--openai-base`) to talk to a real model.",
                "reasoning": "I listed the workspace before answering."}
    names = {t["name"] for t in request["header"].get("tools") or []}
    text = last.get("content", "") if last.get("role") == "user" else ""
    if "ask_user_question" in names and "ask me" in text.lower():
        return {"tool_calls": [{"id": f"q{index}", "name": "ask_user_question",
                                "arguments": json.dumps({"questions": [{
                                    "id": "color", "header": "Choose",
                                    "question": "Which colour should the button be?",
                                    "options": [{"label": "Blue (Recommended)"},
                                                {"label": "Green"}]}]})}]}
    if "glob" in names:
        return {"tool_calls": [{"id": f"g{index}", "name": "glob",
                                "arguments": json.dumps({"pattern": "*"})}]}
    return {"content": "Hello from the demo model."}


# ----- one live session --------------------------------------------------------

class LiveSession:
    def __init__(self, app, agent, meta):
        self.app = app
        self.agent = agent
        self.meta = meta
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.frames = []
        self.base = 0          # index of frames[0] in the whole feed
        self.running = False
        self.pending = {}      # qid -> {"event", "answer", "frame"}
        self.permission = meta.get("permission", "auto")
        agent.session.subscribe(self._on_event)

    # feed ---------------------------------------------------------------
    def push(self, frame):
        with self.cond:
            self.frames.append(frame)
            if len(self.frames) > FRAME_KEEP:
                drop = len(self.frames) - FRAME_KEEP
                del self.frames[:drop]
                self.base += drop
            self.cond.notify_all()

    def since(self, cursor, timeout):
        with self.cond:
            if cursor >= self.base + len(self.frames):
                self.cond.wait(timeout)
            start = max(cursor, self.base)
            return self.frames[start - self.base:], self.base + len(self.frames)

    def cursor(self):
        with self.cond:
            return self.base + len(self.frames)

    def _on_event(self, ev):
        self.app.store.append(self.agent.session.id, ev)
        if ev["type"] == "user/message" and ev["data"].get("source") == "human" \
                and not self.meta.get("title"):
            self.meta["title"] = _title(ev["data"]["content"])
        self.meta["updated"] = time.time()
        self.app.store.save_meta(self.agent.session.id, self.meta)
        self.push({"kind": "event", "event": ev})

    # running ------------------------------------------------------------
    def send(self, text):
        with self.lock:
            self.agent.followup(text)
            if self.running:
                return
            self.running = True
        self.push({"kind": "status", "status": "running"})
        threading.Thread(target=self._drive, daemon=True).start()

    def _drive(self):
        while True:
            try:
                self.agent.run()
            except Exception as exc:  # a bug, not a model error: show it
                self.push({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
                self.agent.inbox.clear()
            with self.lock:
                if not self.agent.inbox:
                    self.running = False
                    break
        self.push({"kind": "status", "status": "idle"})

    def cancel(self):
        self.agent.cancel()
        for qid in list(self.pending):
            self.answer(qid, None)

    # questions ----------------------------------------------------------
    def ask(self, kind, payload, timeout=None):
        qid = uuid.uuid4().hex[:12]
        slot = {"event": threading.Event(), "answer": None,
                "frame": dict(payload, kind=kind, qid=qid)}
        self.pending[qid] = slot
        self.push(slot["frame"])
        slot["event"].wait(timeout)
        self.pending.pop(qid, None)
        self.push({"kind": "question-closed", "qid": qid})
        return slot["answer"]

    def answer(self, qid, value):
        slot = self.pending.get(qid)
        if slot is None:
            return False
        slot["answer"] = value
        slot["event"].set()
        return True

    def snapshot(self):
        a = self.agent
        return {"id": a.session.id, "meta": self.meta, "events": a.session.events,
                "status": "running" if self.running else "idle",
                "cursor": self.cursor(), "permission": self.permission,
                "questions": [s["frame"] for s in self.pending.values()],
                "plan": plan_state(a),
                "usage": usage_summary(a.session.events)["total"]}


def _title(text):
    t = " ".join(text.split())
    if t.startswith("<skill_content"):
        m = re.search(r'name="([^"]+)"', t)
        t = "/" + (m.group(1) if m else "skill")
    return t[:60] + ("…" if len(t) > 60 else "")


# ----- storage -----------------------------------------------------------------

class Store:
    """``<data>/<workspace-hash>/<id>.jsonl`` + ``<id>.meta.json``."""

    def __init__(self, data_dir, workspace):
        key = hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
        self.dir = Path(data_dir).expanduser() / key
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "workspace.txt").write_text(str(workspace), encoding="utf-8")
        self._lock = threading.Lock()

    def _safe(self, sid):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid):
            raise ValueError("bad session id")
        return sid

    def append(self, sid, ev):
        with self._lock, open(self.dir / f"{self._safe(sid)}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")

    def save_meta(self, sid, meta):
        p = self.dir / f"{self._safe(sid)}.meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def load(self, sid):
        p = self.dir / f"{self._safe(sid)}.jsonl"
        events = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        meta_p = self.dir / f"{sid}.meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        return events, meta

    def list(self):
        out = []
        for p in self.dir.glob("*.meta.json"):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(dict(meta, id=p.name[:-len(".meta.json")]))
        return sorted(out, key=lambda m: -m.get("updated", 0))

    def delete(self, sid):
        for suffix in (".jsonl", ".meta.json"):
            p = self.dir / f"{self._safe(sid)}{suffix}"
            if p.exists():
                p.unlink()


# ----- the app -----------------------------------------------------------------

class HarnessApp:
    def __init__(self, workspace, data_dir="~/.hsh/web", providers=None, demo=False,
                 preset="standard", skill_dirs=(), mcp_config=None, egress=None):
        self.workspace = Path(workspace).resolve()
        self.store = Store(data_dir, self.workspace)
        self.default_preset = preset
        self.skill_dirs = list(skill_dirs)
        self.mcp_config = mcp_config
        self.ctx = Context()
        self.ctx.plugin(Sessions, {"clock": lambda: round(time.time(), 3)})
        for p in (Tools, LLM, Agents):
            self.ctx.plugin(p)
        self.ctx.plugin(Workspace, {"root": str(self.workspace)})
        if egress is not None:
            self.ctx.plugin(Egress, egress)
        self.providers = {}
        for name, spec in (DEFAULT_PROVIDERS if providers is None else providers).items():
            adapter = spec.get("adapter") or OpenAICompatAdapter(
                spec["base_url"], spec.get("api_key_env"), stream=spec.get("stream", True))
            self.ctx.llm.register(name, adapter)
            self.providers[name] = dict({k: v for k, v in spec.items() if k != "adapter"})
        if demo:
            self.ctx.llm.register("demo", ScriptedAdapter(fn=demo_reply))
            self.providers["demo"] = {"label": "Demo (no model)", "models": ["demo"],
                                      "api_key_env": None}
        self.ctx.provide("questions", self._questions)
        self.ctx.provide("approval", self._approval)
        self.ctx.on("agent/assistant-stream", self._chunk)
        self.live = {}
        self._lock = threading.Lock()

    # providers -------------------------------------------------------------
    def models(self):
        out = []
        for name, spec in self.providers.items():
            env = spec.get("api_key_env")
            ready = env is None or bool(os.environ.get(env))
            for m in spec.get("models", []):
                out.append({"provider": name, "model": m, "label": spec.get("label", name),
                            "ready": ready})
        return out

    def provider_status(self):
        return [{"name": n, "label": s.get("label", n), "base_url": s.get("base_url"),
                 "api_key_env": s.get("api_key_env"),
                 "key_set": s.get("api_key_env") is None or bool(os.environ.get(s["api_key_env"])),
                 "models": s.get("models", [])} for n, s in self.providers.items()]

    def default_model(self):
        ms = self.models()
        ready = [m for m in ms if m["ready"] and m["provider"] != "demo"] or \
            [m for m in ms if m["ready"]] or ms
        return ready[0] if ready else None

    # sessions ------------------------------------------------------------
    def _patches(self, preset):
        rows = {r["id"] for r in compose(preset)}
        ops = []
        if "skills" in rows:
            ops.append({"replace": "skills", "config": {"dirs": self.skill_dirs}})
        if self.mcp_config:
            ops.append({"insert": [{"id": "mcp", "name": "mcp",
                                    "config": {"path": self.mcp_config}}]})
        return [ops] if ops else []

    def _make_agent(self, sid, meta, seed=None):
        cfg = {"provider": meta["provider"], "model": meta["model"]}
        agent = create_agent(self.ctx, sid, cfg, meta.get("preset", self.default_preset),
                             patches=self._patches(meta.get("preset", self.default_preset)),
                             seed=seed, max_steps=meta.get("max_steps", 40))
        live = LiveSession(self, agent, meta)
        agent.ctx.on("tools/pre-execute", lambda exe, nxt: self._permission(live, exe, nxt))
        return live

    def create(self, preset=None, provider=None, model=None, permission="auto"):
        preset = preset or self.default_preset
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        if provider is None:
            d = self.default_model()
            if d is None:
                raise ValueError("no model provider configured")
            provider, model = d["provider"], d["model"]
        if provider not in self.providers:
            raise ValueError(f"unknown provider {provider!r}")
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        meta = {"title": "", "preset": preset, "provider": provider, "model": model,
                "created": time.time(), "updated": time.time(), "permission": permission}
        with self._lock:
            live = self._make_agent(sid, meta)
            self.live[sid] = live
        self.store.save_meta(sid, meta)
        return live

    def get(self, sid):
        with self._lock:
            live = self.live.get(sid)
            if live is not None:
                return live
            events, meta = self.store.load(sid)
            if not meta:
                return None
            s = Session(events, session_id=sid)
            for ev in s.repair():   # the server stopped mid-turn: seal it
                self.store.append(sid, ev)
            live = self._make_agent(sid, meta, seed=s.events)
            self.live[sid] = live
            return live

    def delete(self, sid):
        with self._lock:
            live = self.live.pop(sid, None)
        if live is not None:
            live.cancel()
            for f in getattr(live.agent, "preset_forks", ()):
                f.dispose()
            unsubscribe = self.ctx.sessions._open.pop(sid, None)
            if unsubscribe is not None:
                unsubscribe()
            self.ctx.agents._live.pop(sid, None)
        self.store.delete(sid)

    def list(self):
        out = []
        for m in self.store.list():
            live = self.live.get(m["id"])
            out.append({"id": m["id"], "title": m.get("title") or "New session",
                        "updated": m.get("updated", 0), "preset": m.get("preset"),
                        "model": m.get("model"),
                        "status": "running" if live is not None and live.running else "idle"})
        return out

    # host seams ----------------------------------------------------------
    def _live_for(self, scope):
        for live in list(self.live.values()):
            if live.agent.key is scope:
                return live
        return None

    def _chunk(self, agent, chunk):
        live = self._live_for(agent.key)
        if live is not None:
            live.push({"kind": "chunk", "chunk": chunk})

    def _questions(self, exe, questions):
        live = self._live_for(exe.scope)
        if live is None:
            return None
        return live.ask("question", {"tool": exe.name, "questions": questions})

    def _approval(self, exe, question):
        live = self._live_for(exe.scope)
        if live is None:
            return False
        ans = live.ask("approval", {"tool": exe.name, "question": question,
                                    "arguments": exe.arguments})
        return ans == "allow"

    def _permission(self, live, exe, nxt):
        guarded = exe.name in WRITE_TOOLS or exe.name.startswith("mcp__")
        if exe.parent is None and guarded:
            if live.permission == "read-only":
                return {"action": "deny", "reason": "the session is read-only; "
                        "ask the user to change the permission mode"}
            if live.permission == "ask":
                return {"action": "ask", "question": f"Allow {exe.name}?"}
        return nxt(exe)

    # read-only views -------------------------------------------------------
    def skills(self):
        prov = FilesystemSkills(default_roots(self.workspace, self.skill_dirs))
        return [{"name": s["name"], "description": s["description"], "source": s["source"],
                 "user": s["invocation"]["user"], "model": s["invocation"]["model"]}
                for s in sorted(prov.list(), key=lambda s: (s["name"], s["rank"]))
                if s["invocation"]["user"] or s["invocation"]["model"]]

    def presets(self):
        return [{"id": k, "rows": [r["id"] for r in v if not r.get("disabled")]}
                for k, v in PRESETS.items()]

    def state(self):
        return {"workspace": str(self.workspace), "workspace_name": self.workspace.name,
                "data_dir": str(self.store.dir), "sessions": self.list(),
                "models": self.models(), "providers": self.provider_status(),
                "default_model": self.default_model(), "presets": self.presets(),
                "default_preset": self.default_preset,
                "mcp_config": self.mcp_config}


# ----- HTTP --------------------------------------------------------------------

def _loopback(host):
    host = (host or "").strip().lower()
    if host.startswith("["):
        name = host[1:].split("]", 1)[0]
    else:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return name in ("127.0.0.1", "localhost", "::1")


class Handler(BaseHTTPRequestHandler):
    app: HarnessApp = None
    server_version = "hsh-web/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet by default
        if os.environ.get("HSH_WEB_LOG"):
            super().log_message(fmt, *args)

    # helpers -------------------------------------------------------------
    def _headers(self, code, ctype, length=None, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                         "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                         "base-uri 'none'; form-action 'none'")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self._headers(code, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _error(self, code, msg):
        self._json(code, {"error": msg})

    def _allowed(self, write):
        if not _loopback(self.headers.get("Host")) and not self.server.allow_remote:
            self._error(403, "non-loopback Host refused")
            return False
        if write:
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin is not None and urlsplit(origin).netloc != host:
                self._error(403, "cross-origin request refused")
                return False
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._error(415, "send application/json")
                return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # routes ----------------------------------------------------------------
    def do_GET(self):
        if not self._allowed(False):
            return
        url = urlsplit(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html") or path.startswith("/s/"):
                return self._static("index.html")
            if path in ("/app.js", "/app.css", "/icon.svg"):
                return self._static(path[1:])
            if path == "/api/state":
                return self._json(200, self.app.state())
            if path == "/api/skills":
                return self._json(200, self.app.skills())
            m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)", path)
            if m:
                live = self.app.get(m.group(1))
                return self._json(200, live.snapshot()) if live else self._error(404, "no such session")
            m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)/stream", path)
            if m:
                live = self.app.get(m.group(1))
                if live is None:
                    return self._error(404, "no such session")
                q = parse_qs(url.query)
                return self._stream(live, int(q.get("cursor", ["0"])[0]))
            m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)/messages", path)
            if m:
                live = self.app.get(m.group(1))
                if live is None:
                    return self._error(404, "no such session")
                msgs = derive_messages(live.agent.session.events)
                return self._json(200, msgs)
            return self._error(404, "not found")
        except (ValueError, KeyError) as exc:
            return self._error(400, str(exc))

    def do_POST(self):
        if not self._allowed(True):
            return
        path = urlsplit(self.path).path
        try:
            body = self._body()
            if path == "/api/sessions":
                live = self.app.create(body.get("preset"), body.get("provider"),
                                       body.get("model"), body.get("permission", "auto"))
                if body.get("text"):
                    live.send(str(body["text"]))
                return self._json(200, {"id": live.agent.session.id})
            m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)/(\w+)", path)
            if not m:
                return self._error(404, "not found")
            live = self.app.get(m.group(1))
            if live is None:
                return self._error(404, "no such session")
            op = m.group(2)
            if op == "messages":
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._error(400, "empty message")
                live.send(text)
            elif op == "cancel":
                live.cancel()
            elif op == "answer":
                if not live.answer(str(body.get("qid")), body.get("answer")):
                    return self._error(409, "that question is no longer open")
            elif op == "settings":
                if "permission" in body:
                    if body["permission"] not in ("auto", "ask", "read-only"):
                        return self._error(400, "bad permission mode")
                    live.permission = live.meta["permission"] = body["permission"]
                if "model" in body and "provider" in body:
                    if body["provider"] not in self.app.providers:
                        return self._error(400, "unknown provider")
                    live.agent.config.update(provider=body["provider"], model=body["model"])
                    live.meta.update(provider=body["provider"], model=body["model"])
                if "plan" in body:
                    set_plan_mode(live.agent, bool(body["plan"]))
                if "title" in body:
                    live.meta["title"] = str(body["title"])[:120]
                self.app.store.save_meta(live.agent.session.id, live.meta)
                live.push({"kind": "meta", "meta": live.meta, "permission": live.permission,
                           "plan": plan_state(live.agent)})
            elif op == "delete":
                self.app.delete(live.agent.session.id)
            else:
                return self._error(404, "not found")
            return self._json(200, {"ok": True})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))

    def _static(self, name):
        p = STATIC / name
        if not p.is_file():
            return self._error(404, "not found")
        data = p.read_bytes()
        ctype = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
                 "css": "text/css; charset=utf-8", "svg": "image/svg+xml"}[name.rsplit(".", 1)[1]]
        self._headers(200, ctype, len(data))
        self.wfile.write(data)

    def _stream(self, live, cursor):
        last = self.headers.get("Last-Event-ID")
        if last and last.isdigit():   # a browser reconnecting resumes where it was
            cursor = int(last)
        self._headers(200, "text/event-stream", extra={"Connection": "close"})
        self.close_connection = True
        deadline = time.time() + self.server.stream_seconds
        try:
            while time.time() < deadline and not self.server.closing:
                frames, cursor = live.since(cursor, 15)
                if frames:
                    payload = "".join(f"id: {cursor}\ndata: {json.dumps(f, ensure_ascii=False)}\n\n"
                                      for f in frames)
                    self.wfile.write(payload.encode())
                else:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_remote = False
    stream_seconds = 600
    closing = False


def serve(app, host="127.0.0.1", port=3080):
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = Server((host, port), handler)
    return srv


def load_providers(args):
    providers = dict(DEFAULT_PROVIDERS)
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            providers = json.load(fh).get("providers", providers)
    if args.openai_base:
        providers["local"] = {"label": "OpenAI-compatible", "base_url": args.openai_base,
                              "api_key_env": args.openai_key_env, "stream": True,
                              "models": [m for m in args.openai_models.split(",") if m]}
    return providers


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m helioslm_v5.hsh.web",
                                 description="Local web UI for the hsh harness.")
    ap.add_argument("--workspace", default=".", help="directory the agents work in")
    ap.add_argument("--port", type=int, default=3080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; anything but loopback also needs --allow-remote")
    ap.add_argument("--allow-remote", action="store_true",
                    help="accept non-loopback Host headers (only behind your own proxy)")
    ap.add_argument("--data-dir", default="~/.hsh/web", help="where session logs are kept")
    ap.add_argument("--preset", default="standard", choices=sorted(PRESETS))
    ap.add_argument("--config", help="JSON file with {\"providers\": {...}}")
    ap.add_argument("--openai-base", help="an OpenAI-compatible server, e.g. "
                    "http://127.0.0.1:8080/v1 (llama.cpp, vLLM, SGLang)")
    ap.add_argument("--openai-models", default="local", help="comma-separated model ids")
    ap.add_argument("--openai-key-env", default=None, help="env var with its key, if any")
    ap.add_argument("--skills", action="append", default=[], help="extra skill directory")
    ap.add_argument("--mcp", help="MCP config (.mcp.json), relative to the workspace")
    ap.add_argument("--demo", action="store_true", help="add a keyless demo model")
    ap.add_argument("--open", action="store_true", help="open a browser")
    ap.add_argument("--egress", choices=["deny", "audit"], help="record network access "
                    "from bash/run_code; deny blocks hosts not given with --egress-allow")
    ap.add_argument("--egress-allow", action="append", default=[], metavar="HOST",
                    help="host pattern the tools may reach (e.g. '*.pypi.org'), repeatable")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        ap.error("--host other than loopback needs --allow-remote")
    app = HarnessApp(args.workspace, args.data_dir, load_providers(args), demo=args.demo,
                     preset=args.preset, skill_dirs=args.skills, mcp_config=args.mcp,
                     egress=({"mode": args.egress, "allow": args.egress_allow}
                             if args.egress else None))
    srv = serve(app, args.host, args.port)
    srv.allow_remote = args.allow_remote
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"hsh web UI on {url}  (workspace {app.workspace})", flush=True)
    for p in app.provider_status():
        if p["api_key_env"] and not p["key_set"]:
            print(f"  note: {p['api_key_env']} is not set; {p['label']} models will fail",
                  flush=True)
    if args.open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.closing = True
        srv.server_close()


if __name__ == "__main__":
    main()
