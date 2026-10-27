"""hsh web UI server (hsh/web.py).

Pins:
1. the page and its assets are served with a strict CSP; a non-loopback
   Host is refused (DNS rebinding); a cross-origin or non-JSON POST is
   refused;
2. a session runs a turn in the background and the event stream carries
   the logged events, the status changes and the final reply;
3. ask_user_question and tool approval (permission "ask") round-trip
   through the page's answer endpoint; read-only refuses writes;
4. sessions are saved and come back after a restart (a turn cut off
   mid-way is sealed as interrupted); cancel stops a running turn; delete
   removes the log;
5. API keys are never accepted or echoed: the state lists only whether
   each environment variable is set.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import ScriptedAdapter  # noqa: E402
from helioslm_v5.hsh.web import HarnessApp, Session, _loopback, serve  # noqa: E402


class Running:
    def __init__(self, app):
        self.app = app
        self.srv = serve(app, "127.0.0.1", 0)
        self.srv.stream_seconds = 5
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path, headers=None):
        req = urllib.request.Request(self.url(path), headers=headers or {})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()

    def post(self, path, body=None, headers=None):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        req = urllib.request.Request(self.url(path), data=json.dumps(body or {}).encode(),
                                     headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def frames(self, sid, until, cursor=0, timeout=10):
        """Read the SSE stream until ``until(frames)`` is true."""
        got = []
        req = urllib.request.Request(self.url(f"/api/sessions/{sid}/stream?cursor={cursor}"))
        deadline = time.time() + timeout
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    got.append(json.loads(line[5:]))
                    if until(got):
                        return got
                if time.time() > deadline:
                    break
        raise AssertionError(f"stream ended early: {[f['kind'] for f in got]}")

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def idle(frames):
    return any(f["kind"] == "status" and f["status"] == "idle" for f in frames)


def make(tmp_path, fn, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "hello.txt").write_text("hi")
    return HarnessApp(ws, tmp_path / "data", providers={
        "fake": {"label": "Fake", "adapter": ScriptedAdapter(fn=fn), "models": ["m1"],
                 "api_key_env": None}}, **kw)


@pytest.fixture
def server(tmp_path):
    made = []

    def start(fn, **kw):
        r = Running(make(tmp_path, fn, **kw))
        made.append(r)
        return r

    yield start
    for r in made:
        r.close()


def test_page_and_guards(server, monkeypatch):
    monkeypatch.setenv("HSH_WEB_TEST_KEY", "sk-secret-value")
    r = server(lambda req, i: "ok")
    r.app.providers["fake"]["api_key_env"] = "HSH_WEB_TEST_KEY"
    code, headers, body = r.get("/")
    assert code == 200 and b'id="app"' in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    for asset in ("/app.js", "/app.css", "/icon.svg"):
        assert r.get(asset)[0] == 200
    state = json.loads(r.get("/api/state")[2])
    assert state["providers"][0]["key_set"] is True
    assert "sk-secret-value" not in json.dumps(state)
    with pytest.raises(urllib.error.HTTPError) as e:
        r.get("/api/state", {"Host": "evil.example:3080"})
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        r.post("/api/sessions", {}, {"Origin": "http://evil.example"})
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        r.post("/api/sessions", {}, {"Content-Type": "text/plain"})
    assert e.value.code == 415
    assert _loopback("127.0.0.1:3080") and _loopback("[::1]:80") and _loopback("localhost")
    assert not _loopback("127.0.0.1.evil.com") and not _loopback("")


def test_turn_streams(server):
    def fn(req, i):
        if req["messages"][-1]["role"] == "tool":
            return {"content": "found " + req["messages"][-1]["content"]}
        return {"tool_calls": [{"id": "g", "name": "glob", "arguments": '{"pattern": "*.txt"}'}]}

    r = server(fn)
    sid = r.post("/api/sessions", {"text": "list files"})["id"]
    frames = r.frames(sid, idle)
    types = [f["event"]["type"] for f in frames if f["kind"] == "event"]
    assert "tool/result" in types and types[-1] == "turn/end"
    snap = json.loads(r.get(f"/api/sessions/{sid}")[2])
    assert snap["status"] == "idle" and snap["meta"]["title"] == "list files"
    last = [e for e in snap["events"] if e["type"] == "assistant/message"][-1]
    assert last["data"]["message"]["content"] == "found hello.txt"
    assert all("time" in e for e in snap["events"])
    sessions = json.loads(r.get("/api/state")[2])["sessions"]
    assert [s["id"] for s in sessions] == [sid]


def test_question_and_approval(server):
    def fn(req, i):
        m = req["messages"][-1]
        if m["role"] == "tool":
            return {"content": "result: " + m["content"]}
        if "ask" in m["content"]:
            return {"tool_calls": [{"id": "q", "name": "ask_user_question", "arguments": json.dumps(
                {"questions": [{"id": "c", "question": "Colour?", "options": [{"label": "Blue"}]}]})}]}
        return {"tool_calls": [{"id": "w", "name": "write",
                                "arguments": '{"path": "new.txt", "content": "x"}'}]}

    r = server(fn)
    sid = r.post("/api/sessions", {"text": "ask me"})["id"]
    q = r.frames(sid, lambda fs: any(f["kind"] == "question" for f in fs))[-1]
    assert q["questions"][0]["question"] == "Colour?"
    r.post(f"/api/sessions/{sid}/answer", {"qid": q["qid"], "answer": {"c": "Blue"}})
    snap = r.app.get(sid)
    for _ in range(100):
        if not snap.running:
            break
        time.sleep(0.05)
    msgs = [e for e in snap.agent.session.events if e["type"] == "assistant/message"]
    assert msgs[-1]["data"]["message"]["content"] == "result: c: Blue"

    r.post(f"/api/sessions/{sid}/settings", {"permission": "ask"})
    cur = snap.cursor()
    r.post(f"/api/sessions/{sid}/messages", {"text": "write it"})
    a = r.frames(sid, lambda fs: any(f["kind"] == "approval" for f in fs), cursor=cur)[-1]
    assert a["tool"] == "write" and a["arguments"]["path"] == "new.txt"
    r.post(f"/api/sessions/{sid}/answer", {"qid": a["qid"], "answer": "deny"})
    r.frames(sid, idle, cursor=cur)
    assert not (r.app.workspace / "new.txt").exists()

    r.post(f"/api/sessions/{sid}/settings", {"permission": "read-only"})
    cur = snap.cursor()
    r.post(f"/api/sessions/{sid}/messages", {"text": "write it"})
    r.frames(sid, idle, cursor=cur)
    res = [e for e in snap.agent.session.events if e["type"] == "tool/result"][-1]
    assert "read-only" in res["data"]["message"]["content"]
    with pytest.raises(urllib.error.HTTPError) as e:
        r.post(f"/api/sessions/{sid}/answer", {"qid": "gone", "answer": "x"})
    assert e.value.code == 409


def test_persistence_cancel_delete(server, tmp_path):
    gate = threading.Event()

    def slow(req, i):
        gate.wait(5)   # the request in flight finishes; the next step does not start
        return {"tool_calls": [{"id": f"g{i}", "name": "glob", "arguments": '{"pattern": "*"}'}]}

    r = server(slow)
    sid = r.post("/api/sessions", {"text": "first"})["id"]
    time.sleep(0.2)
    r.post(f"/api/sessions/{sid}/cancel")
    gate.set()
    r.frames(sid, idle)
    ends = [e["data"]["reason"] for e in r.app.get(sid).agent.session.events
            if e["type"] == "turn/end"]
    assert ends == ["cancelled"]
    r.close()

    # a log cut off mid-turn comes back sealed as interrupted
    store = r.app.store
    s = Session(session_id="cut")
    s.append("turn/start", {"turn": 1})
    s.append("user/message", {"content": "hi", "source": "human"})
    for ev in s.events:
        store.append("cut", ev)
    store.save_meta("cut", {"title": "cut", "preset": "minimal", "provider": "fake",
                            "model": "m1", "updated": 1})
    r2 = server(lambda req, i: "again")
    assert {x["id"] for x in r2.app.list()} == {sid, "cut"}
    back = r2.app.get("cut")
    assert back.agent.session.events[-1]["data"]["reason"] == "interrupted"
    r2.post("/api/sessions/cut/messages", {"text": "more"})
    r2.frames("cut", idle)
    assert back.agent.session.events[-1]["type"] == "turn/end"
    r2.post("/api/sessions/cut/delete")
    assert {x["id"] for x in r2.app.list()} == {sid}
