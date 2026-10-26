"""hsh: ask_user_question, plan mode and web_fetch (hsh/interaction.py).

Pins:
1. ask_user_question goes through ctx.questions; answers are normalised
   (dict or list form, multi-select lists); no host means "nobody
   answered" instead of a crash; duplicate ids are refused;
2. plan mode: /plan logs plan/mode at the turn start and adds the plan
   section to the logged system prompt; exit_plan_mode refused outside
   plan mode, keeps planning with feedback, and approval switches it off
   before the next step; /plan off; set_plan_mode from a host; deny_tools;
   every request rebuilds from the log;
3. web_fetch: HTML to text, private addresses refused, redirects
   re-checked, non-text bodies summarised, size caps.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, ScriptedAdapter,  # noqa: E402
                         Sessions, Tools, Workspace, create_agent, html_to_text,
                         plan_state, rebuild_requests, request_hash, set_plan_mode,
                         web_fetch_tool)

CFG = {"provider": "fake", "model": "m"}


def world(tmp_path, adapter=None):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter or ScriptedAdapter(fn=lambda r, i: "ok"))
    return ctx


def agent(ctx, sid="a", **plan_cfg):
    rows = [[{"replace": "skills", "config": {"home": False}}]]
    if plan_cfg:
        rows[0].append({"replace": "plan-mode", "config": plan_cfg})
    return create_agent(ctx, sid, CFG, "standard", patches=rows)


def rebuilt(a):
    return a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]


def test_ask_user(tmp_path):
    ctx = world(tmp_path)
    a = agent(ctx)
    q = {"questions": [{"id": "db", "question": "Which DB?", "options": [
        {"label": "SQLite (Recommended)"}, {"label": "Postgres"}]},
        {"id": "feat", "question": "Which features?", "multi_select": True}]}
    r = ctx.tools.execute("t", "ask_user_question", q, scope=a.key)
    assert not r.is_error and "did not answer" in r.content
    seen = []
    ctx.provide("questions", lambda exe, qs: seen.append(qs) or {"db": "Postgres",
                                                                  "feat": ["a", "b"]})
    r = ctx.tools.execute("t", "ask_user_question", q, scope=a.key)
    assert r.content == "db: Postgres\nfeat: a, b" and seen[0][0]["id"] == "db"
    assert r.value[1]["answer"] == ["a", "b"]
    dup = {"questions": [{"id": "x", "question": "1"}, {"id": "x", "question": "2"}]}
    assert ctx.tools.execute("t", "ask_user_question", dup, scope=a.key).is_error


def test_plan_mode_flow(tmp_path):
    answers = iter(["Please add tests", "Approve"])
    replies = iter([
        {"tool_calls": [{"id": "1", "name": "exit_plan_mode",
                         "arguments": json.dumps({"plan": "# Plan\n1. edit"})}]},
        {"tool_calls": [{"id": "2", "name": "exit_plan_mode",
                         "arguments": json.dumps({"plan": "# Plan\n1. edit\n2. test"})}]},
        "starting work", "done"])
    ad = ScriptedAdapter(fn=lambda r, i: next(replies))
    ctx = world(tmp_path, ad)
    ctx.provide("questions", lambda exe, qs: {"plan": next(answers)})
    a = agent(ctx)
    r = ctx.tools.execute("t", "exit_plan_mode", {"plan": "# x"}, scope=a.key)
    assert r.is_error and "only in plan mode" in r.content
    assert a.send("/plan refactor the parser") == "starting work"
    sys0 = ad.requests[0]["messages"][0]["content"]
    assert "# Plan mode" in sys0
    assert ad.requests[0]["messages"][-1]["content"] == "refactor the parser"
    assert "Feedback: Please add tests" in ad.requests[1]["messages"][-1]["content"]
    # after approval the next step's system prompt has no plan section
    assert "# Plan mode" not in ad.requests[2]["messages"][0]["content"]
    assert plan_state(a) == {"active": False}
    modes = [e["data"]["active"] for e in a.session.events if e["type"] == "plan/mode"]
    assert modes == [True, False]
    assert rebuilt(a)
    # host toggles: pending until the next turn starts
    set_plan_mode(a, True)
    assert plan_state(a) == {"active": False, "pending": True}
    a.send("go on")
    assert plan_state(a) == {"active": True}
    a.send("/plan off")      # bare command: turn is rejected, mode still switches
    assert plan_state(a) == {"active": False}
    assert rebuilt(a)


def test_plan_deny_tools(tmp_path):
    ctx = world(tmp_path)
    a = agent(ctx, deny_tools=["write", "bash"])
    w = {"path": "f.txt", "content": "x"}
    assert not ctx.tools.execute("t", "write", w, scope=a.key).is_error
    a.send("/plan")
    r = ctx.tools.execute("t", "write", w, scope=a.key)
    assert r.is_error and "not allowed in plan mode" in r.content
    assert not ctx.tools.execute("t", "read", {"path": "f.txt"}, scope=a.key).is_error


class Resp(io.BytesIO):
    def __init__(self, body, ctype, status=200):
        super().__init__(body)
        self.headers = {"Content-Type": ctype}
        self.status = status


def test_web_fetch(tmp_path, monkeypatch):
    title, text = html_to_text("<html><head><title>T</title><style>x{}</style></head>"
                               "<body><h1>Hi</h1><p>one &amp; two</p><script>bad()</script>"
                               "</body></html>")
    assert title == "T" and text == "Hi\n\none & two"
    pages = {
        "https://example.com/a": ("redirect", "https://example.com/b"),
        "https://example.com/b": (b"<p>landed</p>", "text/html; charset=utf-8"),
        "https://example.com/bin": (b"\x00\x01", "application/octet-stream"),
        "https://example.com/big": (b"x" * 500, "text/plain"),
        "https://example.com/evil": ("redirect", "http://127.0.0.1/admin"),
    }

    def opener(req, timeout):
        body, ctype = pages[req.full_url]
        if body == "redirect":
            raise urllib.error.HTTPError(req.full_url, 302, "Found", {"Location": ctype}, None)
        return Resp(body, ctype)

    import socket
    real = socket.getaddrinfo

    def fake_getaddrinfo(host, *a, **kw):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        return real(host, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    ctx = world(tmp_path)
    ctx.plugin(web_fetch_tool, {"opener": opener, "max_chars": 100})
    fetch = lambda url: ctx.tools.execute("t", "web_fetch", {"url": url})  # noqa: E731
    r = fetch("https://example.com/a")
    assert not r.is_error and r.value["url"] == "https://example.com/b"
    assert r.value["text"] == "landed"
    assert "not shown" in fetch("https://example.com/bin").content
    big = fetch("https://example.com/big")
    assert big.value["truncated"] and len(big.value["text"]) == 100
    for bad in ("https://example.com/evil", "http://localhost/", "http://169.254.169.254/",
                "file:///etc/passwd", "ftp://example.com/x"):
        r = fetch(bad)
        assert r.is_error, bad
    assert "private address" in fetch("https://example.com/evil").content
