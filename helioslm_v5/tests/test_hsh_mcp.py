"""hsh: the MCP client (hsh/mcp.py).

Pins (against a small stdio MCP server written in this file):
1. handshake, paged tools/list, tools appear as mcp__<server>__<tool> with
   the server's schema; a call goes through the hsh pipeline and returns
   text plus the full MCP result as the value; isError is a tool error;
2. the child environment drops secret-looking names and expands ${VAR}
   only for the entry that names it;
3. resources: list and read through the shared tools;
4. server instructions reach the logged system prompt; requests still
   rebuild from the log;
5. a server that cannot start is recorded and adds no tools; disposing the
   row closes the process;
6. name normalisation is stable and collision-free; .mcp.json loads;
   Streamable HTTP parses JSON and SSE replies and keeps the session id.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, ScriptedAdapter,  # noqa: E402
                         Sessions, Tools, Workspace, create_agent,
                         rebuild_requests, request_hash)
from helioslm_v5.hsh.mcp import (HttpTransport, MCPError, MCPServer, child_env, mcp_plugin,  # noqa: E402
                             render_content, tool_name)

CFG = {"provider": "fake", "model": "m"}

SERVER = r'''
import json, os, sys
TOOLS = [
  {"name": "echo", "description": "Echo text", "inputSchema": {"type": "object",
   "properties": {"text": {"type": "string"}}, "required": ["text"]}},
  {"name": "env", "description": "Show env", "inputSchema": {"type": "object"}},
  {"name": "boom", "description": "Always fails", "inputSchema": {"type": "object"}},
]
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue
    meth, p, r = m["method"], m.get("params") or {}, None
    if meth == "initialize":
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})
        r = {"protocolVersion": p["protocolVersion"], "serverInfo": {"name": "evil-rename"},
             "capabilities": {"tools": {}, "resources": {}},
             "instructions": "Prefer echo for greetings."}
    elif meth == "tools/list":
        r = {"tools": TOOLS[:2], "nextCursor": "p2"} if not p.get("cursor") else {"tools": TOOLS[2:]}
    elif meth == "tools/call":
        a = p.get("arguments") or {}
        if p["name"] == "echo":
            r = {"content": [{"type": "text", "text": "echo: " + a["text"]},
                             {"type": "image", "mimeType": "image/png", "data": "AAAA"}]}
        elif p["name"] == "env":
            r = {"content": [{"type": "text", "text": json.dumps(
                 {k: os.environ.get(k) for k in ("MY_TOKEN", "PLAIN", "PASSED")})}]}
        else:
            r = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
    elif meth == "resources/list":
        r = {"resources": [{"uri": "mem://a", "name": "a", "mimeType": "text/plain"}]}
    elif meth == "resources/read":
        r = {"contents": [{"uri": p["uri"], "text": "content of " + p["uri"]}]}
    if r is None:
        send({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "nope"}})
    else:
        send({"jsonrpc": "2.0", "id": m["id"], "result": r})
'''


def world(tmp_path, adapter=None):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter or ScriptedAdapter(fn=lambda r, i: "ok"))
    return ctx


def servers(tmp_path, **extra):
    script = tmp_path / "server.py"
    script.write_text(SERVER, encoding="utf-8")
    spec = {"command": sys.executable, "args": ["-I", str(script)]}
    spec.update(extra)
    return {"servers": {"demo": spec}, "timeout_s": 20}


def call(ctx, scope, name, args):
    return ctx.tools.execute("t", name, args, scope=scope)


def test_tools_and_results(tmp_path):
    ctx = world(tmp_path)
    fork = ctx.plugin(mcp_plugin, servers(tmp_path))
    names = [s["name"] for s in ctx.tools.schemas()]
    assert names == ["list_mcp_resources", "mcp__demo__boom", "mcp__demo__echo",
                     "mcp__demo__env", "read_mcp_resource"]
    echo = next(s for s in ctx.tools.schemas() if s["name"] == "mcp__demo__echo")
    assert echo["parameters"]["required"] == ["text"]
    r = call(ctx, None, "mcp__demo__echo", {"text": "hi"})
    assert not r.is_error and r.content.startswith("echo: hi\n[image image/png")
    assert r.value["content"][0]["text"] == "echo: hi"
    assert call(ctx, None, "mcp__demo__echo", {}).error["code"] == "invalid-arguments"
    r = call(ctx, None, "mcp__demo__boom", {})
    assert r.is_error and "it broke" in r.content
    assert "mem://a" in call(ctx, None, "list_mcp_resources", {"server": "demo"}).content
    r = call(ctx, None, "read_mcp_resource", {"server": "demo", "uri": "mem://a"})
    assert r.content == "content of mem://a"
    assert call(ctx, None, "read_mcp_resource", {"server": "x", "uri": "u"}).is_error
    fork.dispose()
    assert ctx.tools.schemas() == []
    srv = MCPServer("demo", servers(tmp_path)["servers"]["demo"], 20)
    assert srv.instructions.startswith("Prefer echo") and srv.transport.proc.poll() is None
    srv.close()
    assert srv.transport.proc.poll() is not None


def test_env_scrub(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "leak")
    monkeypatch.setenv("PLAIN", "ok")
    monkeypatch.setenv("UPSTREAM_SECRET", "s3")
    ctx = world(tmp_path)
    ctx.plugin(mcp_plugin, servers(tmp_path, env={"PASSED": "${UPSTREAM_SECRET}"}))
    seen = json.loads(call(ctx, None, "mcp__demo__env", {}).content)
    assert seen == {"MY_TOKEN": None, "PLAIN": "ok", "PASSED": "s3"}
    env = child_env({"A": "${PLAIN}-x"}, {"PLAIN": "p", "API_KEY": "k", "HSH_X": "1"})
    assert env == {"PLAIN": "p", "A": "p-x"}


def test_instructions_in_prompt_and_rebuild(tmp_path):
    ad = ScriptedAdapter(replies=[
        {"tool_calls": [{"id": "c1", "name": "mcp__demo__echo",
                         "arguments": json.dumps({"text": "yo"})}]}, "done"])
    ctx = world(tmp_path, ad)
    rows = [[{"insert": [{"id": "mcp", "name": "mcp", "config": servers(tmp_path)}]}]]
    a = create_agent(ctx, "a", CFG, "minimal", patches=rows)
    assert a.send("greet") == "done"
    sys_msg = ad.requests[0]["messages"][0]["content"]
    assert "Instructions from MCP server demo" in sys_msg and "Prefer echo" in sys_msg
    assert ad.requests[1]["messages"][-1]["content"].startswith("echo: yo")
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]
    assert ctx.tools.schemas() == []   # mounted on the agent only


def test_failed_server_recorded(tmp_path):
    ctx = world(tmp_path)
    ctx.plugin(mcp_plugin, {"servers": {"gone": {"command": str(tmp_path / "nope")}}})
    assert [s["name"] for s in ctx.tools.schemas()] == ["list_mcp_resources",
                                                         "read_mcp_resource"]
    assert any(k == "mcp/gone" for k, _ in ctx.errors)
    with pytest.raises(MCPError):   # fail_on_error raises to the caller
        ctx.plugin(mcp_plugin, {"servers": {"bad name!": {"command": "x"}},
                                "fail_on_error": True})


def test_names_config_and_http(tmp_path):
    assert tool_name("gh", "create_issue") == "mcp__gh__create_issue"
    a, b = tool_name("s", "a.b"), tool_name("s", "a/b")
    assert a != b and a.startswith("mcp__s__a_b_") and len(a) <= 64
    assert tool_name("s", "x" * 80) == tool_name("s", "x" * 80)
    assert len(tool_name("s", "x" * 80)) <= 64
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": servers(tmp_path)["servers"]}))
    ctx = world(tmp_path)
    ctx.plugin(mcp_plugin, {"path": ".mcp.json"})
    assert call(ctx, None, "mcp__demo__echo", {"text": "x"}).content.startswith("echo: x")
    assert render_content({"structuredContent": {"a": 1}}) == '{"a": 1}'

    sent = []

    class Resp:
        def __init__(self, body, ctype, sid=None):
            self._b, self.headers = body.encode(), {"Content-Type": ctype}
            if sid:
                self.headers["Mcp-Session-Id"] = sid

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        msg = json.loads(req.data)
        sent.append((dict(req.headers), msg))
        if "id" not in msg:
            return Resp("", "application/json")
        if msg["method"] == "initialize":
            return Resp(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "capabilities": {"tools": {}}}}), "application/json", sid="S1")
        body = ("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "x"}) +
                "\n\ndata: " + json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {
                    "tools": [{"name": "t", "inputSchema": {"type": "object"}}]}}) + "\n\n")
        return Resp(body, "text/event-stream")

    t = HttpTransport("https://example.invalid/mcp", {"Authorization": "Bearer ${NOPE_X}"},
                      opener=opener)
    t.request("initialize", {})
    t.notify("notifications/initialized")
    assert t.request("tools/list", {})["tools"][0]["name"] == "t"
    hdrs = {k.lower(): v for k, v in sent[-1][0].items()}
    assert hdrs["mcp-session-id"] == "S1" and hdrs["authorization"] == "Bearer "
    assert t.notifications == ["x"]
