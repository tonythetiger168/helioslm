"""hsh: Model Context Protocol client (the ``mcp`` preset row).

After dsh's ``mcp-client`` and ``mcp-resources`` (docs/subsystems/mcp.md),
in the standard library only: JSON-RPC 2.0 over a child process's
stdin/stdout (newline-delimited, the MCP stdio transport) or over
Streamable HTTP (one POST per message; a JSON or ``text/event-stream``
reply; the ``Mcp-Session-Id`` header is kept).

Config, one entry per server (the same shape as Claude Code's
``.mcp.json``, which ``{"path": ".mcp.json"}`` reads directly)::

    {"servers": {
        "github": {"command": "npx",
                   "args": ["-y", "@modelcontextprotocol/server-github"],
                   "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},
        "web":    {"url": "https://example.com/mcp",
                   "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}},
     "timeout_s": 60, "fail_on_error": false, "max_instruction_bytes": 32768}

- Each server tool becomes the hsh tool ``mcp__<server>__<tool>``; the
  namespace is the configured server name, never the server's own
  ``serverInfo.name``. A name outside ``[A-Za-z0-9_-]`` or over 64
  characters is normalised and gets a 12-hex-digit hash so two tools can
  never collapse into one name.
- A child process gets a scrubbed environment: inherited names matching
  KEY/PASSWORD/SECRET/TOKEN are dropped; the entry's ``env`` is merged on
  top with ``${VAR}`` expanded from the host environment, so a secret
  reaches only the server that names it. Header values expand the same
  way. Secrets are never written to the session log.
- Server ``instructions`` join the system prompt (logged like every other
  prompt section), capped at ``max_instruction_bytes``.
- ``list_mcp_resources`` and ``read_mcp_resource`` appear when at least
  one server is configured.
- A server that fails to start is recorded in ``ctx.errors`` and adds no
  tools (``fail_on_error`` makes it fail the row instead).

Not supported, as upstream: prompt templates, elicitation, sampling,
subscriptions. Image and audio blocks reach the model as a short text
description; the full MCP result stays in the tool result's ``value``.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import urllib.request

from .tools import ToolDefinition

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "hsh", "version": "1"}
_SECRET = re.compile(r"KEY|PASSWORD|SECRET|TOKEN", re.I)
_NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SERVER_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class MCPError(RuntimeError):
    pass


def expand(value, environ=None):
    env = os.environ if environ is None else environ
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: env.get(m.group(1), ""), value)


def child_env(extra=None, environ=None):
    env = os.environ if environ is None else environ
    out = {k: v for k, v in env.items() if not _SECRET.search(k) and not k.startswith("HSH_")}
    for k, v in (extra or {}).items():
        out[k] = expand(str(v), env)
    return out


def tool_name(server, raw):
    name = f"mcp__{server}__{raw}"
    if _NAME_OK.match(name):
        return name
    digest = hashlib.sha256(f"{server}\0{raw}".encode()).hexdigest()[:12]
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64 - 13]
    return f"{clean}_{digest}"


# ----- transports ------------------------------------------------------------

class _Rpc:
    """JSON-RPC bookkeeping shared by both transports."""

    def __init__(self, timeout):
        self.timeout = timeout
        self._next = 0
        self.notifications = []

    def _new_id(self):
        self._next += 1
        return self._next

    @staticmethod
    def _result(msg, method):
        if "error" in msg:
            err = msg["error"] or {}
            raise MCPError(f"{method}: {err.get('message', 'error')} ({err.get('code')})")
        return msg.get("result") or {}


class StdioTransport(_Rpc):
    def __init__(self, command, args=(), env=None, cwd=None, timeout=60):
        super().__init__(timeout)
        try:
            self.proc = subprocess.Popen(
                [command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=cwd, env=child_env(env), text=True,
                encoding="utf-8", bufsize=1)
        except OSError as exc:
            raise MCPError(f"cannot start {command!r}: {exc}") from None
        self._lines = queue.Queue()
        self.stderr = []
        threading.Thread(target=self._pump, args=(self.proc.stdout, self._lines),
                         daemon=True).start()
        threading.Thread(target=self._drain_err, daemon=True).start()

    @staticmethod
    def _pump(stream, q):
        for line in stream:
            q.put(line)
        q.put(None)

    def _drain_err(self):
        for line in self.proc.stderr:
            if len(self.stderr) < 200:
                self.stderr.append(line.rstrip())

    def _write(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise MCPError("server process is gone") from None

    def notify(self, method, params=None):
        self._write({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params=None):
        rid = self._new_id()
        self._write({"jsonrpc": "2.0", "id": rid, "method": method,
                     **({"params": params} if params is not None else {})})
        while True:
            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"{method}: no reply in {self.timeout}s") from None
            if line is None:
                tail = "; ".join(self.stderr[-3:])
                raise MCPError(f"{method}: server exited" + (f" ({tail})" if tail else ""))
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray output on stdout
            if msg.get("id") == rid and "method" not in msg:
                return self._result(msg, method)
            if "method" in msg and "id" in msg:   # a request from the server
                reply = {"jsonrpc": "2.0", "id": msg["id"]}
                if msg["method"] == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "not supported by hsh"}
                self._write(reply)
            elif "method" in msg:
                self.notifications.append(msg["method"])

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()


class HttpTransport(_Rpc):
    def __init__(self, url, headers=None, timeout=60, opener=None):
        super().__init__(timeout)
        self.url = url
        self.headers = {k: expand(str(v)) for k, v in (headers or {}).items()}
        self.session_id = None
        self.opener = opener or urllib.request.urlopen

    def _post(self, msg):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": PROTOCOL_VERSION, **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(msg).encode(),
                                     headers=headers, method="POST")
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8")
        except OSError as exc:
            raise MCPError(f"{msg.get('method')}: {exc}") from None
        if "text/event-stream" in ctype:
            out = []
            for block in body.split("\n\n"):
                data = "\n".join(x[5:].lstrip() for x in block.splitlines()
                                 if x.startswith("data:"))
                if data:
                    out.append(json.loads(data))
            return out
        return [json.loads(body)] if body.strip() else []

    def notify(self, method, params=None):
        self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def request(self, method, params=None):
        rid = self._new_id()
        for msg in self._post({"jsonrpc": "2.0", "id": rid, "method": method,
                               **({"params": params} if params is not None else {})}):
            if msg.get("id") == rid and "method" not in msg:
                return self._result(msg, method)
            if "method" in msg and "id" not in msg:
                self.notifications.append(msg["method"])
        raise MCPError(f"{method}: no reply")

    def close(self):
        pass


# ----- one connection -------------------------------------------------------

class MCPServer:
    """An initialised connection to one server."""

    def __init__(self, name, spec, timeout=60, cwd=None):
        if not SERVER_RE.match(name):
            raise MCPError(f"server name {name!r} must match [A-Za-z0-9_-]{{1,32}}")
        self.name = name
        if spec.get("url"):
            self.transport = HttpTransport(spec["url"], spec.get("headers"), timeout,
                                           spec.get("opener"))
        elif spec.get("command"):
            self.transport = StdioTransport(spec["command"], spec.get("args", ()),
                                            spec.get("env"), spec.get("cwd", cwd), timeout)
        else:
            raise MCPError(f"server {name!r} needs a command or a url")
        try:
            init = self.transport.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": CLIENT_INFO})
            self.transport.notify("notifications/initialized")
        except Exception:
            self.transport.close()
            raise
        self.capabilities = init.get("capabilities") or {}
        self.instructions = init.get("instructions") or ""
        self.server_info = init.get("serverInfo") or {}

    def _paged(self, method, key):
        out, cursor = [], None
        for _ in range(100):
            res = self.transport.request(method, {"cursor": cursor} if cursor else {})
            out.extend(res.get(key) or [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return out

    def list_tools(self):
        return self._paged("tools/list", "tools") if "tools" in self.capabilities else []

    def call_tool(self, raw, arguments):
        return self.transport.request("tools/call", {"name": raw, "arguments": arguments})

    def request(self, method, params=None):
        return self.transport.request(method, params)

    def close(self):
        self.transport.close()


def render_content(result):
    """Model-facing text of an MCP tool or resource result."""
    parts = []
    for block in result.get("content") or result.get("contents") or []:
        kind = block.get("type")
        if kind == "text" or (kind is None and "text" in block):
            parts.append(block["text"])
        elif kind in ("image", "audio") or "blob" in block:
            size = len(block.get("data") or block.get("blob") or "") * 3 // 4
            parts.append(f"[{kind or 'binary'} {block.get('mimeType', '?')}, about {size} bytes"
                         " (not shown to the model)]")
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(res.get("text") or f"[resource {res.get('uri', '?')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link {block.get('uri', '?')}: {block.get('name', '')}]")
        else:
            parts.append(f"[unsupported MCP content {kind!r}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def _as_schema(schema):
    s = dict(schema or {"type": "object"})
    s.setdefault("type", "object")
    if s.get("type") != "object":
        s = {"type": "object"}
    s.pop("$schema", None)
    return s


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {"servers": data.get("mcpServers", data.get("servers", {}))}


def mcp_plugin(ctx, config=None):
    """Preset row ``mcp``; see the module docstring for its config."""
    cfg = dict(config or {})
    if cfg.get("path"):
        path = cfg["path"]
        if ctx.has("workspace") and not os.path.isabs(path):
            path = str(ctx.workspace.root / path)
        cfg = dict(load_config(path), **{k: v for k, v in cfg.items() if k != "path"})
    timeout = cfg.get("timeout_s", 60)
    cap = cfg.get("max_instruction_bytes", 32_768)
    cwd = str(ctx.workspace.root) if ctx.has("workspace") else None
    servers = {}

    def close_all():
        for s in servers.values():
            s.close()
        servers.clear()

    ctx.effect(close_all)
    for name, spec in (cfg.get("servers") or {}).items():
        if spec.get("disabled"):
            continue
        try:
            srv = MCPServer(name, spec, timeout, cwd)
            servers[name] = srv
            listed = srv.list_tools()
        except Exception as exc:
            servers.pop(name, None)
            if cfg.get("fail_on_error"):
                raise
            ctx._root.errors.append((f"mcp/{name}", exc))
            continue
        for t in listed:
            raw = t["name"]

            def run(args, exe, srv=srv, raw=raw):
                res = srv.call_tool(raw, args)
                if res.get("isError"):
                    raise RuntimeError(render_content(res) or "MCP tool reported an error")
                return res

            ctx.tools.register(ToolDefinition(
                tool_name(name, raw), (t.get("description") or raw)[:2048],
                _as_schema(t.get("inputSchema")), run,
                render=lambda a, v: render_content(v) or "(no content)",
                meta={"mcp": {"server": name, "tool": raw}}), ctx=ctx)

    if not (cfg.get("servers") or {}):
        return servers

    def pick(args):
        srv = servers.get(args["server"])
        if srv is None:
            raise ValueError(f"no connected MCP server {args['server']!r}; "
                             f"connected: {sorted(servers)}")
        return srv

    def list_resources(args, exe):
        res = pick(args).request("resources/list", {"cursor": args["cursor"]}
                                 if args.get("cursor") else {})
        return res

    def read_resource(args, exe):
        return pick(args).request("resources/read", {"uri": args["uri"]})

    server_prop = {"type": "string", "description": "Configured MCP server name."}
    ctx.tools.register(ToolDefinition(
        "list_mcp_resources", "List the resources an MCP server offers.",
        {"type": "object", "properties": {"server": server_prop, "cursor": {"type": "string"}},
         "required": ["server"], "additionalProperties": False}, list_resources,
        render=lambda a, v: "\n".join(
            f"{r.get('uri')}  {r.get('name', '')}  {r.get('mimeType', '')}".rstrip()
            for r in v.get("resources") or []) or "(no resources)"), ctx=ctx)
    ctx.tools.register(ToolDefinition(
        "read_mcp_resource", "Read one resource from an MCP server by URI.",
        {"type": "object", "properties": {"server": server_prop, "uri": {"type": "string"}},
         "required": ["server", "uri"], "additionalProperties": False}, read_resource,
        render=lambda a, v: render_content(v) or "(empty)"), ctx=ctx)

    sections = []
    for name, srv in sorted(servers.items()):
        text = srv.instructions.strip()
        if text:
            section = f"# Instructions from MCP server {name}\n\n{text}"
            if len(section.encode()) > cap:
                section = section.encode()[:cap].decode("utf-8", "ignore")
            sections.append(section)
    if sections and ctx.has("agents"):
        def assemble(agent, prompt, nxt):
            return nxt(agent, "\n\n".join([p for p in (prompt,) if p] + sections))

        ctx.on("system-prompt/assemble", assemble)
    return servers


mcp_plugin.inject = ["tools"]
