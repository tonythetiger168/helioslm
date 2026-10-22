"""hsh H6: subagents, agent instructions and the Claude Code hooks bridge.

``subagent_tool`` (after dsh's ``tool-subagent``): the ``subagent`` tool
starts a fresh child agent on its own session with its own preset, runs
the task to completion and returns the child's final answer. Lineage is
data, not visibility (dsh glossary): the child records its parent and
depth, gets none of the parent's scoped tools, and cannot go deeper than
``max_depth``.

``agent_instructions`` (after dsh's ``agent-instructions``): reads
``AGENTS.md`` (or ``CLAUDE.md``) from the workspace root into the system
prompt, capped at ``max_bytes``.

``claude_hooks`` (after dsh's ``dsh-hooks-claude-code`` compatibility
bridge): runs an existing Claude Code ``hooks.json`` against hsh's
interception points. Supported events and what they map to:

- ``PreToolUse``    -> ``tools/pre-execute``: exit 2 or
  ``permissionDecision: deny`` denies, ``ask`` asks, ``allow`` allows;
- ``PostToolUse``   -> ``tools/post-execute``: exit 2 or
  ``decision: block`` turns the result into an error carrying the reason;
  ``additionalContext`` is deferred to the next request;
- ``UserPromptSubmit`` -> ``agent/pre-step``: block rejects the turn;
  plain stdout is added as context;
- ``Stop``          -> ``agent/turn-stopping``: ``decision: block`` with a
  reason continues the turn with that reason.

A hook gets the Claude Code JSON payload on stdin and runs in the
workspace with a timeout. Rewriting a tool's arguments is not supported
(the logged call runs as logged, the same rule dsh applies).
"""
from __future__ import annotations

import json
import re
import subprocess

from .tools import ToolDefinition


# ----- subagent ------------------------------------------------------------

def subagent_tool(ctx, config=None):
    cfg = config or {}
    max_depth = cfg.get("max_depth", 1)
    preset = cfg.get("preset", "standard")
    counter = [0]

    def run(args, exe):
        from .presets import create_agent
        parent = next((a for a in ctx.agents._live.values() if a.key is exe.scope), None)
        if parent is None:
            raise RuntimeError("subagent must be called by an agent")
        depth = getattr(parent, "depth", 0) + 1
        if depth > max_depth:
            raise RuntimeError(f"subagent depth limit {max_depth} reached")
        counter[0] += 1
        child_id = f"{parent.session.id}/sub{counter[0]}"
        child = create_agent(ctx, child_id, dict(parent.config),
                             args.get("preset", preset),
                             system_prompt=cfg.get("system_prompt", ""),
                             max_steps=cfg.get("max_steps", parent.max_steps))
        child.depth = depth
        child.parent_session = parent.session.id
        answer = child.send(args["task"])
        return {"answer": answer or "", "session": child_id}

    ctx.tools.register(ToolDefinition(
        "subagent", "Delegate a self-contained task to a fresh agent with its own "
        "context. Give it everything it needs in `task`; it returns its final answer.",
        {"type": "object", "properties": {
            "task": {"type": "string"},
            "preset": {"enum": ["minimal", "standard", "code"]}},
         "required": ["task"], "additionalProperties": False},
        run, render=lambda a, v: v["answer"] or "(no answer)",
        meta=None), ctx=ctx)


subagent_tool.inject = ["tools", "agents"]


# ----- AGENTS.md -----------------------------------------------------------

def agent_instructions(ctx, config=None):
    cfg = config or {}
    names = cfg.get("files", ["AGENTS.md", "CLAUDE.md"])
    cap = cfg.get("max_bytes", 65_536)

    def assemble(agent, prompt, nxt):
        for name in names:
            p = ctx.workspace.root / name
            if p.is_file():
                text = p.read_bytes()[:cap].decode("utf-8", "ignore")
                section = f"# Instructions from {name}\n\n{text}"
                prompt = "\n\n".join(x for x in (prompt, section) if x)
                break
        return nxt(agent, prompt)

    ctx.on("system-prompt/assemble", assemble)


agent_instructions.inject = ["agents", "workspace"]


# ----- Claude Code hooks bridge ---------------------------------------------

def _matching(groups, tool_name=None):
    for g in groups or ():
        m = g.get("matcher", "")
        if tool_name is None or m in ("", "*") or re.fullmatch(m, tool_name):
            for h in g.get("hooks", ()):
                if h.get("type", "command") == "command":
                    yield h


def _run_hook(hook, payload, cwd):
    try:
        p = subprocess.run(hook["command"], shell=True, cwd=cwd, input=json.dumps(payload),
                           capture_output=True, text=True, timeout=hook.get("timeout", 60))
    except subprocess.TimeoutExpired:
        return {"code": None, "stdout": "", "stderr": "hook timed out", "json": None}
    try:
        parsed = json.loads(p.stdout) if p.stdout.strip().startswith("{") else None
    except json.JSONDecodeError:
        parsed = None
    return {"code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip(),
            "json": parsed}


def claude_hooks(ctx, config=None):
    cfg = config or {}
    spec = cfg.get("hooks")
    if spec is None and cfg.get("path"):
        with open(cfg["path"], encoding="utf-8") as fh:
            spec = json.load(fh)
    spec = (spec or {}).get("hooks", spec or {})
    cwd = str(ctx.workspace.root) if ctx.has("workspace") else None

    def agent_of(scope):
        for a in ctx.agents._live.values():
            if a.key is scope:
                return a
        return None

    def session_id(scope):
        a = agent_of(scope)
        return a.session.id if a else None

    def pre(exe, nxt):
        for h in _matching(spec.get("PreToolUse"), exe.name):
            r = _run_hook(h, {"hook_event_name": "PreToolUse", "tool_name": exe.name,
                              "tool_input": exe.arguments,
                              "session_id": session_id(exe.scope)}, cwd)
            if r["code"] == 2:
                return {"action": "deny", "reason": r["stderr"] or "blocked by hook"}
            out = ((r["json"] or {}).get("hookSpecificOutput") or {})
            decision = out.get("permissionDecision")
            reason = out.get("permissionDecisionReason", "")
            if decision == "deny":
                return {"action": "deny", "reason": reason or "denied by hook"}
            if decision == "ask":
                return {"action": "ask", "question": reason or f"Allow {exe.name}?"}
            if decision == "allow":
                return {"action": "allow"}
        return nxt()

    def post(exe, result, nxt):
        for h in _matching(spec.get("PostToolUse"), exe.name):
            r = _run_hook(h, {"hook_event_name": "PostToolUse", "tool_name": exe.name,
                              "tool_input": exe.arguments,
                              "tool_response": result.to_json(),
                              "session_id": session_id(exe.scope)}, cwd)
            j = r["json"] or {}
            if r["code"] == 2 or j.get("decision") == "block":
                reason = j.get("reason") or r["stderr"] or "blocked by hook"
                result = result.replace(content=f"{result.content}\n[hook] {reason}",
                                        is_error=True,
                                        error={"name": exe.name, "code": "hook-blocked",
                                               "reason": reason})
            extra = (j.get("hookSpecificOutput") or {}).get("additionalContext")
            if extra:
                exe.defer_context(extra, source="hook")
        return nxt(exe, result)

    def prompt(agent, decision, nxt):
        groups = spec.get("UserPromptSubmit")
        human = [m for m in decision.get("messages", []) if m["source"] == "human"]
        if groups and human:
            for h in _matching(groups):
                r = _run_hook(h, {"hook_event_name": "UserPromptSubmit",
                                  "prompt": human[-1]["content"],
                                  "session_id": agent.session.id}, cwd)
                j = r["json"] or {}
                if r["code"] == 2 or j.get("decision") == "block":
                    return {"action": "reject"}
                if r["code"] == 0 and r["stdout"] and r["json"] is None:
                    decision = {**decision, "messages": decision["messages"] + [
                        {"content": r["stdout"], "source": "hook"}]}
        return nxt(agent, decision)

    def stop(agent, out):
        for h in _matching(spec.get("Stop")):
            r = _run_hook(h, {"hook_event_name": "Stop", "session_id": agent.session.id,
                              "stop_hook_active": False,
                              "last_assistant_message": out.get("content", "")}, cwd)
            j = r["json"] or {}
            if j.get("decision") == "block" and j.get("reason"):
                return j["reason"]
        return None

    if spec.get("PreToolUse"):
        ctx.on("tools/pre-execute", pre)
    if spec.get("PostToolUse"):
        ctx.on("tools/post-execute", post)
    if spec.get("UserPromptSubmit"):
        ctx.on("agent/pre-step", prompt)
    if spec.get("Stop"):
        ctx.on("agent/turn-stopping", stop)


claude_hooks.inject = ["tools", "agents"]
