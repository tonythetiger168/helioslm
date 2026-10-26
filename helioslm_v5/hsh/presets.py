"""hsh H5: agent presets, profiles and patches.

After dsh's agent presets and Cordis config rows
(packages/bundle/web-app/presets/*.patch.yml). A preset is a list of
rows ``{"id", "name", "config"?, "disabled"?}``; ``name`` picks a plugin
from :data:`PLUGINS`. :func:`create_agent` mounts the rows on the new
agent's scoped context in the setup window, so each agent sees exactly
the tools and prompt sections of its preset.

Shipped presets (dsh's four work modes):

- ``minimal``: ``bash`` + ``str_replace_editor`` and a short persona;
- ``standard``: ``bash`` (with background jobs), ``read``/``write``/``edit``/``glob``/``grep``,
  ``todo_write``, ``subagent``, ``skill``, ``ask_user_question``,
  ``exit_plan_mode``; ``AGENTS.md`` in the system prompt; the skill
  catalog; plan mode; automatic compaction;
- ``code``: standard plus ``run_code`` in PTC mode: the model sees only
  ``run_code`` and calls the other tools from inside its program;
- ``creator``: standard plus ``inspect`` for looking at the running
  harness (plugins, services, tools, events, the session log, the preset).

Patches (applied in order, like dsh's profile layers):

- ``{"insert": [rows], "after": "<id>"}`` (default: append);
- ``{"replace": "<id>", "config": {...}}`` replaces that row's config;
- ``{"remove": "<id>"}``; ``{"disable": "<id>"}``; ``{"enable": "<id>"}``.

Patch files are JSON (YAML too when PyYAML happens to be installed; it is
not a dependency).
"""
from __future__ import annotations

import copy
import json

from .builtin import bash_tool, editor_tool, fs_tools, run_code_tool, todo_tool
from .compaction import compaction
from .interaction import ask_user_tool, plan_mode, web_fetch_tool
from .mcp import mcp_plugin
from .skills import skills_plugin
from .subagent import agent_instructions, claude_hooks, subagent_tool
from .tools import ToolDefinition


class PresetError(ValueError):
    pass


def persona(ctx, config=None):
    """Prepend ``prefix`` to the agent's system prompt (scoped listener)."""
    prefix = (config or {}).get("prefix", "")

    def assemble(agent, prompt, nxt):
        joined = "\n\n".join(p for p in (prefix, prompt) if p)
        return nxt(agent, joined)

    ctx.on("system-prompt/assemble", assemble)


persona.inject = ["agents"]


def code_mode(ctx, config=None):
    ctx.tools.ptc_mode(ctx, (config or {}).get("transport", "run_code"))


code_mode.inject = ["tools"]


def inspect_tool(ctx, config=None):
    """creator preset: read-only view of the running harness."""
    root = ctx

    def run(args, exe):
        what = args["what"]
        if what == "plugins":
            return [{"id": f.id, "name": f.name, "state": f.state} for f in root.forks]
        if what == "services":
            return sorted(root._root.services)
        if what == "events":
            return dict(sorted(root._root.modes.items()))
        if what == "tools":
            return [s["name"] for s in root.tools.schemas(exe.scope)]
        if what == "session":
            for agent in root.agents._live.values():
                if agent.key is exe.scope:
                    evs = agent.session.events
                    return [{"seq": e["seq"], "type": e["type"]} for e in evs[-args.get("last", 50):]]
            return []
        if what == "preset":
            return copy.deepcopy((config or {}).get("rows", []))
        raise ValueError(f"unknown inspect target {what!r}")

    ctx.tools.register(ToolDefinition(
        "inspect", "Inspect the running harness: plugins, services, events, tools, "
        "the session log, or this agent's preset.",
        {"type": "object", "properties": {
            "what": {"enum": ["plugins", "services", "events", "tools", "session", "preset"]},
            "last": {"type": "integer", "minimum": 1}},
         "required": ["what"], "additionalProperties": False}, run), ctx=ctx)


inspect_tool.inject = ["tools", "agents"]

PLUGINS = {
    "persona": persona,
    "tool-bash": bash_tool,
    "tool-editor": editor_tool,
    "tool-fs": fs_tools,
    "tool-todo": todo_tool,
    "tool-run-code": run_code_tool,
    "code-mode": code_mode,
    "creator-inspect": inspect_tool,
    "agent-instructions": agent_instructions,
    "compaction": compaction,
    "tool-subagent": subagent_tool,
    "claude-hooks": claude_hooks,
    "skills": skills_plugin,
    "mcp": mcp_plugin,
    "tool-ask-user": ask_user_tool,
    "plan-mode": plan_mode,
    "tool-web-fetch": web_fetch_tool,
}

_STANDARD = [
    {"id": "persona", "name": "persona",
     "config": {"prefix": "You are a coding agent running in the HeliosLM harness. "
                          "Work in the workspace, verify with tools, and say when you "
                          "are not sure."}},
    {"id": "agent-instructions", "name": "agent-instructions"},
    {"id": "tool-bash", "name": "tool-bash", "config": {"jobs": True}},
    {"id": "tool-fs", "name": "tool-fs"},
    {"id": "tool-todo", "name": "tool-todo"},
    {"id": "tool-subagent", "name": "tool-subagent"},
    {"id": "skills", "name": "skills"},
    {"id": "tool-ask-user", "name": "tool-ask-user"},
    {"id": "plan-mode", "name": "plan-mode"},
    {"id": "compaction", "name": "compaction", "config": {"threshold_tokens": 24000}},
]

PRESETS = {
    "minimal": [
        {"id": "persona", "name": "persona",
         "config": {"prefix": "You are a helpful software engineer assistant."}},
        {"id": "tool-bash", "name": "tool-bash"},
        {"id": "tool-editor", "name": "tool-editor"},
    ],
    "standard": _STANDARD,
    "code": _STANDARD + [
        {"id": "tool-run-code", "name": "tool-run-code"},
        {"id": "code-mode", "name": "code-mode"},
    ],
    "creator": _STANDARD + [
        {"id": "creator-inspect", "name": "creator-inspect"},
    ],
}


def _index(rows, rid):
    for i, r in enumerate(rows):
        if r["id"] == rid:
            return i
    raise PresetError(f"no row with id {rid!r}")


def compose(base, *patches):
    """Apply patch lists to a copy of ``base`` (a preset name or rows)."""
    rows = copy.deepcopy(PRESETS[base] if isinstance(base, str) else base)
    for patch in patches:
        for op in patch:
            if "insert" in op:
                new = copy.deepcopy(op["insert"])
                for r in new:
                    if any(x["id"] == r["id"] for x in rows):
                        raise PresetError(f"row id {r['id']!r} already exists")
                at = _index(rows, op["after"]) + 1 if "after" in op else len(rows)
                rows[at:at] = new
            elif "replace" in op:
                rows[_index(rows, op["replace"])]["config"] = copy.deepcopy(op["config"])
            elif "remove" in op:
                del rows[_index(rows, op["remove"])]
            elif "disable" in op:
                rows[_index(rows, op["disable"])]["disabled"] = True
            elif "enable" in op:
                rows[_index(rows, op["enable"])].pop("disabled", None)
            else:
                raise PresetError(f"unknown patch op {sorted(op)}")
    for r in rows:
        if r["name"] not in PLUGINS:
            raise PresetError(f"row {r['id']!r}: unknown plugin {r['name']!r}")
    return rows


def load_patch(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if str(path).endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError:
            raise PresetError("YAML patches need PyYAML; use JSON") from None
        return yaml.safe_load(text)
    return json.loads(text)


def apply_preset(agent, rows):
    """Mount ``rows`` on ``agent.ctx``; returns the forks."""
    forks = []
    for r in rows:
        if r.get("disabled"):
            continue
        cfg = copy.deepcopy(r.get("config"))
        if r["name"] == "creator-inspect":
            cfg = dict(cfg or {}, rows=copy.deepcopy(rows))
        fork = agent.ctx.plugin(PLUGINS[r["name"]], cfg)
        if fork.state != "active":
            raise PresetError(f"row {r['id']!r} did not activate "
                              f"(state {fork.state}; needs {list(fork.inject)})")
        forks.append(fork)
    return forks


def create_agent(ctx, session_id, config, preset="standard", patches=(),
                 setup=None, **kw):
    """``ctx.agents.create`` with a preset mounted in the setup window.

    ``setup(agent)``, if given, runs after the preset is mounted."""
    rows = compose(preset, *patches)
    forks = []

    def mount(agent):
        forks.extend(apply_preset(agent, rows))
        if setup is not None:
            setup(agent)

    agent = ctx.agents.create(session_id, config, setup=mount, **kw)
    agent.preset = rows
    agent.preset_forks = forks
    return agent
