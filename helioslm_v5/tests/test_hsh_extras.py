"""hsh H6: compaction, subagents, AGENTS.md and the Claude Code hooks bridge.

Pins:
1. compaction shadows everything before the current turn behind one
   summary message, through log events only, so every request (before and
   after compaction) still rebuilds from the log; summaries nest; a failed
   summarizer is logged and leaves history intact; an orphaned lock is
   refused; automatic mode triggers on the size threshold;
2. subagent: a fresh child session with its own preset answers the task,
   the parent gets the answer as the tool result, depth is limited, and
   the child does not see the parent's scoped tools;
3. AGENTS.md lands in the system prompt (capped);
4. hooks bridge: PreToolUse exit 2 and permissionDecision deny/allow,
   PostToolUse block and additionalContext, UserPromptSubmit block and
   added context, Stop continuing the turn, matcher regex.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from helioslm_v5.hsh import (LLM, Agents, Context, ScriptedAdapter,  # noqa: E402
                             Sessions, ToolDefinition, Tools, Workspace, compact,
                             create_agent, derive_messages, rebuild_requests,
                             request_hash)

CFG = {"provider": "fake", "model": "m"}


def world(tmp_path, adapter):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter)
    return ctx


def logged(agent):
    return agent.sent == [request_hash(r) for _, r in rebuild_requests(
        agent.session.events, agent.keep_thinking)]


def summarizer(agent, messages):
    return f"{len(messages)} messages about arithmetic", {"output_tokens": 5}


# ----- compaction --------------------------------------------------------

def test_manual_compaction_shadows_and_replays(tmp_path):
    ad = ScriptedAdapter(fn=lambda r, i: f"answer {i}")
    ctx = world(tmp_path, ad)
    a = create_agent(ctx, "s", CFG, "minimal", system_prompt="sys")
    a.send("q1")
    a.send("q2")
    # between turns, a manual compaction folds everything
    assert compact(a, summarizer) == "4 messages about arithmetic"
    msgs = derive_messages(a.session.events)
    contents = [m["content"] for m in msgs if m["role"] != "system"]
    assert contents == ["[Summary of earlier conversation]\n4 messages about arithmetic"]
    a.send("q3")
    req = ad.requests[-1]["messages"]
    assert req[0]["role"] == "system" and "q1" not in json.dumps(req)
    assert logged(a)
    types = [e["type"] for e in a.session.events if e["type"].startswith(("compaction", "context"))]
    assert types == ["compaction/start", "compaction/summary", "context/shadow", "compaction/end"]

    # nesting: a second compaction folds the first summary in
    a.send("q4")
    compact(a, lambda ag, m: (f"nested over {len(m)}", {}))
    users = [m for m in derive_messages(a.session.events) if m["role"] == "user"]
    assert len(users) == 1 and users[0]["content"].endswith("nested over 5")
    assert logged(a)


def test_compaction_failure_and_orphan(tmp_path):
    ctx = world(tmp_path, ScriptedAdapter(fn=lambda r, i: "x"))
    a = create_agent(ctx, "s", CFG, "minimal")
    a.send("one")
    a.send("two")
    before = derive_messages(a.session.events)

    def broken(agent, m):
        raise RuntimeError("model down")

    assert compact(a, broken) is None
    assert derive_messages(a.session.events) == before
    assert "model down" in a.session.last("compaction/end")["data"]["error"]
    a.session.append("compaction/start", {"turn": a.turn})
    with pytest.raises(RuntimeError):
        compact(a, summarizer)


def test_automatic_compaction_threshold(tmp_path):
    ad = ScriptedAdapter(fn=lambda r, i: "y" * 400)
    ctx = world(tmp_path, ad)
    a = create_agent(ctx, "s", CFG, "standard",
                     patches=[[{"replace": "compaction",
                                "config": {"threshold_tokens": 150, "summarize": summarizer}}]])
    for q in ("a", "b", "c"):
        a.send(q)
    assert any(e["type"] == "compaction/summary" for e in a.session.events)
    last = ad.requests[-1]["messages"]
    assert any(m["role"] == "user" and m["content"].startswith("[Summary") for m in last)
    assert logged(a)


# ----- subagent ----------------------------------------------------------

def test_subagent_delegates(tmp_path):
    def model(req, i):
        last = req["messages"][-1]
        if i == 0:
            return {"tool_calls": [{"id": "s1", "name": "subagent",
                                    "arguments": json.dumps({"task": "count files"})}]}
        if last["role"] == "user" and last["content"] == "count files":
            names = [t["name"] for t in req["header"]["tools"]]
            return f"child tools: {','.join(names)}"
        return "parent got: " + last["content"]

    ctx = world(tmp_path, ScriptedAdapter(fn=model))

    def secret(agent):
        ctx.tools.register(ToolDefinition("parent_only", "", {"type": "object"},
                                          lambda a, r: 1), ctx=agent.ctx)

    parent = create_agent(ctx, "p", CFG, "standard", setup=secret)
    out = parent.send("delegate")
    child = ctx.agents.get("p/sub1")
    assert child is not None and child.depth == 1 and child.parent_session == "p"
    assert out.startswith("parent got: child tools: ")
    assert "parent_only" not in out
    assert logged(parent) and logged(child)


def test_subagent_depth_limit(tmp_path):
    def model(req, i):
        if req["messages"][-1]["role"] == "tool":
            return "done: " + req["messages"][-1]["content"]
        return {"tool_calls": [{"id": f"s{i}", "name": "subagent",
                                "arguments": json.dumps({"task": "go deeper"})}]}

    ctx = world(tmp_path, ScriptedAdapter(fn=model))
    rows = [[{"insert": [{"id": "sub", "name": "tool-subagent",
                          "config": {"max_depth": 1, "preset": "standard"}}]},
             {"remove": "tool-subagent"}]]
    parent = create_agent(ctx, "p", CFG, "standard", patches=rows)
    parent.send("start")
    child = ctx.agents.get("p/sub1")
    errors = [e for e in child.session.events if e["type"] == "tool/result"
              and e["data"]["message"].get("is_error")]
    assert errors and "depth limit" in errors[0]["data"]["message"]["content"]


# ----- AGENTS.md ---------------------------------------------------------

def test_agents_md_in_system_prompt(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Run pytest before committing.\n" + "x" * 100)
    ad = ScriptedAdapter(["ok"])
    ctx = world(tmp_path, ad)
    a = create_agent(ctx, "s", CFG, "standard",
                     patches=[[{"replace": "agent-instructions", "config": {"max_bytes": 40}}]])
    a.send("hi")
    sys_msg = ad.requests[0]["messages"][0]["content"]
    assert "# Instructions from AGENTS.md" in sys_msg and "Run pytest" in sys_msg
    assert "x" * 20 not in sys_msg


# ----- hooks bridge ------------------------------------------------------

def hook(cmd, matcher=""):
    return [{"matcher": matcher, "hooks": [{"type": "command", "command": cmd}]}]


def test_pre_tool_hooks(tmp_path):
    ctx = world(tmp_path, ScriptedAdapter([]))
    spec = {"hooks": {"PreToolUse": hook("echo 'no rm' >&2; exit 2", "bash") + hook(
        "echo '{\"hookSpecificOutput\":{\"permissionDecision\":\"deny\","
        "\"permissionDecisionReason\":\"read-only session\"}}'", "write|edit")}}
    a = create_agent(ctx, "s", CFG, "standard",
                     patches=[[{"insert": [{"id": "h", "name": "claude-hooks",
                                            "config": spec}]}]])
    r = ctx.tools.execute("1", "bash", {"description": "d", "command": "ls"}, scope=a.key)
    assert r.is_error and "no rm" in r.content
    r = ctx.tools.execute("2", "write", {"path": "f", "content": "x"}, scope=a.key)
    assert r.is_error and "read-only session" in r.content
    assert not (tmp_path / "f").exists()
    r = ctx.tools.execute("3", "glob", {"pattern": "*"}, scope=a.key)  # no matcher hit
    assert not r.is_error


def test_hooks_file_post_prompt_stop(tmp_path):
    payloads = tmp_path / "payloads.jsonl"
    spec = {"hooks": {
        "PostToolUse": hook(
            f"cat >> {payloads}; echo; echo '{{\"decision\":\"block\",\"reason\":\"lint failed\","
            "\"hookSpecificOutput\":{\"additionalContext\":\"run ruff\"}}'", "write"),
        "UserPromptSubmit": hook(
            "python3 -c \"import json,sys; p=json.load(sys.stdin)['prompt']; "
            "sys.exit(2) if 'secret' in p else print('today is Thursday')\""),
        "Stop": hook(
            "python3 -c \"import json,sys; m=json.load(sys.stdin)['last_assistant_message']; "
            "print(json.dumps({'decision':'block','reason':'add tests'}) if m=='draft' else '')\""),
    }}
    hooks_file = tmp_path / "hooks.json"
    hooks_file.write_text(json.dumps(spec))
    replies = iter([{"tool_calls": [{"id": "w", "name": "write",
                                     "arguments": json.dumps({"path": "a.py", "content": "x"})}]},
                    "draft", "final"])
    ad = ScriptedAdapter(fn=lambda r, i: next(replies))
    ctx = world(tmp_path, ad)
    a = create_agent(ctx, "s", CFG, "standard",
                     patches=[[{"insert": [{"id": "h", "name": "claude-hooks",
                                            "config": {"path": str(hooks_file)}}]}]])
    assert a.send("tell me a secret") is None  # UserPromptSubmit exit 2
    assert a.session.last("turn/end")["data"]["reason"] == "rejected"

    assert a.send("write a.py") == "final"
    first = ad.requests[0]["messages"]
    assert [m["content"] for m in first[-2:]] == ["write a.py", "today is Thursday"]
    tool_msg = ad.requests[1]["messages"][-2]
    assert tool_msg["role"] == "tool" and tool_msg["is_error"] and "lint failed" in tool_msg["content"]
    assert ad.requests[1]["messages"][-1] == {"role": "user", "content": "run ruff"}
    assert ad.requests[2]["messages"][-1]["content"] == "add tests"  # Stop hook continued
    sent = json.loads(payloads.read_text().splitlines()[0])
    assert sent["hook_event_name"] == "PostToolUse" and sent["tool_input"]["path"] == "a.py"
    assert logged(a)
