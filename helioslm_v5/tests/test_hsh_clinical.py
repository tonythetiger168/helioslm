"""v5.223: hsh clinical preset (hsh/clinical.py, presets.py).

Pins:
1. the clinical preset mounts no bash / read / write / edit / glob / grep
   / subagent; calling bash is refused (unknown tool) and audited;
2. only allow-listed host tools are visible and run; a host tool that is
   registered but not listed is denied by the guard;
3. every call needs approval: without an approval service, with a refusal,
   and even when a host listener answers "allow" (an auto permission
   mode), the tool body does not run; with approval it runs and the
   approver ("by") is logged;
4. the preset is locked: patches at creation are refused; a live patch is
   refused and logged as policy/refused;
5. egress is deny with an empty allow list;
6. audit_export rebuilds the audit table from the log alone (also after a
   save / load round trip), matches the tool/call events, and the logged
   requests still equal rebuild_requests;
7. other presets are unchanged (allow_tools refused there, live insert
   patches still work for unlocked presets).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, AuditError, Context,  # noqa: E402
                             PresetError, ScriptedAdapter, Session, Sessions,
                             ToolDefinition, Tools, Workspace, audit_export,
                             compose, create_agent, is_locked, patch_agent,
                             rebuild_requests, request_hash)

CFG = {"provider": "fake", "model": "m"}
SCHED = {"type": "object", "properties": {"patient_ref": {"type": "string"},
                                          "months": {"type": "integer"}},
         "required": ["patient_ref", "months"], "additionalProperties": False}


def world(tmp_path, adapter=None, approval=None):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter or ScriptedAdapter(fn=lambda r, i: "ok"))
    if approval is not None:
        ctx.provide("approval", approval)
    return ctx


def clinical(ctx, sid="c", ran=None, extra_tool=False, **kw):
    ran = ran if ran is not None else []

    def setup(agent):
        agent.ctx.tools.register(ToolDefinition(
            "schedule_followup", "Schedule a follow-up LDCT.", SCHED,
            lambda a, e: ran.append(a) or {"scheduled": a["months"]}), ctx=agent.ctx)
        if extra_tool:
            agent.ctx.tools.register(ToolDefinition(
                "cancel_all", "Cancel every appointment.", {"type": "object"},
                lambda a, e: ran.append("cancel") or "done"), ctx=agent.ctx)

    return create_agent(ctx, sid, CFG, "clinical", setup=setup,
                        allow_tools=["schedule_followup"], **kw)


def call(ctx, agent, name, args, cid="t1"):
    return ctx.tools.execute(cid, name, args, scope=agent.key)


def names(ctx, agent):
    return [s["name"] for s in ctx.tools.schemas(agent.key)]


ARGS = {"patient_ref": "SYN-0001", "months": 3}


def test_no_builtin_tools_and_bash_refused(tmp_path):
    ctx = world(tmp_path, approval=lambda e, q: True)
    a = clinical(ctx)
    assert names(ctx, a) == ["schedule_followup"]
    for tool in ("bash", "read", "write", "edit", "glob", "grep", "subagent"):
        r = call(ctx, a, tool, {"command": "ls"})
        assert r.is_error and r.error["code"] == "unknown-tool"
    ev = [e["data"] for e in a.session.events if e["type"] == "audit/tool-call"]
    assert [(x["tool"], x["outcome"], x["ran"]) for x in ev][0] == ("bash", "unknown-tool", False)


def test_global_tools_hidden(tmp_path):
    ctx = world(tmp_path, approval=lambda e, q: True)
    ctx.tools.register(ToolDefinition("shell", "global shell", {"type": "object"},
                                      lambda a, e: "ran"))
    a = clinical(ctx)
    assert "shell" not in names(ctx, a)
    assert call(ctx, a, "shell", {}).error["code"] == "unknown-tool"


@pytest.mark.parametrize("approval,auto,expect", [
    (None, False, False),                                    # no approval service
    (lambda e, q: False, False, False),                      # refused
    (lambda e, q: "yes", False, False),                      # only True / dict approves
    (lambda e, q: False, True, False),                       # host says allow: still asks
    (None, True, False),                                     # auto mode, no clinician
    (lambda e, q: True, False, True),
    (lambda e, q: {"approved": True, "by": "dr-lin"}, True, True),
])
def test_every_call_needs_approval(tmp_path, approval, auto, expect):
    ran = []
    ctx = world(tmp_path, approval=approval)
    a = clinical(ctx, ran=ran)
    if auto:   # a host "auto" permission listener that allows without asking
        a.ctx.on("tools/pre-execute", lambda e, n: {"action": "allow"})
    r = call(ctx, a, "schedule_followup", ARGS)
    assert bool(ran) == expect and r.is_error == (not expect)
    if not expect:
        assert r.error["code"] == "denied"
    row = audit_export(_with_call(a, "schedule_followup", ARGS))[-1]
    assert row["approved"] == expect and row["ran"] == expect
    assert row["outcome"] == ("ok" if expect else "denied")


def _with_call(agent, name, args, cid="t1"):
    """The agent's events plus the tool/call the loop would have logged
    (these tests call ctx.tools.execute directly)."""
    ev = agent.session.events
    call_ev = {"seq": -1, "type": "tool/call",
               "data": {"turn": 0, "step": 0, "call_id": cid, "name": name, "arguments": args}}
    return [call_ev] + ev


def test_unlisted_host_tool_denied(tmp_path):
    ran = []
    ctx = world(tmp_path, approval=lambda e, q: True)
    a = clinical(ctx, ran=ran, extra_tool=True)
    r = call(ctx, a, "cancel_all", {})
    assert r.is_error and "not allow-listed" in r.content and not ran


def test_locked(tmp_path):
    assert is_locked("clinical") and not is_locked("standard")
    with pytest.raises(PresetError, match="locked"):
        compose("clinical", [{"insert": [{"id": "b", "name": "tool-bash"}]}])
    with pytest.raises(PresetError, match="locked"):
        compose(compose("clinical"), [{"remove": "clinical-policy"}])
    ctx = world(tmp_path, approval=lambda e, q: True)
    with pytest.raises(PresetError, match="locked"):
        clinical(ctx, "x", patches=[[{"disable": "clinical-policy"}]])
    a = clinical(ctx)
    with pytest.raises(PresetError, match="locked"):
        patch_agent(a, [{"insert": [{"id": "b", "name": "tool-bash"}]}])
    refused = [e for e in a.session.events if e["type"] == "policy/refused"]
    assert len(refused) == 1 and refused[0]["data"]["patches"][0][0]["insert"][0]["name"] == "tool-bash"
    assert "bash" not in names(ctx, a)


def test_egress_deny():
    rows = compose("clinical")
    eg = [r for r in rows if r["name"] == "egress"]
    assert eg == [{"id": "egress", "name": "egress", "config": {"mode": "deny", "allow": []}}]
    banned = {"tool-bash", "tool-fs", "tool-editor", "tool-subagent", "tool-run-code",
              "code-mode", "skills", "mcp", "tool-web-fetch", "creator-inspect"}
    assert not banned & {r["name"] for r in rows}


def test_end_to_end_audit_and_replay(tmp_path):
    replies = [
        {"tool_calls": [{"id": "1", "name": "schedule_followup",
                         "arguments": json.dumps(ARGS)}]},
        {"tool_calls": [{"id": "2", "name": "bash",
                         "arguments": json.dumps({"description": "d", "command": "ls"})}]},
        {"tool_calls": [{"id": "3", "name": "schedule_followup",
                         "arguments": json.dumps(dict(ARGS, months=12))}]},
        "Follow-up scheduled in 3 months; the 12-month request was not approved.",
    ]
    answers = iter([{"approved": True, "by": "rn-chen"}, False])
    ad = ScriptedAdapter(replies)
    ran = []
    ctx = world(tmp_path, ad, approval=lambda e, q: next(answers))
    a = clinical(ctx, "e2e", ran=ran)
    a.send("schedule the follow-up for SYN-0001")
    assert ran == [ARGS]
    rows = audit_export(a.session.events)
    assert [(r["call_id"], r["tool"], r["approved"], r["by"], r["outcome"]) for r in rows] == [
        ("1", "schedule_followup", True, "rn-chen", "ok"),
        ("2", "bash", False, None, "unknown-tool"),
        ("3", "schedule_followup", False, None, "denied")]
    assert rows[0]["arguments"] == ARGS and "SYN-0001" in rows[0]["question"]
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]
    # the audit table rebuilds from a saved log alone
    path = tmp_path / "s.jsonl"
    a.session.save(path)
    again = Session.load(path)
    assert audit_export(again.events) == rows
    # audit events are log-only: requests rebuild the same without them
    plain = [e for e in a.session.events if e["type"] != "audit/tool-call"]
    assert [request_hash(r) for _, r in rebuild_requests(plain)] == a.sent
    # tampering is caught
    bad = json.loads(json.dumps(a.session.events))
    next(e for e in bad if e["type"] == "audit/tool-call")["data"]["arguments"]["months"] = 6
    with pytest.raises(AuditError, match="differs"):
        audit_export(bad)
    with pytest.raises(AuditError, match="without an audit row"):
        audit_export([e for e in a.session.events if e["type"] != "audit/tool-call"])


def test_other_presets_unchanged(tmp_path):
    ctx = world(tmp_path)
    with pytest.raises(PresetError, match="allow_tools"):
        create_agent(ctx, "s", CFG, "minimal", allow_tools=["x"])
    a = create_agent(ctx, "m", CFG, "minimal")
    assert names(ctx, a) == ["bash", "str_replace_editor"]
    patch_agent(a, [{"insert": [{"id": "todo", "name": "tool-todo"}]}])
    assert "todo_write" in names(ctx, a)
    with pytest.raises(PresetError, match="only insert"):
        patch_agent(a, [{"remove": "todo"}])
    r = call(ctx, a, "todo_write", {"todos": [{"content": "x", "status": "pending"}]})
    assert not r.is_error and r.content == "[ ] x"
