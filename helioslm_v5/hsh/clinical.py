"""hsh clinical policy: approval on every call, allow-listed tools, audit
log (v5.223).

The ``clinical`` preset (presets.py) is for agents that act in a clinical
workflow, e.g. scheduling follow-ups from a rules engine's result. It
mounts no shell, file, search or sub-agent tool; the host registers its
own tools in ``create_agent(..., setup=...)`` and lists them in the
policy's ``tools`` config. This plugin then enforces, for the agent's
scope:

* only allow-listed tools run: global tools outside the list are hidden
  (``restrict``) and a guard denies any other name that shows up later;
* every call is "Ask first": a ``tools/pre-execute`` listener asks, and a
  guard denies any call that was not approved through ``ctx.approval``
  (``Execution.approval``), so a host listener answering ``allow`` (an
  "auto" permission mode) cannot skip the clinician;
* every call is written to the session log as an ``audit/tool-call``
  event: call id, tool, arguments, question, approved, approver
  (``by``, when the approval callback returns ``{"approved": True,
  "by": ...}``), outcome and the event time when the session has a clock.

The preset is locked: patches are refused at creation and during the
session (``presets.patch_agent``), and a refused live patch is logged as a
``policy/refused`` event. :func:`audit_export` rebuilds the audit table
from a session log alone and checks it against the ``tool/call`` events.
Both event types are log-only, so the requests rebuilt by
``session.rebuild_requests`` are unchanged.
"""
from __future__ import annotations

import json

from .session import register_event_type

register_event_type("audit/tool-call")
register_event_type("policy/refused")


class AuditError(ValueError):
    pass


def clinical_policy(ctx, config=None):
    cfg = config or {}
    allowed = set(cfg.get("tools", ()))
    ctx.tools.restrict(ctx, allow=allowed)

    def ask(exe, nxt):
        return {"action": "ask",
                "question": f"Clinician approval needed: run {exe.name} with {exe.arguments}?"}

    ctx.on("tools/pre-execute", ask)

    def guard(exe):
        if exe.name not in allowed:
            return f"clinical preset: tool {exe.name!r} is not allow-listed"
        if not (exe.approval and exe.approval.get("approved")):
            return "clinical preset: every tool call needs clinician approval"
        return None

    ctx.tools.guard(guard, ctx=ctx)

    def audit(exe, result):
        agent = ctx.agents.by_key(exe.scope)
        if agent is None:
            return
        ap = exe.approval or {}
        agent.session.append("audit/tool-call", {
            "call_id": exe.call_id, "tool": exe.name, "arguments": exe.arguments,
            "question": ap.get("question"), "approved": bool(ap.get("approved")),
            "by": ap.get("by"), "ran": exe.body_ran,
            "outcome": (result.error or {}).get("code", "error") if result.is_error else "ok"})

    ctx.on("tools/result", audit)


clinical_policy.inject = ["tools", "agents"]


def audit_export(events):
    """Audit rows ``{call_id, tool, arguments, approved, by, question,
    outcome, ran, time}`` rebuilt from a session log. Raises AuditError
    when a ``tool/call`` has no audit row, or a row's tool / arguments
    differ from the logged call."""
    calls, rows = {}, []
    for ev in events:
        d = ev["data"]
        if ev["type"] == "tool/call":
            calls[d["call_id"]] = d
        elif ev["type"] == "audit/tool-call":
            rows.append(dict(d, time=ev.get("time"), seq=ev["seq"]))
    seen = set()
    for r in rows:
        c = calls.get(r["call_id"])
        if c is None:
            raise AuditError(f"audit row {r['call_id']} has no tool/call event")
        args = c["arguments"]
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        if c["name"] != r["tool"] or args != r["arguments"]:
            raise AuditError(f"audit row {r['call_id']} differs from its tool/call")
        seen.add(r["call_id"])
    missing = sorted(set(calls) - seen)
    if missing:
        raise AuditError(f"tool calls without an audit row: {missing}")
    return rows
