"""hsh H6: context compaction (``compaction`` plugin).

After dsh's compaction seam (docs/subsystems/compaction.md). Compaction
replaces old history with a summary *in the log*, so the rule "every
request rebuilds from the log" still holds:

  compaction/start   {turn}                         log-only lock
  compaction/summary {summary, first, last, tokens, usage}
  context/shadow     {first, last}                  hides that range
  user/message       {content: summary, source: "compaction"}
  compaction/end     {turn, error?}                 releases the lock

Inside a turn the kept part is the current turn: the shadowed range ends
just before the latest ``turn/start``, so a tool call and its result are
never split. Between turns (``compact`` by hand) everything is folded.
A ``compaction/start`` without its ``compaction/end`` (the process died
mid-way) is detectable from the log.

Automatic mode: mounted on an agent, the plugin listens to
``agent/request`` and compacts when the estimated size of the next
request passes ``threshold_tokens``. ``compact(agent)`` runs it by hand.
The summary comes from ``summarize(messages) -> (text, usage)``; the
default asks the agent's own model through ``ctx.llm``.
"""
from __future__ import annotations

import json

from .session import derive_messages, register_event_type

for _t in ("compaction/start", "compaction/summary", "compaction/end"):
    register_event_type(_t)

SUMMARY_PROMPT = (
    "Summarize the conversation so far for your own later use: the task, "
    "decisions made, files and commands involved, results, and what is "
    "still open. Be specific and brief.")


def estimate_tokens(messages):
    """~4 characters per token over the JSON of the messages."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4


def _cut(events, in_turn):
    """(first, last) seq range to shadow, or None if nothing new to compact.

    Inside a turn: everything model-visible before the current turn.
    Between turns (manual compaction): everything. Minus what an earlier
    compaction already shadowed; its summary message is included, so
    summaries nest. The summary is appended at the end of the log, so the
    kept part must not have surface events yet: automatic compaction only
    runs before the turn's first request (see ``compaction``)."""
    starts = [e["seq"] for e in events if e["type"] == "turn/start"]
    boundary = starts[-1] if in_turn and starts else len(events)
    done = [(e["data"]["first"], e["data"]["last"]) for e in events
            if e["type"] == "context/shadow"]
    shadowable = {"user/message", "assistant/message", "tool/result"}
    seqs = [e["seq"] for e in events
            if e["type"] in shadowable and e["seq"] < boundary
            and not any(a <= e["seq"] <= b for a, b in done)]
    return (seqs[0], seqs[-1]) if seqs else None


def _lock_open(events):
    marks = [e["type"] for e in events
             if e["type"] in ("compaction/start", "compaction/end")]
    return bool(marks) and marks[-1] == "compaction/start"


def compact(agent, summarize=None):
    """Compact ``agent``'s history now. Returns the summary, or None when
    there is nothing before the current turn to compact (or it failed;
    the failure is logged on ``compaction/end``)."""
    s = agent.session
    events = s.events
    if _lock_open(events):
        raise RuntimeError("an earlier compaction never finished (orphaned lock)")
    rng = _cut(events, s.open_turn() is not None)
    if rng is None:
        return None
    first, last = rng
    old = [m for m in derive_messages([e for e in events if e["seq"] <= last], "none")
           if m["role"] != "system"]
    s.append("compaction/start", {"turn": agent.turn})
    try:
        text, usage = (summarize or _model_summary)(agent, old)
    except Exception as exc:
        s.append("compaction/end", {"turn": agent.turn,
                                    "error": f"{type(exc).__name__}: {exc}"})
        return None
    s.append("compaction/summary", {"summary": text, "first": first, "last": last,
                                    "tokens": estimate_tokens(old), "usage": usage or {}})
    s.append("context/shadow", {"first": first, "last": last})
    s.append("user/message", {"content": "[Summary of earlier conversation]\n" + text,
                              "source": "compaction"})
    s.append("compaction/end", {"turn": agent.turn})
    return text


def _model_summary(agent, messages):
    req = {"header": {"config": dict(agent.config)},
           "messages": messages + [{"role": "user", "content": SUMMARY_PROMPT}]}
    out = agent._agents.ctx.llm.call(req, scope=agent.key)
    return out["content"], out["usage"]


def compaction(ctx, config=None):
    """Automatic compaction for the agent this plugin is mounted on."""
    cfg = config or {}
    threshold = cfg.get("threshold_tokens", 24_000)
    summarize = cfg.get("summarize")

    def on_request(agent, config, nxt):
        events = agent.session.events
        start = max(e["seq"] for e in events if e["type"] == "turn/start")
        fresh_turn = not any(e["seq"] > start and e["type"] in
                             ("user/message", "assistant/message", "tool/result")
                             for e in events)
        # Only before the turn's first request: later, the summary would land
        # after this turn's own messages. A long tool loop inside one turn is
        # compacted at the start of the next turn.
        if fresh_turn and estimate_tokens(derive_messages(events, agent.keep_thinking)) > threshold:
            compact(agent, summarize)
        return nxt(agent, config)

    ctx.on("agent/request", on_request)


compaction.inject = ["agents", "llm"]
