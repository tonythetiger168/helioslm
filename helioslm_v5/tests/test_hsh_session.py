"""hsh H2: the append-only session log (helioslm_v5/hsh/session.py).

Pins:
1. append: contiguous seq, unknown types refused, data must be lossless
   JSON and is detached from the caller's objects;
2. derive_messages: the surface events in order, the system head replaced
   or cleared, attempts and log-only events never in history;
3. keep_thinking (OS2): "none" / "turn" / "all" keep exactly the
   reasoning they promise;
4. model-visible means logged: requests rebuilt from the log equal the
   requests a hand-driven loop sent, byte for byte, including a retry;
5. JSONL round trip, version check, repair of an interrupted tail, fork at
   a turn boundary with the inherited marker;
6. usage_summary (OS1) per turn, retried attempts counted apart;
7. ctx.sessions broadcasts session/event and unwinds with its plugin.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (Context, Session, SessionError, Sessions,  # noqa: E402
                             build_request, derive_messages, rebuild_requests,
                             register_event_type, request_hash, usage_summary)

HEADER = {"config": {"model": "helios-mid", "temperature": 0.0},
          "tools": [{"name": "calc", "parameters": {"type": "object"}}]}


def drive(session, script, keep_thinking="turn"):
    """A minimal hand-written loop: logs what it admits, then sends
    build_request(log). Returns the requests it sent, in order."""
    sent = []
    session.append("request/header", {"header": HEADER, "reason": "initial"})
    for turn, (user, steps) in enumerate(script, start=1):
        session.append("turn/start", {"turn": turn})
        for step, reply in enumerate(steps, start=1):
            session.append("step/start", {"turn": turn, "step": step})
            if step == 1:
                if turn == 1:
                    session.append("system/message", {
                        "turn": turn, "step": step,
                        "message": {"role": "system", "content": "be brief"}})
                session.append("user/message", {"content": user, "source": "human"})
            req = build_request(session.events, keep_thinking)
            sent.append(req)
            if reply.get("retry"):
                session.append("assistant/attempt", {
                    "turn": turn, "step": step, "stream": [],
                    "usage": {"input_tokens": 5, "output_tokens": 1}})
                sent.append(build_request(session.events, keep_thinking))
            msg = {"role": "assistant", "content": reply.get("text", "")}
            if reply.get("reasoning"):
                msg["reasoning"] = reply["reasoning"]
            if reply.get("call"):
                msg["tool_calls"] = [{"id": f"c{turn}{step}", "name": "calc",
                                      "arguments": reply["call"]}]
            session.append("assistant/message", {
                "turn": turn, "step": step, "message": msg, "stream": [],
                "usage": reply.get("usage", {})})
            if reply.get("call"):
                cid = f"c{turn}{step}"
                session.append("tool/call", {"turn": turn, "step": step,
                                             "call_id": cid, "name": "calc",
                                             "arguments": reply["call"]})
                session.append("tool/result", {"turn": turn, "step": step,
                                               "message": {"role": "tool", "call_id": cid,
                                                           "content": "4"}})
            session.append("step/end", {"turn": turn, "step": step})
        session.append("turn/end", {"turn": turn, "reason": "completed"})
    return sent


SCRIPT = [
    ("2+2?", [{"reasoning": "use calc", "call": '{"x":"2+2"}',
               "usage": {"input_tokens": 20, "output_tokens": 8, "reasoning_tokens": 3}},
              {"reasoning": "got 4", "text": "4", "retry": True,
               "usage": {"input_tokens": 30, "output_tokens": 2}}]),
    ("thanks", [{"reasoning": "polite", "text": "you're welcome",
                 "usage": {"input_tokens": 40, "output_tokens": 4}}]),
]


def test_append_contiguous_and_validated():
    s = Session()
    e0 = s.append("turn/start", {"turn": 1})
    assert e0["seq"] == 0 and s.append("turn/end", {"turn": 1, "reason": "x"})["seq"] == 1
    with pytest.raises(SessionError):
        s.append("made/up", {})
    with pytest.raises(SessionError):
        s.append("user/message", {"content": math.nan})
    with pytest.raises(SessionError):
        s.append("user/message", {"content": {1, 2}})
    data = {"content": "hi"}
    s.append("user/message", data)
    data["content"] = "changed"
    assert s.last("user/message")["data"]["content"] == "hi"
    with pytest.raises(SessionError):
        Session([{"seq": 3, "type": "turn/start", "data": {"turn": 1}}])


def test_register_event_type_log_only():
    register_event_type("todo/write")
    Session().append("todo/write", {"items": []})
    with pytest.raises(SessionError):
        register_event_type("note/message", surface=True)
    with pytest.raises(SessionError):
        register_event_type("user/message")


def test_derive_history_shape():
    s = Session()
    drive(s, SCRIPT)
    msgs = derive_messages(s.events, keep_thinking="none")
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "be brief"
    assert msgs[3] == {"role": "tool", "call_id": "c11", "content": "4"}
    # the retried attempt left no message behind
    assert sum(m["role"] == "assistant" for m in msgs) == 3


def test_system_head_replaced_and_cleared():
    s = Session()
    sm = lambda c: {"turn": 1, "step": 1, "message": {"role": "system", "content": c}}
    s.append("system/message", sm("v1"))
    s.append("user/message", {"content": "q"})
    s.append("system/message", sm("v2"))
    assert derive_messages(s.events)[0] == {"role": "system", "content": "v2"}
    s.append("system/message", sm(""))
    assert [m["role"] for m in derive_messages(s.events)] == ["user"]


def test_keep_thinking_modes():
    s = Session()
    drive(s, SCRIPT)
    ev = s.events

    def reasoning(mode):
        return [m.get("reasoning") for m in derive_messages(ev, mode)
                if m["role"] == "assistant"]

    assert reasoning("none") == [None, None, None]
    assert reasoning("turn") == [None, None, "polite"]       # current turn only
    assert reasoning("all") == ["use calc", "got 4", "polite"]
    # inside turn 1, before turn 2 starts, "turn" keeps turn 1's reasoning
    cut = next(e["seq"] for e in ev if e["type"] == "turn/start" and e["data"]["turn"] == 2)
    assert [m.get("reasoning") for m in derive_messages(ev[:cut], "turn")
            if m["role"] == "assistant"] == ["use calc", "got 4"]
    with pytest.raises(SessionError):
        derive_messages(ev, "some")


@pytest.mark.parametrize("mode", ["none", "turn", "all"])
def test_model_visible_means_logged(mode):
    s = Session()
    sent = drive(s, SCRIPT, keep_thinking=mode)
    rebuilt = [req for _, req in rebuild_requests(s.events, keep_thinking=mode)]
    assert len(sent) == len(rebuilt) == 4  # 3 steps + 1 retry
    assert [request_hash(r) for r in sent] == [request_hash(r) for r in rebuilt]
    # within turn 1 the second step saw the first step's reasoning ("turn")
    if mode == "turn":
        assert sent[1]["messages"][2]["reasoning"] == "use calc"


def test_unlogged_input_is_detected():
    # A loop that slips a message into the request without logging it is
    # caught: the replayed request differs.
    s = Session()
    sent = drive(s, SCRIPT)
    sent[0]["messages"].append({"role": "user", "content": "secret context"})
    rebuilt = [req for _, req in rebuild_requests(s.events)]
    assert request_hash(sent[0]) != request_hash(rebuilt[0])


def test_request_needs_header():
    s = Session()
    s.append("user/message", {"content": "hi"})
    with pytest.raises(SessionError):
        build_request(s.events)


def test_jsonl_roundtrip_and_version(tmp_path):
    s = Session(session_id="abc", clock=lambda: 123)
    drive(s, SCRIPT)
    p = tmp_path / "s.jsonl"
    s.save(p)
    back = Session.load(p)
    assert back.id == "abc" and back.events == s.events
    lines = p.read_text().splitlines()
    head = json.loads(lines[0])
    head["version"] = 99
    with pytest.raises(SessionError):
        Session.from_jsonl("\n".join([json.dumps(head)] + lines[1:]))
    with pytest.raises(SessionError):
        Session.from_jsonl('{"format": "other"}\n')


def test_repair_interrupted_tail():
    s = Session()
    s.append("turn/start", {"turn": 1})
    s.append("step/start", {"turn": 1, "step": 1})
    s.append("tool/call", {"turn": 1, "step": 1, "call_id": "c1", "name": "bash",
                           "arguments": "{}"})
    added = s.repair()
    assert [e["type"] for e in added] == ["tool/result", "step/end", "turn/end"]
    assert added[0]["data"]["message"]["is_error"] is True
    assert added[-1]["data"]["reason"] == "interrupted"
    assert s.open_turn() is None and s.repair() == []


def test_fork_at_turn_boundary():
    s = Session(session_id="p")
    drive(s, SCRIPT)
    child = s.fork(after_turn=1)
    ev = child.events
    assert ev[-1]["type"] == "session/end-seed" and ev[-1]["data"] == {"inherited": True}
    assert ev[:-1] == [e for e in s.events if e["seq"] < len(ev) - 1]
    assert not any(e["type"] == "turn/start" and e["data"]["turn"] == 2 for e in ev)
    with pytest.raises(SessionError):
        s.fork(after_turn=9)


def test_usage_summary():
    s = Session()
    drive(s, SCRIPT)
    u = usage_summary(s.events)
    assert u["turns"][1] == {"input_tokens": 55, "output_tokens": 11,
                             "reasoning_tokens": 3, "cached_input_tokens": 0,
                             "requests": 2, "attempts": 1}
    assert u["total"]["output_tokens"] == 15 and u["total"]["requests"] == 3


def test_sessions_service_broadcasts_and_unwinds():
    ctx = Context()
    fork = ctx.plugin(Sessions)
    seen = []
    ctx.on("session/event", lambda sid, ev: seen.append((sid, ev["type"])))
    s = ctx.sessions.create("s1")
    s.append("turn/start", {"turn": 1})
    assert seen == [("s1", "turn/start")]
    with pytest.raises(SessionError):
        ctx.sessions.create("s1")
    fork.dispose()
    s.append("turn/end", {"turn": 1, "reason": "x"})
    assert seen == [("s1", "turn/start")] and not ctx.has("sessions")
