"""hsh H2: the append-only session log.

After dsh's ``core/session`` (docs/architecture.md "Session log",
docs/subsystems/session.md). The log is the single source of truth for an
agent's history; the message list sent to the model is *derived* from it.

The rule this module enforces is dsh's **model-visible means logged**:
every request the loop sends must be a pure function of the log. The
loop (H4) logs ``request/header`` and the messages it admits, then builds
the request with :func:`build_request`; :func:`rebuild_requests` replays
the log and yields, for every model attempt, the request that attempt
must have seen. Tests compare the two byte for byte.

Event vocabulary (core types; plugins add more with
:func:`register_event_type`):

- surface events, which become messages: ``system/message``,
  ``user/message``, ``assistant/message``, ``tool/result``;
- log-only events: ``turn/start``, ``turn/end``, ``step/start``,
  ``step/end``, ``assistant/attempt`` (a settled failed or retried attempt,
  never history), ``tool/call``, ``request/header``, ``session/end-seed``,
  ``context/shadow`` (``{"first": seq, "last": seq}``: surface events in
  that range leave the model history; compaction uses it, H6).

Thinking (roadmap OS2). An assistant message may carry ``reasoning``.
``derive_messages(..., keep_thinking=...)`` decides which of it the model
sees again:

- ``"none"``: never;
- ``"turn"`` (default): only reasoning from the current turn, so it
  survives the tool calls of one turn but not the next user message (the
  DeepSeek thinking-mode convention);
- ``"all"``: every turn's reasoning (what MiniMax M3 needs; dropping it
  between tool turns degrades its multi-turn tool use).

Token accounting (roadmap OS1). ``assistant/message`` carries the
adapter's ``usage``; :func:`usage_summary` folds it per turn.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .kernel import Service

FORMAT = "hsh-session"
VERSION = 1

_EVENT_TYPES = {
    "turn/start": False,
    "turn/end": False,
    "step/start": False,
    "step/end": False,
    "system/message": True,
    "user/message": True,
    "assistant/message": True,
    "assistant/attempt": False,
    "tool/call": False,
    "tool/result": True,
    "request/header": False,
    "session/end-seed": False,
    "context/shadow": False,
}

KEEP_THINKING = ("none", "turn", "all")


class SessionError(ValueError):
    pass


def register_event_type(name, surface=False):
    """Add a plugin-owned event type. Only log-only types can be added:
    a new model-visible input would need its own projection rule here."""
    if surface:
        raise SessionError("plugins may add log-only event types only")
    if _EVENT_TYPES.get(name, False):
        raise SessionError(f"{name} is a core surface event")
    _EVENT_TYPES.setdefault(name, False)


def event_types():
    return dict(_EVENT_TYPES)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionError(f"event data is not lossless JSON: {exc}") from None


def request_hash(request):
    """sha256 of a request's canonical JSON: the identity tests compare."""
    return hashlib.sha256(_canonical(request).encode()).hexdigest()


class Session:
    """An append-only list of ``{"seq", "type", "data"}`` events."""

    def __init__(self, events=None, session_id="session", clock=None):
        self.id = session_id
        self._events = []
        self._clock = clock
        self._listeners = []
        for ev in events or ():
            self._append_event(copy.deepcopy(ev))

    # ----- writing ------------------------------------------------------
    def append(self, type_, data=None):
        ev = {"seq": len(self._events), "type": type_,
              "data": {} if data is None else data}
        if self._clock is not None:
            ev["time"] = self._clock()
        return self._append_event(ev)

    def _append_event(self, ev):
        if ev.get("type") not in _EVENT_TYPES:
            raise SessionError(f"unknown event type {ev.get('type')!r}")
        if ev.get("seq") != len(self._events):
            raise SessionError(
                f"seq {ev.get('seq')} breaks contiguity at {len(self._events)}")
        _canonical(ev)  # rejects NaN, sets, objects ...
        frozen = json.loads(_canonical(ev))  # detach from caller's objects
        self._events.append(frozen)
        for fn in list(self._listeners):
            fn(copy.deepcopy(frozen))
        return copy.deepcopy(frozen)

    def subscribe(self, fn):
        self._listeners.append(fn)

        def unsubscribe():
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsubscribe

    # ----- reading ------------------------------------------------------
    @property
    def events(self):
        return copy.deepcopy(self._events)

    def __len__(self):
        return len(self._events)

    def last(self, type_=None):
        for ev in reversed(self._events):
            if type_ is None or ev["type"] == type_:
                return copy.deepcopy(ev)
        return None

    # ----- persistence --------------------------------------------------
    def to_jsonl(self):
        head = {"format": FORMAT, "version": VERSION, "id": self.id}
        return "\n".join([_canonical(head)] + [_canonical(e) for e in self._events]) + "\n"

    @classmethod
    def from_jsonl(cls, text, clock=None):
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise SessionError("empty session file")
        head = json.loads(lines[0])
        if head.get("format") != FORMAT:
            raise SessionError("not an hsh session file")
        if head.get("version") != VERSION:
            raise SessionError(f"unsupported session version {head.get('version')}")
        return cls([json.loads(ln) for ln in lines[1:]],
                   session_id=head.get("id", "session"), clock=clock)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_jsonl())

    @classmethod
    def load(cls, path, clock=None):
        with open(path, encoding="utf-8") as fh:
            return cls.from_jsonl(fh.read(), clock=clock)

    # ----- lifecycle ----------------------------------------------------
    def open_turn(self):
        """The number of a turn with ``turn/start`` and no ``turn/end``."""
        open_ = None
        for ev in self._events:
            if ev["type"] == "turn/start":
                open_ = ev["data"]["turn"]
            elif ev["type"] == "turn/end":
                open_ = None
        return open_

    def repair(self):
        """Seal an interrupted tail (process died mid-turn) so it replays.

        Every ``tool/call`` without a ``tool/result`` gets an error result,
        an open step gets ``step/end``, and the open turn gets
        ``turn/end`` with reason ``interrupted``. Returns the events added.
        """
        added = []
        turn = self.open_turn()
        if turn is None:
            return added
        answered = {ev["data"]["message"]["call_id"] for ev in self._events
                    if ev["type"] == "tool/result"}
        step = None
        for ev in self._events:
            if ev["type"] == "step/start":
                step = (ev["data"]["turn"], ev["data"]["step"])
            elif ev["type"] == "step/end":
                step = None
        for ev in list(self._events):
            d = ev["data"]
            if ev["type"] == "tool/call" and d["call_id"] not in answered:
                added.append(self.append("tool/result", {
                    "turn": d["turn"], "step": d["step"],
                    "message": {"role": "tool", "call_id": d["call_id"],
                                "content": "interrupted before a result",
                                "is_error": True},
                    "error": {"name": d["name"], "code": "interrupted"}}))
        if step is not None:
            added.append(self.append("step/end", {"turn": step[0], "step": step[1]}))
        added.append(self.append("turn/end", {"turn": turn, "reason": "interrupted"}))
        return added

    def fork(self, after_turn):
        """A new Session seeded with this one up to the end of ``after_turn``.

        The child gets the inherited prefix plus one
        ``session/end-seed {"inherited": true}`` marker at the cut, so later
        readers can tell seed history from the child's own work.
        """
        cut = None
        for ev in self._events:
            if ev["type"] == "turn/end" and ev["data"]["turn"] == after_turn:
                cut = ev["seq"] + 1
        if cut is None:
            raise SessionError(f"turn {after_turn} has not ended; fork at a turn boundary")
        child = Session(self._events[:cut], session_id=f"{self.id}/fork{after_turn}",
                        clock=self._clock)
        child.append("session/end-seed", {"inherited": True})
        return child


# ----- projections ------------------------------------------------------

def _assistant_view(msg, keep_reasoning):
    out = {"role": "assistant", "content": msg.get("content", "")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if keep_reasoning and msg.get("reasoning"):
        out["reasoning"] = msg["reasoning"]
    return out


def derive_messages(events, keep_thinking="turn", projections=()):
    """The model-visible history projected from ``events``.

    ``projections`` are pure ``fn(message, event) -> message`` callables
    applied in order to every surface message (dsh "plugin-owned message
    projections"); they must not depend on anything outside their inputs.
    """
    if keep_thinking not in KEEP_THINKING:
        raise SessionError(f"keep_thinking must be one of {KEEP_THINKING}")
    events = list(events)
    current_turn = None
    for ev in events:
        if ev["type"] == "turn/start":
            current_turn = ev["data"]["turn"]

    shadowed = [(ev["data"]["first"], ev["data"]["last"]) for ev in events
                 if ev["type"] == "context/shadow"]
    system = None
    messages = []
    for ev in events:
        t, d = ev["type"], ev["data"]
        if t != "system/message" and any(a <= ev["seq"] <= b for a, b in shadowed):
            continue
        if t == "system/message":
            # One system head: a new rendering replaces it, an empty one
            # clears it (no older instructions stay model-visible).
            content = d["message"]["content"]
            system = {"role": "system", "content": content} if content else None
            continue
        if t == "user/message":
            msg = {"role": "user", "content": d["content"]}
        elif t == "assistant/message":
            keep = (keep_thinking == "all" or
                    (keep_thinking == "turn" and d.get("turn") == current_turn))
            msg = _assistant_view(d["message"], keep)
        elif t == "tool/result":
            m = d["message"]
            msg = {"role": "tool", "call_id": m["call_id"], "content": m["content"]}
            if m.get("is_error"):
                msg["is_error"] = True
        else:
            continue
        for fn in projections:
            msg = fn(copy.deepcopy(msg), ev)
        messages.append(msg)
    return ([system] if system else []) + messages


def fold_request_header(events):
    """The latest ``request/header`` snapshot's header, or None."""
    header = None
    for ev in events:
        if ev["type"] == "request/header":
            header = ev["data"]["header"]
    return copy.deepcopy(header)


def build_request(events, keep_thinking="turn", projections=()):
    """The request a model call at the end of ``events`` sends."""
    header = fold_request_header(events)
    if header is None:
        raise SessionError("no request/header logged before the request")
    req = {"header": header,
           "messages": derive_messages(events, keep_thinking, projections)}
    return json.loads(_canonical(req))


def rebuild_requests(events, keep_thinking="turn", projections=()):
    """Replay: ``[(seq, request)]`` for every assistant attempt in the log.

    Each attempt saw exactly the events before it, so its request is
    :func:`build_request` of that prefix. A loop that logs everything it
    sends passes ``request_hash(sent) == request_hash(rebuilt)``.
    """
    events = list(events)
    out = []
    for i, ev in enumerate(events):
        if ev["type"] in ("assistant/message", "assistant/attempt"):
            out.append((ev["seq"], build_request(events[:i], keep_thinking, projections)))
    return out


def usage_summary(events):
    """Token accounting per turn and in total, from ``assistant/message`` usage.

    Returns ``{"turns": {turn: {...}}, "total": {...}}`` where each record
    has ``input_tokens``, ``output_tokens``, ``reasoning_tokens``,
    ``requests`` and ``attempts`` (retried or failed attempts, which spend
    tokens without adding history).
    """
    keys = ("input_tokens", "output_tokens", "reasoning_tokens")

    def blank():
        return {k: 0 for k in keys} | {"requests": 0, "attempts": 0}

    turns, total = {}, blank()
    for ev in events:
        t, d = ev["type"], ev["data"]
        if t not in ("assistant/message", "assistant/attempt"):
            continue
        rec = turns.setdefault(d["turn"], blank())
        field = "requests" if t == "assistant/message" else "attempts"
        rec[field] += 1
        total[field] += 1
        for k in keys:
            v = int((d.get("usage") or {}).get(k, 0))
            rec[k] += v
            total[k] += v
    return {"turns": turns, "total": total}


class Sessions(Service):
    """``ctx.sessions``: creates and opens Sessions and broadcasts every
    appended event as ``session/event`` (mode ``emit``)."""

    key = "sessions"

    def __init__(self, ctx, config=None):
        super().__init__(ctx, config)
        self.clock = (config or {}).get("clock")
        self._open = {}

    def start(self):
        self.ctx.declare("session/event", "emit")

    def stop(self):
        for unsubscribe in self._open.values():
            unsubscribe()
        self._open.clear()

    def _watch(self, session):
        self._open[session.id] = session.subscribe(
            lambda ev: self.ctx.emit("session/event", session.id, ev))
        return session

    def create(self, session_id, seed=None):
        if session_id in self._open:
            raise SessionError(f"session {session_id!r} is already open")
        return self._watch(Session(seed, session_id=session_id, clock=self.clock))

    def load(self, path):
        return self._watch(Session.load(path, clock=self.clock))
