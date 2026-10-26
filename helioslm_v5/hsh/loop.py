"""hsh H4: agents and the default turn/step driver (``ctx.agents``).

After dsh's ``core/agent`` + ``core/agent-loop`` (docs/architecture.md
"Turn flow", docs/agent-lifecycle.md). A **step** is one model request
plus the tools it calls; a **turn** is zero or more steps that drains the
agent's inbox.

::

    turn/start
      claim injected context + one queued message
      agent/pre-step   waterfall -> {"action": "enter", "messages": [...]}
                                    | {"action": "reject"}
      step/start
        agent/request  waterfall over the call config (route, sampling)
        system-prompt/assemble waterfall -> system/message if it changed
        user/message per admitted message
        request/header if the header (config + tool schemas) changed
        request = session.build_request(log)       <- the only source
        llm.call -> assistant/message   (agent/assistant-stream: live deltas)
                 | assistant/attempt + agent/request-error waterfall
                   ({"action": "retry"} retries in the same step)
        tool/call -> ctx.tools.execute(scope=agent) -> tool/result
        deferred tool contexts -> user/message (source "tool-context")
      step/end; more steps while the model keeps calling tools
      agent/turn-stopping serial (a listener may return a follow-up
                                  message, which starts another step)
    turn/end {reason}

Every model-visible input is logged before the request is built from the
log, so ``session.rebuild_requests`` reproduces each request exactly; the
agent keeps the hash of every request it sent (``agent.sent``) so tests
can check that.
"""
from __future__ import annotations

from .kernel import Service
from .session import build_request, fold_request_header, request_hash

_EVENTS = {"agent/created": "serial", "agent/status": "emit",
           "agent/assistant-stream": "emit",
           "agent/pre-step": "waterfall", "agent/request": "waterfall",
           "agent/request-error": "waterfall", "agent/turn-stopping": "serial",
           "system-prompt/assemble": "waterfall"}


class Agent:
    """One live agent: its session, scope, inbox and configuration."""

    def __init__(self, agents, session, config, system_prompt, keep_thinking,
                 max_steps, max_retries, scope_key):
        self._agents = agents
        self.session = session
        self.config = dict(config)
        self.system_prompt = system_prompt
        self.keep_thinking = keep_thinking
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.key = scope_key
        self.ctx = agents.ctx.scoped(scope_key)
        self.inbox = []           # queued human messages
        self.pending = []         # injected context for the next request
        self.sent = []            # request hashes, in order
        self.status = "idle"
        self._cancel = False
        self.turn = max([e["data"]["turn"] for e in session.events
                         if e["type"] == "turn/start"], default=0)

    def followup(self, content):
        self.inbox.append({"content": content, "source": "human"})

    def inject(self, content, source="inject"):
        """Context that lands in the next admitted request."""
        self.pending.append({"content": content, "source": source})

    def send(self, content):
        """Queue ``content`` and run turns until the inbox is empty.
        Returns the last assistant text."""
        self.followup(content)
        return self.run()

    def run(self):
        last = None
        self._cancel = False
        while self.inbox:
            last = self._agents._turn(self)
            if self._cancel:
                break
        return last

    def cancel(self):
        """Stop at the next step boundary (the running request and tool
        finish first) and drop queued messages; the turn ends with reason
        ``cancelled``. Safe to call from another thread."""
        self._cancel = True
        self.inbox.clear()

    def _set_status(self, status):
        self.status = status
        self.ctx.emit("agent/status", self, status)


class Agents(Service):
    key = "agents"
    inject = ("sessions", "tools", "llm")

    def start(self):
        for ev, mode in _EVENTS.items():
            self.ctx.declare(ev, mode)
        self._live = {}

    def stop(self):
        self._live.clear()

    def create(self, session_id, config, system_prompt="", keep_thinking="turn",
               max_steps=16, max_retries=1, setup=None, seed=None):
        """Create an agent on a new session (``seed`` = forked events).

        ``setup(agent)`` runs in the setup window: after the agent and its
        scope exist, before ``agent/created`` and the first prompt. Use it
        to register scoped tools, guards and restrictions on
        ``agent.ctx``.
        """
        session = self.ctx.sessions.create(session_id, seed=seed)
        key = object()
        agent = Agent(self, session, config, system_prompt, keep_thinking,
                      max_steps, max_retries, key)
        if setup is not None:
            setup(agent)
        self.ctx.serial("agent/created", agent)
        self._live[session_id] = agent
        return agent

    def get(self, session_id):
        return self._live.get(session_id)

    # ----- the driver ---------------------------------------------------
    def _turn(self, agent):
        s = agent.session
        agent.turn += 1
        turn = agent.turn
        agent._set_status("running")
        s.append("turn/start", {"turn": turn})
        claimed = agent.pending + agent.inbox[:1]
        agent.pending, agent.inbox = [], agent.inbox[1:]
        decision = agent.ctx.waterfall(
            "agent/pre-step", agent, {"action": "enter", "messages": claimed},
            final=lambda a, d: d)
        if decision.get("action") != "enter" or not decision.get("messages"):
            s.append("turn/end", {"turn": turn, "reason": "rejected"})
            agent._set_status("idle")
            return None
        entering = decision["messages"]
        reason, last_text = "completed", None
        step = 0
        try:
            while True:
                step += 1
                if agent._cancel:
                    reason = "cancelled"
                    break
                if step > agent.max_steps:
                    reason = "max-steps"
                    break
                out = self._step(agent, turn, step, entering)
                entering = []
                if out is None:
                    reason = "error"
                    break
                last_text = out["content"]
                if out["tool_calls"]:
                    entering = list(agent.pending)
                    agent.pending = []
                    continue
                follow = [r for r in agent.ctx.serial("agent/turn-stopping", agent, out)
                          if r]
                if not follow and not agent.pending:
                    break
                entering = list(agent.pending) + [
                    {"content": f, "source": "turn-stopping"} for f in follow]
                agent.pending = []
        finally:
            s.append("turn/end", {"turn": turn, "reason": reason})
            agent._set_status("idle")
        return last_text

    def _step(self, agent, turn, step, entering):
        s = agent.session
        s.append("step/start", {"turn": turn, "step": step})
        config = agent.ctx.waterfall("agent/request", agent, dict(agent.config),
                                     final=lambda a, c: c)
        prompt = agent.ctx.waterfall("system-prompt/assemble", agent,
                                     agent.system_prompt, final=lambda a, p: p)
        last_sys = s.last("system/message")
        if (last_sys is None and prompt) or (
                last_sys is not None and last_sys["data"]["message"]["content"] != prompt):
            s.append("system/message", {"turn": turn, "step": step,
                                        "message": {"role": "system", "content": prompt}})
        for m in entering:
            s.append("user/message", {"content": m["content"], "source": m["source"]})
        header = {"config": config}
        schemas = self.ctx.tools.schemas(agent.key)
        if schemas:
            header["tools"] = schemas
        old = fold_request_header(s.events)
        if old != header:
            s.append("request/header", {"header": header,
                                        "reason": "initial" if old is None else "change"})

        out, attempt = None, 0
        while out is None:
            request = build_request(s.events, agent.keep_thinking)
            agent.sent.append(request_hash(request))
            try:
                out = self.ctx.llm.call(
                    request, scope=agent.key,
                    on_chunk=lambda c: agent.ctx.emit("agent/assistant-stream", agent, c))
            except Exception as exc:
                s.append("assistant/attempt", {"turn": turn, "step": step, "stream": [],
                                               "error": f"{type(exc).__name__}: {exc}"})
                action = agent.ctx.waterfall(
                    "agent/request-error", agent, exc, {"action": "fail"},
                    final=lambda a, e, d: d)
                attempt += 1
                limit = max(agent.max_retries, int(action.get("max_retries", 0)))
                if action.get("action") != "retry" or attempt > limit:
                    s.append("step/end", {"turn": turn, "step": step})
                    return None
        msg = {"role": "assistant", "content": out["content"]}
        if out.get("reasoning"):
            msg["reasoning"] = out["reasoning"]
        if out["tool_calls"]:
            msg["tool_calls"] = out["tool_calls"]
        s.append("assistant/message", {"turn": turn, "step": step, "message": msg,
                                       "stream": out["stream"], "usage": out["usage"]})
        for call in out["tool_calls"]:
            s.append("tool/call", {"turn": turn, "step": step, "call_id": call["id"],
                                   "name": call["name"], "arguments": call["arguments"]})
            result = self.ctx.tools.execute(call["id"], call["name"], call["arguments"],
                                            scope=agent.key)
            s.append("tool/result", result.session_data(turn, step, call["id"]))
            for c in result.contexts:
                agent.inject(c["content"], source="tool-context")
        s.append("step/end", {"turn": turn, "step": step})
        return out
