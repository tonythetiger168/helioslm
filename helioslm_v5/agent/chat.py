"""HeliosLM v5.30 — chat capability: multi-turn conversation over the
existing tool protocol.

Design decisions (recorded, not hidden):
- Dual-mode assistant output: a plain text reply OR a @@tool@@ block
  (schema.parse_chat_turn). Text replies bypass the gate entirely; the
  gate governs tool routing only. Rationale: a text reply commits to
  nothing actionable, so there is nothing to route or escalate.
- Transcript markers are printable ASCII (##user## / ##assistant## /
  ##tool##) because the model vocab is ord(c) < 1024 char-level.
- Recovery mirrors loop.py: malformed output becomes a PARSE_ERROR tool
  message in the transcript, never a crash.
- Replay mirrors trajectory.verify_replay: it re-derives observations
  under DIRECT routing; oracle/escalate observations are outside replay
  scope by design (same contract as trajectories).
"""
import re
from dataclasses import dataclass

try:
    from .gate import Route
    from .loop import render_tool_docs
    from .schema import (ToolCall, ToolCallError, ToolRegistry,
                         parse_chat_turn, render_tool_call)
    from .tools import execute
    from .trajectory import ids_to_text, text_to_ids
except ImportError:
    from gate import Route
    from loop import render_tool_docs
    from schema import (ToolCall, ToolCallError, ToolRegistry,
                        parse_chat_turn, render_tool_call)
    from tools import execute
    from trajectory import ids_to_text, text_to_ids

CHAT_SYSTEM = (
    "You are HeliosLM, a chat assistant. Reply in plain text, or call a "
    "tool by emitting exactly one tool block of the form "
    '@@tool@@{"calls":[{"name":...,"args":{...}}]}@@end@@. '
    "Available tools: %%TOOLS%%.")

ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL = "##user##", "##assistant##", "##tool##"
FOLLOWUP_TEXT = "Now multiply that by 2."


@dataclass(frozen=True)
class ChatTurn:
    role: str          # ROLE_USER / ROLE_ASSISTANT / ROLE_TOOL
    content: str


@dataclass(frozen=True)
class ChatStep:
    index: int
    prompt_ids: tuple
    generated_ids: tuple
    kind: str          # "text" | "tool" | "parse_error"
    parsed: ToolCall | None
    parse_error: str | None
    route: str
    observation: str   # "" for text replies


class ChatReplayMismatch(Exception):
    """Replayed chat step differs from the recorded session."""


class ChatSession:
    """Multi-turn chat over the tool protocol.

    send() drives up to max_steps assistant generations per user turn:
    text reply -> turn ends immediately (gate untouched); tool block ->
    gate -> execute/escalate -> observation re-enters the transcript as
    a ##tool## turn; finish ends the turn with a final answer.
    """

    def __init__(self, model_fn, registry: ToolRegistry, impls: dict,
                 gate, max_steps: int = 8, confidence_fn=None,
                 system: str | None = None):
        self.model_fn, self.registry, self.impls = model_fn, registry, impls
        self.gate, self.max_steps, self.confidence_fn = \
            gate, max_steps, confidence_fn
        self.system = system or CHAT_SYSTEM.replace(
            "%%TOOLS%%", render_tool_docs(registry))
        self.turns: list[ChatTurn] = []
        self.steps: list[ChatStep] = []

    def build_prompt(self) -> str:
        parts = [self.system]
        parts += [f"{t.role} {t.content}" for t in self.turns]
        return "\n".join(parts)

    def send(self, user_text: str, seed: int = 0) -> str | None:
        self.turns.append(ChatTurn(ROLE_USER, user_text))
        final = None
        for i in range(self.max_steps):
            prompt = self.build_prompt()
            gen_text = self.model_fn(prompt, seed, i)
            try:
                kind, payload = parse_chat_turn(gen_text, self.registry)
                perr = None
            except ToolCallError as e:
                kind, payload, perr = "parse_error", None, str(e)
            call = payload[0] if kind == "tool" else None
            if kind == "text":
                self.turns.append(ChatTurn(ROLE_ASSISTANT, payload))
                self.steps.append(ChatStep(
                    i, text_to_ids(prompt), text_to_ids(payload), "text",
                    None, None, "-", ""))
                return payload
            if call is None:  # parse_error
                route_s, obs = "-", f"PARSE_ERROR: {perr}"
            else:
                ctx = {"step": i, "task": user_text,
                       "confidence": (self.confidence_fn(prompt, i)
                                      if self.confidence_fn else None)}
                route = self.gate.decide(call, ctx)
                route_s = route.value
                if route == Route.ESCALATE:
                    obs = self.gate.escalate(call, ctx)
                else:
                    try:
                        obs = execute(call, self.registry, self.impls)
                    except ToolCallError as e:
                        obs = f"TOOL_ERROR: {e}"  # recovery, not crash
            self.turns.append(ChatTurn(ROLE_ASSISTANT, gen_text.strip()))
            self.turns.append(ChatTurn(ROLE_TOOL, obs))
            self.steps.append(ChatStep(
                i, text_to_ids(prompt), text_to_ids(gen_text), kind,
                call, perr, route_s, obs))
            if call is not None and call.name == "finish":
                final = (call.args["answer"] if route_s == "DIRECT"
                         else obs.removeprefix("FINISH: "))
                break
        return final


def verify_chat_replay(steps: list, registry: ToolRegistry,
                       impls: dict) -> None:
    """Re-derive every observation from the recorded generation.

    Same contract as trajectory.verify_replay: assumes DIRECT routing
    (oracle escalate observations are not replayable by design).
    """
    for s in steps:
        text = ids_to_text(s.generated_ids)
        try:
            kind, payload = parse_chat_turn(text, registry)
            perr = None
        except ToolCallError as e:
            kind, payload, perr = "parse_error", None, str(e)
        call = payload[0] if kind == "tool" else None
        if (call is None) != (s.parsed is None):
            raise ChatReplayMismatch(f"step {s.index}: parse presence differs")
        if call is not None and call != s.parsed:
            raise ChatReplayMismatch(f"step {s.index}: parsed call differs")
        if perr != s.parse_error:
            raise ChatReplayMismatch(f"step {s.index}: parse error differs")
        if kind == "text":
            obs = ""
        elif call is None:
            obs = f"PARSE_ERROR: {perr}"
        else:
            try:
                obs = execute(call, registry, impls)
            except ToolCallError as e:
                obs = f"TOOL_ERROR: {e}"
        if obs != s.observation:
            raise ChatReplayMismatch(f"step {s.index}: observation differs:\n"
                                     f"  replay  : {obs!r}\n"
                                     f"  recorded: {s.observation!r}")


# --- scripted policy for tests and SFT data generation ---------------------

def _transcript(prompt: str) -> list:
    out = []
    for line in prompt.splitlines():
        for role in (ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL):
            if line.startswith(role + " "):
                out.append((role, line[len(role) + 1:]))
                break
    return out


def scripted_chat_policy(prompt: str, seed: int, step: int) -> str:
    """Deterministic correct policy over chat-rendered transcripts.

    Handles three intents (v5.30 scope):
    - direct question ("The magic word is X. What is the magic word?")
      -> plain text reply (no tool)
    - env calc task ("Compute the value of: ...") -> calc, then finish
    - follow-up (FOLLOWUP_TEXT) -> calc on the previous FINISH answer,
      then finish (exercises transcript memory)
    """
    turns = _transcript(prompt)
    users = [c for r, c in turns if r == ROLE_USER]
    tools = [c for r, c in turns if r == ROLE_TOOL]
    last_user = users[-1]
    last_user_idx = max(i for i, (r, _) in enumerate(turns)
                        if r == ROLE_USER)
    tools_since_user = sum(1 for i, (r, _) in enumerate(turns)
                           if r == ROLE_TOOL and i > last_user_idx)
    m = re.match(r"The magic word is (\w+)\. What is the magic word\?+$",
                 last_user)
    if m:
        return m.group(1)
    m = re.match(r"Compute the value of: (.*)", last_user)
    if m:
        if tools_since_user == 0:
            return render_tool_call([ToolCall("calc", {"expr": m.group(1)})])
        return render_tool_call([ToolCall("finish", {"answer": tools[-1]})])
    m = re.match(r'Apply (\w+) to the string: "(.*)"', last_user)
    if m:
        if tools_since_user == 0:
            return render_tool_call([ToolCall(
                "str_op", {"s": m.group(2), "op": m.group(1), "n": 0})])
        return render_tool_call([ToolCall("finish", {"answer": tools[-1]})])
    m = re.match(
        r"First compute: (.*?)\. Then apply (\w+) to the digits of the result\.",
        last_user)
    if m:
        if tools_since_user == 0:
            return render_tool_call([ToolCall("calc", {"expr": m.group(1)})])
        if tools_since_user == 1:
            return render_tool_call([ToolCall(
                "str_op", {"s": tools[-1], "op": m.group(2), "n": 0})])
        return render_tool_call([ToolCall("finish", {"answer": tools[-1]})])
    if last_user == FOLLOWUP_TEXT:
        if tools_since_user == 0:
            base = next((c[len("FINISH: "):] for c in reversed(tools)
                         if c.startswith("FINISH: ")), "0")
            return render_tool_call([ToolCall("calc", {"expr": f"{base} * 2"})])
        return render_tool_call([ToolCall("finish", {"answer": tools[-1]})])
    return render_tool_call([ToolCall("finish", {"answer": tools[-1] if tools else ""})])
