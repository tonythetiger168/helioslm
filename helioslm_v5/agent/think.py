"""HeliosLM v5.31 — think mode + experience reuse (Qwen3-Max direction).

Qwen3-Max-Thinking reuses intermediate reasoning across rounds
("experience-cumulative test-time scaling"). Our version at protocol
level:

- Tri-mode assistant output: @@think@@...@@end@@ | @@tool@@...@@end@@ |
  plain text. parse_tri_mode never swallows markers (same discipline as
  parse_chat_turn).
- ThinkSession: think blocks are recorded as assistant transcript turns
  (visible context, no gate, no execution — thinking commits to nothing
  actionable) and the turn continues. The loop's other semantics (gate on
  tools only, recovery, finish) are inherited unchanged from ChatSession.
- ExperienceStore: episodic memory keyed by normalized task text. On
  finish, (thought, final_answer) is stored; on a repeated task the
  policy can recall the prior thought and skip re-deriving it — the
  cheapest credible test-time-scaling substrate we have.

Deviation recorded: Qwen's variant reuses token-level reasoning traces
inside one long generation; ours reuses whole thoughts across turns at
the protocol level. Same principle, different granularity — ours is
auditable (transcript + replay).
"""
from dataclasses import dataclass, field

try:
    from .chat import (CHAT_SYSTEM, ChatSession, ChatTurn, ChatStep,
                       ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER)
    from .schema import (ToolCallError, parse_chat_turn)
except ImportError:
    from chat import (CHAT_SYSTEM, ChatSession, ChatTurn, ChatStep,
                      ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER)
    from schema import (ToolCallError, parse_chat_turn)

THINK_OPEN, THINK_CLOSE = "@@think@@", "@@end@@"


class ThinkParseError(ToolCallError):
    pass


def verify_think_replay(steps: list, registry, impls: dict) -> None:
    """Replay mirror of verify_chat_replay for tri-mode transcripts.

    Same contract (DIRECT routing, oracle escalate out of scope); the only
    difference is parse_tri_mode instead of parse_chat_turn, so think
    steps re-derive as ("think", content) with empty observation.
    """
    try:
        from .chat import ChatReplayMismatch
        from .tools import execute
        from .trajectory import ids_to_text
    except ImportError:
        from chat import ChatReplayMismatch
        from tools import execute
        from trajectory import ids_to_text

    for s in steps:
        text = ids_to_text(s.generated_ids)
        try:
            kind, payload = parse_tri_mode(text, registry)
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
        if kind in ("think", "text"):
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


def parse_tri_mode(text, registry=None) -> tuple:
    """Returns ("think", content) | ("tool", calls) | ("text", str).

    Think has its own open marker and reuses the @@end@@ close marker.
    Any marker occurrence forces strict parsing of its block; plain text
    is never silently promoted or demoted.
    """
    if not isinstance(text, str):
        raise ThinkParseError("assistant output must be a string")
    s = text.strip()
    if not s:
        raise ThinkParseError("empty assistant reply")
    if THINK_OPEN in s:
        if not (s.startswith(THINK_OPEN) and s.endswith(THINK_CLOSE)
                and s.count(THINK_OPEN) == 1 and s.count(THINK_CLOSE) == 1):
            raise ThinkParseError("malformed think block")
        return "think", s[len(THINK_OPEN):-len(THINK_CLOSE)].strip()
    return parse_chat_turn(s, registry)


@dataclass
class ExperienceStore:
    """Episodic memory: key -> list of (thought, outcome), most recent
    first. recall() returns the thought of the most recent success."""
    max_per_key: int = 4
    _mem: dict = field(default_factory=dict)

    @staticmethod
    def normalize(task_text: str) -> str:
        return " ".join(task_text.split()).lower()

    def remember(self, key: str, thought: str, outcome: str) -> None:
        entries = self._mem.setdefault(self.normalize(key), [])
        entries.insert(0, (thought, outcome))
        del entries[self.max_per_key:]

    def recall(self, key: str) -> str | None:
        """Most recent thought for this task (entries are newest-first
        and only successes are worth recalling — callers store failures
        with outcome='FAIL' and we skip them)."""
        for thought, outcome in self._mem.get(self.normalize(key), []):
            if outcome != "FAIL":
                return thought
        return None

    def __len__(self):
        return sum(len(v) for v in self._mem.values())


class ThinkSession(ChatSession):
    """ChatSession + think mode + experience store.

    Differences from ChatSession.send (recorded, not hidden):
    - assistant output parses via parse_tri_mode;
    - "think" -> assistant transcript turn, loop continues (no gate);
    - on finish, (thought, final) is remembered in the store.
    Replay: verify_chat_replay is compatible because think steps carry
    observation "" and no parsed call (same as text steps).
    """

    def __init__(self, *args, store: ExperienceStore | None = None, **kw):
        super().__init__(*args, **kw)
        # explicit None check: ExperienceStore defines __len__, so an empty
        # store is falsy and `store or ExperienceStore()` would silently
        # replace the caller's store (found by T25)
        self.store = store if store is not None else ExperienceStore()

    def send(self, user_text: str, seed: int = 0) -> str | None:
        try:
            from .gate import Route
            from .tools import execute
            from .trajectory import text_to_ids
        except ImportError:
            from gate import Route
            from tools import execute
            from trajectory import text_to_ids

        self.turns.append(ChatTurn(ROLE_ASSISTANT and "##user##" or "##user##",
                                   user_text))
        final = None
        last_thought = None
        for i in range(self.max_steps):
            prompt = self.build_prompt()
            gen_text = self.model_fn(prompt, seed, i)
            try:
                kind, payload = parse_tri_mode(gen_text, self.registry)
                perr = None
            except ToolCallError as e:
                kind, payload, perr = "parse_error", None, str(e)
            call = payload[0] if kind == "tool" else None
            if kind == "think":
                last_thought = payload
                self.turns.append(ChatTurn(ROLE_ASSISTANT, gen_text.strip()))
                self.steps.append(ChatStep(
                    i, text_to_ids(prompt), text_to_ids(gen_text), "think",
                    None, None, "-", ""))
                continue  # thinking commits to nothing: no gate, no exec
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
                        obs = f"TOOL_ERROR: {e}"
            self.turns.append(ChatTurn(ROLE_ASSISTANT, gen_text.strip()))
            self.turns.append(ChatTurn(ROLE_TOOL, obs))
            self.steps.append(ChatStep(
                i, text_to_ids(prompt), text_to_ids(gen_text), kind,
                call, perr, route_s, obs))
            if call is not None and call.name == "finish":
                final = (call.args["answer"] if route_s == "DIRECT"
                         else obs.removeprefix("FINISH: "))
                if last_thought is not None:
                    self.store.remember(user_text, last_thought, final)
                break
        return final
