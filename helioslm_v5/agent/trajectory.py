import json
from dataclasses import dataclass, field

try:
    from .schema import ToolCall, ToolCallError, ToolRegistry, parse_tool_call
    from .tools import execute
except ImportError:
    from schema import ToolCall, ToolCallError, ToolRegistry, parse_tool_call
    from tools import execute


class ReplayMismatch(Exception):
    """Replayed step differs from the recorded trajectory."""


@dataclass(frozen=True)
class Step:
    index: int
    prompt_ids: tuple
    generated_ids: tuple
    parsed: ToolCall | None
    parse_error: str | None
    route: str
    observation: str


@dataclass
class Trajectory:
    task: str
    seed: int
    steps: list = field(default_factory=list)
    final_answer: str | None = None

    def to_dict(self) -> dict:
        return {"task": self.task, "seed": self.seed,
                "final_answer": self.final_answer,
                "steps": [{"index": s.index,
                           "prompt_ids": list(s.prompt_ids),
                           "generated_ids": list(s.generated_ids),
                           "parsed": None if s.parsed is None else
                                     {"name": s.parsed.name,
                                      "args": s.parsed.args},
                           "parse_error": s.parse_error,
                           "route": s.route,
                           "observation": s.observation}
                          for s in self.steps]}

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        steps = []
        for raw in d["steps"]:
            p = raw["parsed"]
            steps.append(Step(raw["index"], tuple(raw["prompt_ids"]),
                              tuple(raw["generated_ids"]),
                              None if p is None else ToolCall(p["name"],
                                                              p["args"]),
                              raw["parse_error"], raw["route"],
                              raw["observation"]))
        return Trajectory(d["task"], d["seed"], steps, d["final_answer"])

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def loads(s: str) -> "Trajectory":
        return Trajectory.from_dict(json.loads(s))


def text_to_ids(text: str) -> tuple:
    return tuple(ord(c) for c in text)


def ids_to_text(ids) -> str:
    return "".join(chr(int(i)) for i in ids)


def verify_replay(traj: Trajectory, registry: ToolRegistry,
                  impls: dict) -> None:
    for s in traj.steps:
        text = ids_to_text(s.generated_ids)
        try:
            call, perr = parse_tool_call(text, registry)[0], None
        except ToolCallError as e:
            call, perr = None, str(e)
        if (call is None) != (s.parsed is None):
            raise ReplayMismatch(f"step {s.index}: parse presence differs")
        if call is not None and call != s.parsed:
            raise ReplayMismatch(f"step {s.index}: parsed call differs")
        if perr != s.parse_error:
            raise ReplayMismatch(f"step {s.index}: parse error differs")
        obs = f"PARSE_ERROR: {perr}" if call is None \
            else execute(call, registry, impls)
        if obs != s.observation:
            raise ReplayMismatch(f"step {s.index}: observation differs:\n"
                                 f"  replay  : {obs!r}\n"
                                 f"  recorded: {s.observation!r}")
