import json
from dataclasses import dataclass, field

MARK_OPEN, MARK_CLOSE = "@@tool@@", "@@end@@"
MAX_JSON_DEPTH, MAX_CALLS, MAX_TEXT = 32, 16, 64 * 1024


class ToolCallError(Exception):
    """Raised for ANY malformed tool output. Never swallowed silently."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict = field(default_factory=dict)
    returns: type = str

    def validate_args(self, args: object) -> None:
        if not isinstance(args, dict):
            raise ToolCallError(f"{self.name}: args must be an object")
        got, want = set(args), set(self.params)
        if got != want:
            raise ToolCallError(
                f"{self.name}: arg mismatch (missing={sorted(want - got)}, "
                f"extra={sorted(got - want)})")
        for k, t in self.params.items():
            v = args[k]
            if type(v) is not t:
                raise ToolCallError(
                    f"{self.name}: arg '{k}' must be {t.__name__}, "
                    f"got {type(v).__name__}")


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


class ToolRegistry:
    def __init__(self, specs: list):
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool names")
        self._specs = dict(zip(names, specs))

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise ToolCallError(f"unknown tool: {name!r}")
        return self._specs[name]

    @property
    def specs(self) -> dict:
        return dict(self._specs)


def render_tool_call(calls: list) -> str:
    if not calls:
        raise ValueError("render_tool_call: empty calls")
    payload = {"calls": [{"name": c.name, "args": c.args} for c in calls]}
    return f"{MARK_OPEN}{json.dumps(payload, ensure_ascii=False)}{MARK_CLOSE}"


def _check_depth(obj: object, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ToolCallError(f"payload nested deeper than {MAX_JSON_DEPTH}")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _check_depth(v, depth + 1)


def parse_tool_call(text: str, registry=None) -> list:
    if not isinstance(text, str):
        raise ToolCallError("tool output must be a string")
    s = text.strip()
    if not (s.startswith(MARK_OPEN) and s.endswith(MARK_CLOSE)):
        raise ToolCallError("missing or misplaced tool markers")
    body = s[len(MARK_OPEN):-len(MARK_CLOSE)]
    if not body:
        raise ToolCallError("empty tool payload")
    if body != body.strip():
        raise ToolCallError("whitespace inside tool markers")
    if len(body) > MAX_TEXT:
        raise ToolCallError("tool payload too large")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"invalid JSON: {e}") from e
    _check_depth(payload)
    if not isinstance(payload, dict) or set(payload) != {"calls"}:
        raise ToolCallError("payload must be exactly {'calls': [...]}")
    raw = payload["calls"]
    if not isinstance(raw, list) or not raw:
        raise ToolCallError("'calls' must be a non-empty list")
    if len(raw) > MAX_CALLS:
        raise ToolCallError(f"more than {MAX_CALLS} calls")
    out = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or set(c) != {"name", "args"}:
            raise ToolCallError(f"call[{i}]: must have exactly 'name' and 'args'")
        if not isinstance(c["name"], str):
            raise ToolCallError(f"call[{i}]: name must be a string")
        call = ToolCall(name=c["name"], args=c["args"])
        if registry is not None:
            registry.spec(call.name).validate_args(call.args)
        out.append(call)
    return out
