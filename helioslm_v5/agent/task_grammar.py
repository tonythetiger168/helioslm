"""HeliosLM v5.35 — the task grammar: ONE table for parse AND render.

Why this exists (recorded): the task-text grammar lived in three places
-- env generators (f-strings), grounding's parser (regexes), and test
policies (their own regexes). The v5.34 T33 debug chain fixed the SAME
class of drift bug three separate times (greedy-first parse, tail
handling, char-class gaps). Skew between a producer and its consumers
is a structural hazard, not a coding-slip hazard; the fix is structural:
this module is the single source of truth. Envs RENDER through it;
grounding and policies PARSE through it. If render and parse ever
disagree, T34 fails immediately.

Adding a task family = adding one entry to GRAMMAR + using render() in
the env. Nothing else changes.
"""
import re

# Each kind: render(spec) -> task text; parse(text) -> spec or None.
# parse uses the debugged anchors from the v5.34 chain: specific kinds
# first, calc LAST (its pattern is a prefix of write_read/accumulate).
GRAMMAR = {}


def _reg(kind):
    def deco(pair):
        GRAMMAR[kind] = pair
        return pair
    return deco


@_reg("str")
class _Str:
    @staticmethod
    def render(spec):
        op_text = (f"{spec['op']} {spec['n']} times"
                   if spec["op"] == "repeat" else spec["op"])
        return f'Apply {op_text} to the string: "{spec["s"]}"'

    _RX = re.compile(r'Apply (\w+)(?: (\d+) times)? to the string: "(.*)"')

    @classmethod
    def parse(cls, text):
        m = cls._RX.fullmatch(text)
        if not m:
            return None
        return {"kind": "str", "op": m.group(1),
                "n": int(m.group(2) or 0), "s": m.group(3)}


@_reg("compose")
class _Compose:
    @staticmethod
    def render(spec):
        return (f"First compute: {spec['expr']}. Then apply {spec['op']} "
                f"to the digits of the result.")

    _RX = re.compile(
        r"First compute: (.*?)\. Then apply (\w+) to the digits of the "
        r"result\.")

    @classmethod
    def parse(cls, text):
        m = cls._RX.fullmatch(text)
        if not m:
            return None
        return {"kind": "compose", "expr": m.group(1), "op": m.group(2)}


@_reg("write_read")
class _WriteRead:
    @staticmethod
    def render(spec):
        return (f"Compute the value of: {spec['expr']}. Write the result "
                f"to {spec['path']}, then read {spec['path']} and finish "
                f"with its exact content.")

    _RX = re.compile(
        r"Compute the value of: (.*?)\. Write the result to (\S+?), then")

    @classmethod
    def parse(cls, text):
        m = cls._RX.match(text)
        if not m:
            return None
        return {"kind": "write_read", "expr": m.group(1),
                "path": m.group(2)}


@_reg("write_transform")
class _WriteTransform:
    @staticmethod
    def render(spec):
        op_text = (f"{spec['op']} {spec['n']} times"
                   if spec["op"] == "repeat" else spec["op"])
        return (f'Write the string "{spec["s"]}" to {spec["a"]}. Then '
                f"apply {op_text} to it and write the result to {spec['b']}. "
                f"Read {spec['b']} and finish with its exact content.")

    _RX = re.compile(
        r'Write the string "(.*?)" to (\S+)\. Then apply (\w+)'
        r"(?: (\d+) times)? to it and write the result to (\S+)\.")

    @classmethod
    def parse(cls, text):
        m = cls._RX.match(text)
        if not m:
            return None
        return {"kind": "write_transform", "s": m.group(1),
                "a": m.group(2), "op": m.group(3),
                "n": int(m.group(4) or 0), "b": m.group(5)}


@_reg("accumulate")
class _Accumulate:
    @staticmethod
    def render(spec):
        return (f"Compute the value of: {spec['e1']} and write the result "
                f"to {spec['p1']}. Compute the value of: {spec['e2']} and "
                f"write the result to {spec['p2']}. Read both files, "
                f"compute the sum of the two values, and finish with the "
                f"sum.")

    _RX = re.compile(
        r"Compute the value of: (.*?) and write the result to (\S+)\. "
        r"Compute the value of: (.*?) and write the result to (\S+)\. ")

    @classmethod
    def parse(cls, text):
        m = cls._RX.match(text)
        if not m:
            return None
        return {"kind": "accumulate", "e1": m.group(1), "p1": m.group(2),
                "e2": m.group(3), "p2": m.group(4)}


@_reg("calc")
class _Calc:
    @staticmethod
    def render(spec):
        return f"Compute the value of: {spec['expr']}"

    # anchored: no dots (exprs use + - * // % and parens only), so
    # write_read/accumulate texts (which share the prefix) fall through
    _RX = re.compile(r"Compute the value of: [-0-9+*/()% ]*")

    @classmethod
    def parse(cls, text):
        m = cls._RX.fullmatch(text)
        if not m or not m.group(0).endswith(" "):
            # reject the bare prefix ("Compute the value of: " with no
            # expression) and any non-expr tail caught by the class
            body = m.group(0)[len("Compute the value of: "):] if m else ""
            if not body.strip():
                return None
        if not m:
            return None
        expr = m.group(0)[len("Compute the value of: "):]
        return {"kind": "calc", "expr": expr} if expr.strip() else None


@_reg("echo")
class _Echo:
    @staticmethod
    def render(spec):
        return f'Repeat back exactly: "{spec["s"]}"'

    _RX = re.compile(r'Repeat back exactly: "(.*)"')

    @classmethod
    def parse(cls, text):
        m = cls._RX.fullmatch(text)
        return {"kind": "echo", "s": m.group(1)} if m else None


# dispatch: specific kinds first, calc/echo last
_PARSE_ORDER = ("str", "compose", "write_read", "write_transform",
                "accumulate", "echo", "calc")


def render(kind, **fields):
    return GRAMMAR[kind].render(fields)


def parse(text):
    """Parse task text -> spec dict (with "kind") or None. This is the
    ONLY parser consumers should use."""
    for kind in _PARSE_ORDER:
        spec = GRAMMAR[kind].parse(text)
        if spec is not None:
            return spec
    return None
