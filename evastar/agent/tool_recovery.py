"""Tool-call recovery layer (roadmap Phase AC, AC3).

``schema.parse_tool_call`` is strict and stays strict. Models in the
wild emit near-misses: the block wrapped in a code fence or prose, a
truncated closing marker, the OpenAI ``arguments`` key (often as a JSON
string), a bare call object without the ``calls`` wrapper, trailing
commas. ``recover_tool_call`` tries a FIXED list of named repairs, in
order, and returns the calls together with the repairs it applied, so
every recovery is visible in the trajectory.

What is never repaired (still a ``ToolCallError``):
- unknown tool names and argument name/type mismatches — the registry
  check runs unchanged on the repaired calls;
- JSON that is truncated or otherwise unparseable after the repairs;
- more than one tool block in one reply (ambiguous intent).

The only exception type that can leave this module is ``ToolCallError``
(pinned by the T99 fuzz test).
"""
import json
import re

from evastar.agent.schema import (MARK_CLOSE, MARK_OPEN, MAX_TEXT,
                                      ToolCallError, parse_tool_call,
                                      render_tool_call, ToolCall)

REPAIRS = ("extract_block", "close_marker", "strip_inner_ws",
           "trailing_commas", "arguments_key", "args_json_string",
           "wrap_calls")

_FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.S)


def _strip_trailing_commas(body):
    """Remove commas directly before } or ] OUTSIDE string literals."""
    out, in_str, esc = [], False, False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < len(body) and body[j] in " \t\r\n":
                j += 1
            if j < len(body) and body[j] in "}]":
                i += 1
                continue
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _normalize_call(c, applied):
    if not isinstance(c, dict):
        return c
    c = dict(c)
    if "arguments" in c and "args" not in c:
        c["args"] = c.pop("arguments")
        applied.add("arguments_key")
    if isinstance(c.get("args"), str):
        try:
            parsed = json.loads(c["args"])
        except json.JSONDecodeError as e:
            raise ToolCallError(f"args string is not JSON: {e}") from e
        c["args"] = parsed
        applied.add("args_json_string")
    return c


def recover_tool_call(text, registry=None):
    """-> (calls, repairs applied as a sorted tuple). Strict-valid input
    returns ``(calls, ())`` with exactly the strict parser's calls.

    Deeply nested payloads can make ``json.loads`` raise RecursionError
    (CPython <= 3.12's decoder recurses in Python); that is converted to
    ToolCallError so the module's only-ToolCallError contract holds on
    every supported Python.
    """
    try:
        return _recover(text, registry)
    except RecursionError as e:
        raise ToolCallError("payload nested too deeply") from e


def _recover(text, registry):
    if not isinstance(text, str):
        raise ToolCallError("tool output must be a string")
    if len(text) > 4 * MAX_TEXT:
        raise ToolCallError("tool output too large")
    try:
        return parse_tool_call(text, registry), ()
    except ToolCallError:
        pass

    applied = set()
    s = text.strip()
    n_open = s.count(MARK_OPEN)
    if n_open > 1:
        raise ToolCallError("more than one tool block in one reply")
    if n_open == 0:
        fences = _FENCE.findall(s)
        if len(fences) == 1 and fences[0].strip().startswith("{"):
            body = fences[0].strip()       # fenced bare JSON, no markers
            applied.add("extract_block")
        else:
            raise ToolCallError("no tool block found")
    else:
        start = s.index(MARK_OPEN)
        end = s.find(MARK_CLOSE, start)
        if start > 0 or (end != -1 and end + len(MARK_CLOSE) < len(s)):
            applied.add("extract_block")
        if end == -1:
            body = s[start + len(MARK_OPEN):]
            fence_tail = body.rfind("```")
            if fence_tail != -1:
                body = body[:fence_tail]
            applied.add("close_marker")
        else:
            body = s[start + len(MARK_OPEN):end]
    if body != body.strip():
        body = body.strip()
        applied.add("strip_inner_ws")
    if not body:
        raise ToolCallError("empty tool payload")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        fixed = _strip_trailing_commas(body)
        if fixed == body:
            raise ToolCallError("invalid JSON (not repairable)")
        try:
            payload = json.loads(fixed)
        except json.JSONDecodeError as e:
            raise ToolCallError(f"invalid JSON after repairs: {e}") from e
        applied.add("trailing_commas")
    except RecursionError as e:
        raise ToolCallError("payload nested too deeply") from e

    if isinstance(payload, dict) and set(payload) == {"calls"}:
        calls = payload["calls"]
    elif isinstance(payload, dict) and "name" in payload:
        calls = [payload]
        applied.add("wrap_calls")
    elif isinstance(payload, list):
        calls = payload
        applied.add("wrap_calls")
    else:
        raise ToolCallError("payload is not a tool call")
    if not isinstance(calls, list):
        raise ToolCallError("'calls' must be a list")
    calls = [_normalize_call(c, applied) for c in calls]
    # re-validate through the strict parser on the canonical rendering
    try:
        canonical = render_tool_call([ToolCall(c["name"], c["args"])
                                      for c in calls])
    except (KeyError, TypeError, ValueError) as e:
        raise ToolCallError(f"call shape not repairable: {e!r}") from e
    for c in calls:
        if set(c) != {"name", "args"}:
            raise ToolCallError("call must have exactly 'name' and 'args'")
    return parse_tool_call(canonical, registry), tuple(sorted(applied))
