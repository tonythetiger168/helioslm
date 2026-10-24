"""T99 — roadmap Phase AC, AC3: tool-call recovery layer, fuzz-tested.

Pins:

1. strict-valid input is returned exactly as the strict parser returns
   it, with no repairs recorded;
2. each repairable corruption class recovers to EXACTLY the original
   calls, with the expected repair named;
3. never-repaired classes (unknown tool, wrong arg type, truncated JSON,
   two blocks) raise ToolCallError;
4. seeded random fuzz (byte deletions / insertions / swaps on valid
   blocks): the only exception that ever escapes is ToolCallError, and
   any recovered call still passes the registry check;
5. the recovery-rate table (``RECOVERY_TABLE``) the docs quote.

Run from repo root: python3 evastar/tests/test_tool_recovery.py
"""
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.agent.schema import (MARK_CLOSE, MARK_OPEN, ToolCall,  # noqa: E402
                                      ToolCallError, ToolRegistry, ToolSpec,
                                      parse_tool_call, render_tool_call)
from evastar.agent.tool_recovery import recover_tool_call  # noqa: E402

REG = ToolRegistry([
    ToolSpec("read_file", "read a file", {"path": str}),
    ToolSpec("run_tests", "run tests", {"target": str, "verbose": bool}),
    ToolSpec("edit", "replace text", {"path": str, "old": str, "new": str,
                                      "count": int}),
])


def _sample_calls(rng):
    pool = [
        ToolCall("read_file", {"path": rng.choice(["a.py", "src/x y.py", "é.txt"])}),
        ToolCall("run_tests", {"target": "tests/", "verbose": rng.random() < .5}),
        ToolCall("edit", {"path": "m.py", "old": 'say("hi, {x}")',
                          "new": "a,\n]b}", "count": rng.randint(0, 3)}),
    ]
    return rng.sample(pool, rng.randint(1, 3))


def _body(calls):
    return json.dumps({"calls": [{"name": c.name, "args": c.args}
                                 for c in calls]}, ensure_ascii=False)


# corruption classes: name -> (fn(calls) -> text, expected repair or None)
def _fence(calls):
    return "Sure, calling it now:\n```json\n" + render_tool_call(calls) + "\n```\nDone."


def _no_close(calls):
    return MARK_OPEN + _body(calls)


def _inner_ws(calls):
    return MARK_OPEN + "\n  " + _body(calls) + "\n" + MARK_CLOSE


def _trailing_commas(calls):
    calls_json = ", ".join(
        '{"name": %s, "args": {%s,},}' % (
            json.dumps(c.name),
            ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in c.args.items()))
        for c in calls)
    return MARK_OPEN + '{"calls": [' + calls_json + ",]}" + MARK_CLOSE


def _openai_style(calls):
    return MARK_OPEN + json.dumps({"calls": [
        {"name": c.name, "arguments": json.dumps(c.args)} for c in calls]}) + MARK_CLOSE


def _bare_list(calls):
    return MARK_OPEN + json.dumps([{"name": c.name, "args": c.args}
                                   for c in calls]) + MARK_CLOSE


def _fenced_bare_json(calls):
    return "```json\n" + _body(calls) + "\n```"


REPAIRABLE = {
    "fence_and_prose": (_fence, "extract_block"),
    "missing_close_marker": (_no_close, "close_marker"),
    "inner_whitespace": (_inner_ws, "strip_inner_ws"),
    "trailing_commas": (_trailing_commas, "trailing_commas"),
    "openai_arguments_string": (_openai_style, "args_json_string"),
    "bare_call_list": (_bare_list, "wrap_calls"),
    "fenced_json_no_markers": (_fenced_bare_json, "extract_block"),
}

NOT_REPAIRED = {
    "unknown_tool": lambda: render_tool_call([ToolCall("rm_rf", {"path": "/"})]),
    "wrong_arg_type": lambda: render_tool_call(
        [ToolCall("run_tests", {"target": "t", "verbose": "yes"})]),
    "truncated_json": lambda: MARK_OPEN + '{"calls": [{"name": "read_file", "ar',
    "two_blocks": lambda: render_tool_call([ToolCall("read_file", {"path": "a"})]) * 2,
    "prose_only": lambda: "I will read the file now.",
    "deep_nesting": lambda: MARK_OPEN + "[" * 5000 + "]" * 5000 + MARK_CLOSE,
}

RECOVERY_TABLE = {}


def test_strict_input_unchanged():
    rng = random.Random(0)
    for _ in range(50):
        calls = _sample_calls(rng)
        text = render_tool_call(calls)
        got, repairs = recover_tool_call(text, REG)
        assert got == parse_tool_call(text, REG) == calls and repairs == ()
    print("PASS test_strict_input_unchanged")


def test_repairable_classes():
    rng = random.Random(1)
    for name, (fn, repair) in REPAIRABLE.items():
        ok = 0
        for _ in range(40):
            calls = _sample_calls(rng)
            text = fn(calls)
            try:
                parse_tool_call(text, REG)
                raise AssertionError(f"{name}: strict parser accepted it")
            except ToolCallError:
                pass
            got, repairs = recover_tool_call(text, REG)
            assert got == calls, (name, text)
            assert repair in repairs, (name, repairs)
            ok += 1
        RECOVERY_TABLE[name] = ok / 40
    print(f"PASS test_repairable_classes ({len(REPAIRABLE)} classes, all 40/40)")


def test_not_repaired_classes():
    for name, fn in NOT_REPAIRED.items():
        try:
            recover_tool_call(fn(), REG)
        except ToolCallError:
            RECOVERY_TABLE[name] = 0.0
            continue
        raise AssertionError(f"{name}: must not be repaired")
    print(f"PASS test_not_repaired_classes ({len(NOT_REPAIRED)} classes)")


def test_random_fuzz_never_crashes():
    rng = random.Random(2)
    alphabet = '{}[]",:@ \n\\abcdeftrueslnm0123456789`'
    recovered = errors = 0
    for _ in range(3000):
        text = list(render_tool_call(_sample_calls(rng)))
        for _ in range(rng.randint(1, 4)):
            op, i = rng.random(), rng.randrange(len(text))
            if op < 0.4:
                del text[i]
            elif op < 0.8:
                text.insert(i, rng.choice(alphabet))
            else:
                j = rng.randrange(len(text))
                text[i], text[j] = text[j], text[i]
            if not text:
                text = ["x"]
        try:
            calls, _ = recover_tool_call("".join(text), REG)
        except ToolCallError:
            errors += 1
            continue
        for c in calls:                       # still registry-valid
            REG.spec(c.name).validate_args(c.args)
        recovered += 1
    assert recovered + errors == 3000 and recovered > 0 and errors > 0
    RECOVERY_TABLE["random_byte_fuzz"] = round(recovered / 3000, 4)
    print(f"PASS test_random_fuzz_never_crashes (3000 cases: {recovered} "
          f"parsed to registry-valid calls, {errors} ToolCallError, 0 other)")


if __name__ == "__main__":
    test_strict_input_unchanged()
    test_repairable_classes()
    test_not_repaired_classes()
    test_random_fuzz_never_crashes()
    print(json.dumps(RECOVERY_TABLE, indent=1))
