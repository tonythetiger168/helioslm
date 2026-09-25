"""HeliosLM v5.30 — T20: chat capability tests.

Covers: dual-mode protocol, text-reply gate bypass, parse/tool error
recovery, follow-up transcript memory, and replay integrity. Also re-runs
T17 (test_benchmark) to prove the v5.30 protocol extension is regression-
free for the pure tool pipeline.

Repo style: plain asserts + __main__ runner (no pytest dependency).
Run from repo root: python3 helioslm_v5/tests/test_chat.py
"""
import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from benchmark import scripted_correct_policy
from chat import (ChatReplayMismatch, ChatSession, FOLLOWUP_TEXT,
                  ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER,
                  scripted_chat_policy, verify_chat_replay)
from envs import make_envs
from gate import FixedGate, Gate, Route
from schema import (ToolCall, ToolCallError, parse_chat_turn,
                    render_tool_call)
from tools import build_default_registry


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def reg_impls():
    return build_default_registry("/tmp/helioslm_chat_test")


def make_session(model_fn, reg_impls_, gate=None, max_steps=8):
    reg, impls = reg_impls_
    return ChatSession(model_fn, reg, impls,
                       gate or FixedGate(Route.DIRECT), max_steps=max_steps)


# --- protocol ---------------------------------------------------------------

def test_parse_chat_turn_dual_mode():
    reg, _ = reg_impls()
    kind, calls = parse_chat_turn(
        render_tool_call([ToolCall("calc", {"expr": "1+1"})]), reg)
    assert kind == "tool" and calls[0].name == "calc"
    kind, text = parse_chat_turn("The magic word is kiwi.", reg)
    assert kind == "text" and text == "The magic word is kiwi."
    print("PASS test_parse_chat_turn_dual_mode")


def test_parse_chat_turn_never_swallows_markers():
    reg, _ = reg_impls()
    with raises(ToolCallError):
        parse_chat_turn("", reg)                      # empty
    with raises(ToolCallError):
        parse_chat_turn("@@tool@@ not json @@end@@", reg)   # bad block
    with raises(ToolCallError):
        parse_chat_turn("trailing @@end@@", reg)      # stray marker
    with raises(ToolCallError):
        parse_chat_turn(None, reg)                    # non-string
    print("PASS test_parse_chat_turn_never_swallows_markers")


# --- session: tool task -------------------------------------------------------

def test_chat_session_tool_task():
    rng = random.Random(11)
    env = make_envs()[0]  # CalcEnv
    task = env.sample(rng)
    s = make_session(scripted_chat_policy, reg_impls())
    final = s.send(task.text, seed=1)
    assert final is not None and env.verify(task, final), \
        f"task={task.text!r} final={final!r}"
    roles = [t.role for t in s.turns]
    assert roles[0] == ROLE_USER
    assert roles[-1] == ROLE_TOOL and "FINISH: " in s.turns[-1].content
    # every assistant tool turn is followed by a tool observation
    for i, r in enumerate(roles[:-1]):
        if r == ROLE_ASSISTANT and i > 0:
            assert roles[i + 1] == ROLE_TOOL
    print(f"PASS test_chat_session_tool_task (final={final!r})")


def test_chat_session_compose_task():
    rng = random.Random(13)
    env = make_envs()[2]  # ComposeEnv
    task = env.sample(rng)
    s = make_session(scripted_chat_policy, reg_impls())
    final = s.send(task.text, seed=2)
    assert final is not None and env.verify(task, final), \
        f"task={task.text!r} final={final!r}"
    print(f"PASS test_chat_session_compose_task (final={final!r})")


# --- session: text reply bypasses the gate ------------------------------------

class ExplodingGate(Gate):
    def decide(self, call, context):
        raise AssertionError("gate must not see text replies")

    def escalate(self, call, context):
        raise AssertionError("unreachable")


def test_text_reply_bypasses_gate():
    s = make_session(lambda p, seed, i: "It is 42, trust me.",
                     reg_impls(), gate=ExplodingGate())
    out = s.send("How many roads must a man walk down?", seed=3)
    assert out == "It is 42, trust me."
    assert [t.role for t in s.turns] == [ROLE_USER, ROLE_ASSISTANT]
    assert s.steps[-1].kind == "text" and s.steps[-1].observation == ""
    print("PASS test_text_reply_bypasses_gate")


# --- session: recovery --------------------------------------------------------

def test_parse_error_recovery():
    calls = {"n": 0}

    def flaky(prompt, seed, i):
        # n=1: malformed block; n=2: retry the real calc; n=3: finish
        calls["n"] += 1
        if calls["n"] == 1:
            return "junk @@tool@@ not json @@end@@"
        if calls["n"] == 2:
            expr = task.text.split(": ", 1)[1]
            return render_tool_call([ToolCall("calc", {"expr": expr})])
        return scripted_chat_policy(prompt, seed, 0)

    rng = random.Random(17)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(flaky, reg_impls())
    final = s.send(task.text, seed=4)
    assert final is not None and env.verify(task, final)
    assert any(st.kind == "parse_error"
               and st.observation.startswith("PARSE_ERROR: ") for st in s.steps)
    print("PASS test_parse_error_recovery")


def test_tool_error_recovery():
    calls = {"n": 0}

    def divzero_then_fixed(prompt, seed, i):
        # n=1: tool error (1/0); n=2: retry real calc; n=3: scripted finish
        calls["n"] += 1
        if calls["n"] == 1:
            return render_tool_call([ToolCall("calc", {"expr": "1/0"})])
        if calls["n"] == 2:
            expr = task.text.split(": ", 1)[1]
            return render_tool_call([ToolCall("calc", {"expr": expr})])
        return scripted_chat_policy(prompt, seed, 0)

    rng = random.Random(19)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(divzero_then_fixed, reg_impls())
    final = s.send(task.text, seed=5)
    assert final is not None and env.verify(task, final)
    assert any(st.observation.startswith("TOOL_ERROR: calc: division by zero")
               for st in s.steps)
    print("PASS test_tool_error_recovery")


# --- session: follow-up memory ------------------------------------------------

def test_followup_uses_transcript_memory():
    rng = random.Random(23)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(scripted_chat_policy, reg_impls())
    first = s.send(task.text, seed=6)
    assert first is not None and env.verify(task, first)
    second = s.send(FOLLOWUP_TEXT, seed=6)
    assert second == repr(eval(f"({task.answer}) * 2")), \
        f"expected 2x {task.answer!r}, got {second!r}"
    print(f"PASS test_followup_uses_transcript_memory (2x={second!r})")


# --- replay -------------------------------------------------------------------

def test_chat_replay_roundtrip():
    reg, impls = reg_impls()
    rng = random.Random(29)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(scripted_chat_policy, reg_impls())
    s.send(task.text, seed=7)
    s.send(FOLLOWUP_TEXT, seed=7)
    verify_chat_replay(s.steps, reg, impls)  # must not raise

    tampered = [s.steps[0].__class__(**{**s.steps[0].__dict__,
                                        "observation": "bogus"})]
    with raises(ChatReplayMismatch):
        verify_chat_replay(tampered + list(s.steps[1:]), reg, impls)
    print("PASS test_chat_replay_roundtrip")


# --- regression: pure tool pipeline untouched ---------------------------------

def test_v530_extension_regression():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_benchmark import test_benchmark_three_modes
    test_benchmark_three_modes()
    print("PASS test_v530_extension_regression (T17 re-run)")


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} v5.30 chat tests passed")
