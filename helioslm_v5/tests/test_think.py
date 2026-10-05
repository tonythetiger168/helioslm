"""T25 - v5.31: think mode + experience reuse oracles.

Covers: tri-mode parsing (never swallow markers), think flow through
ThinkSession (no gate contact), experience store remember/recall with
failure skipping, cross-turn reuse on a repeated task, and replay
compatibility with verify_chat_replay.
Run from repo root: python3 helioslm_v5/tests/test_think.py
"""
import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from chat import scripted_chat_policy, verify_chat_replay
from envs import make_envs
from gate import Gate, FixedGate, Route
from schema import ToolCall, render_tool_call
from think import (ExperienceStore, ThinkParseError, ThinkSession,
                   parse_tri_mode, verify_think_replay)
from tools import build_default_registry


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def reg_impls():
    return build_default_registry("/tmp/helioslm_think_test")


def make_session(model_fn, reg_impls_, gate=None, store=None, max_steps=8):
    reg, impls = reg_impls_
    return ThinkSession(model_fn, reg, impls,
                        gate or FixedGate(Route.DIRECT), max_steps=max_steps,
                        store=store)


# --- protocol ---------------------------------------------------------------

def test_parse_tri_mode():
    reg, _ = reg_impls()
    kind, content = parse_tri_mode("@@think@@plan: use calc@@end@@", reg)
    assert kind == "think" and content == "plan: use calc"
    kind, calls = parse_tri_mode(
        render_tool_call([ToolCall("calc", {"expr": "1"})]), reg)
    assert kind == "tool"
    kind, text = parse_tri_mode("plain answer", reg)
    assert kind == "text" and text == "plain answer"
    with raises(ThinkParseError):
        parse_tri_mode("", reg)
    with raises(ThinkParseError):
        parse_tri_mode("@@think@@unclosed", reg)
    with raises(ThinkParseError):
        parse_tri_mode("@@think@@a@@end@@ @@think@@b@@end@@", reg)
    print("PASS test_parse_tri_mode")


# --- session ------------------------------------------------------------------

class ExplodingGate(Gate):
    def decide(self, call, context):
        raise AssertionError("think must not reach the gate")

    def escalate(self, call, context):
        raise AssertionError("unreachable")


def scripted_think_policy(prompt, seed, step):
    # no assistant/tool turn yet -> open with a thought
    if "##assistant##" not in prompt and "##tool##" not in prompt:
        return "@@think@@plan: compute then finish@@end@@"
    return scripted_chat_policy(prompt, seed, step)


def test_think_session_flow():
    rng = random.Random(11)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(scripted_think_policy, reg_impls())  # FixedGate DIRECT
    final = s.send(task.text, seed=1)
    assert final is not None and env.verify(task, final)
    kinds = [st.kind for st in s.steps]
    assert kinds[0] == "think", kinds
    # think commits to nothing: recorded with route "-" and empty obs
    assert s.steps[0].route == "-" and s.steps[0].observation == ""
    assert "think" in s.turns[1].content  # thought visible in transcript
    print(f"PASS test_think_session_flow (kinds={kinds})")


def test_experience_store_recall():
    store = ExperienceStore()
    store.remember("Compute 1+1", "think A", "2")
    store.remember("Compute 1+1", "think B", "FAIL")
    store.remember("compute   1+1", "think C", "2")  # normalization check
    assert store.recall("Compute 1+1") == "think C"
    assert store.recall("never seen") is None
    store2 = ExperienceStore()
    store2.remember("t", "only-fail", "FAIL")
    assert store2.recall("t") is None, "failures must not be recalled"
    print("PASS test_experience_store_recall")


def test_experience_reuse_across_turns():
    rng = random.Random(23)
    env = make_envs()[0]
    task = env.sample(rng)
    store = ExperienceStore()
    s = make_session(scripted_think_policy, reg_impls(), store=store)
    s.send(task.text, seed=1)
    recalled = store.recall(task.text)
    assert recalled == "plan: compute then finish"
    # same session, repeated task: the thought is now recallable; a policy
    # that consults the store can skip re-deriving it (documented reuse)
    reg, impls = reg_impls()
    s2 = make_session(scripted_think_policy, reg_impls(), store=store)
    final2 = s2.send(task.text, seed=1)
    assert final2 is not None and env.verify(task, final2)
    print("PASS test_experience_reuse_across_turns")


def test_replay_compatible():
    reg, impls = reg_impls()
    rng = random.Random(29)
    env = make_envs()[0]
    task = env.sample(rng)
    s = make_session(scripted_think_policy, reg_impls())
    s.send(task.text, seed=1)
    verify_think_replay(s.steps, reg, impls)  # tri-mode parser required
    print("PASS test_replay_compatible")


if __name__ == "__main__":
    test_parse_tri_mode()
    test_think_session_flow()
    test_experience_store_recall()
    test_experience_reuse_across_turns()
    test_replay_compatible()
    print("\n5/5 think tests passed")
