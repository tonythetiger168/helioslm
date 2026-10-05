"""T36 - v5.35c: copy curriculum oracles.

Covers: echo env sample/verify, the text-channel copy flow through a
real ChatSession (the v5.30 direct-reply path), dataset builder output
(copy-exact replies, ord discipline, chat mixing), and determinism.
Run from repo root: python3 helioslm_v5/tests/test_copy_curriculum.py
"""
import json
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from chat import ChatSession
from copy_curriculum import (EchoEnv, build_copy_dataset,
                             gen_echo_episode, make_copy_envs,
                             scripted_echo_policy)
from gate import FixedGate, Route
from tools import build_default_registry


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_echo_env_verify():
    rng = random.Random(3)
    env = EchoEnv()
    for _ in range(50):
        task = env.sample(rng)
        assert task.family == "echo" and task.step_budget == 1
        assert task.text == f'Repeat back exactly: "{task.answer}"'
        assert task.answer, "empty echo string"
        assert env.verify(task, task.answer)
        assert not env.verify(task, task.answer + "x")
    print("PASS test_echo_env_verify")


def test_echo_through_chat_session():
    """The copy flows through the v5.30 text channel: ChatSession, no
    tool calls, one turn, exact string back."""
    rng = random.Random(5)
    env = EchoEnv()
    reg, impls = build_default_registry("/tmp/helioslm_t36")
    ok = 0
    for _ in range(10):
        task = env.sample(rng)
        s = ChatSession(scripted_echo_policy, reg, impls,
                        FixedGate(Route.DIRECT), max_steps=1)
        final = s.send(task.text, seed=1)
        if final is not None and env.verify(task, final):
            ok += 1
        assert all(st.kind == "text" for st in s.steps), \
            "echo must never touch tools"
    assert ok == 10, f"{ok}/10"
    print(f"PASS test_echo_through_chat_session ({ok}/10, text-only)")


def test_copy_dataset_builder():
    with tempfile.TemporaryDirectory() as d:
        chat = Path(d) / "chat.jsonl"
        chat.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
        out = Path(d) / "copy.jsonl"
        build_copy_dataset(str(out), n_echo=40, seed=9, chat_path=str(chat))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
    echo_lines = [l for l in lines if 'Repeat back exactly:' in l["prompt"]]
    assert len(lines) == 41 and len(echo_lines) == 40
    for l in echo_lines:
        want = l["prompt"].split('Repeat back exactly: "', 1)[1].rstrip('"')
        assert l["response"] == want, "echo response is not an exact copy"
        assert "@@tool@@" not in l["response"]
    print(f"PASS test_copy_dataset_builder (40 echo + 1 chat, "
          f"responses copy-exact)")


def test_echo_episode_deterministic():
    env = EchoEnv()
    a = gen_echo_episode(env, random.Random(11))
    b = gen_echo_episode(env, random.Random(11))
    assert a == b
    assert make_copy_envs()[0].__class__ is EchoEnv
    print("PASS test_echo_episode_deterministic")


if __name__ == "__main__":
    test_echo_env_verify()
    test_echo_through_chat_session()
    test_copy_dataset_builder()
    test_echo_episode_deterministic()
    print("\n4/4 copy-curriculum tests passed")
