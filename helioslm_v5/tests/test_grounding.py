"""T33 - v5.34: deterministic grounding oracles.

The core oracle replays the mid failure: a model that CONFABULATES
arguments ("19 * -92" -> "12 * -9") scores 0 without grounding and
passes env.verify WITH GroundingGate -- the model keeps only the tool
sequence; policy in code fills both ends.
Run from repo root: python3 helioslm_v5/tests/test_grounding.py
"""
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from envs import make_envs, make_long_envs
from gate import FixedGate, Gate, Route
from grounding import GroundingGate
from loop import AgentLoop
from schema import ToolCall, render_tool_call
from tools import build_default_registry


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _confabulating_policy(prompt, seed, step):
    """Right tool SEQUENCE, universally WRONG content (mid's failure
    mode, extended to every family): grounding must fix both ends."""
    obs = re.findall(r"step \d+: (.*)", prompt)
    task_m = re.search(r"Task: (.*?)(?:\n|$)", prompt)
    task = task_m.group(1) if task_m else prompt

    def ph(call):
        return render_tool_call([call])

    # exprs contain no periods: anchor calc to a no-dot tail so the
    # file-env tasks (which ALSO start with "Compute the value of:")
    # fall through to their own branches (same greedy-first bug class
    # the grounding parser had)
    if re.fullmatch(r"Compute the value of: [-0-9+*/()% ]*", task):
        return ph(ToolCall("finish", {"answer": "2"})) if obs else \
            ph(ToolCall("calc", {"expr": "1+1"}))
    m = re.fullmatch(r'Apply (\w+)(?: (\d+) times)? to the string: "(.*)"', task)
    if m:
        return ph(ToolCall("finish", {"answer": "xx"})) if obs else \
            ph(ToolCall("str_op", {"s": "xx", "op": "upper", "n": 0}))
    if re.fullmatch(r"First compute: .*", task):
        n = len(obs)
        if n == 0:
            return ph(ToolCall("calc", {"expr": "1+1"}))
        if n == 1:
            return ph(ToolCall("str_op", {"s": "xx", "op": "upper", "n": 0}))
        return ph(ToolCall("finish", {"answer": "x"}))
    m = re.match(r"Compute the value of: (.*?)\. Write the result to (\S+?), then read", task)
    if m:
        n = len(obs)
        if n == 0:
            return ph(ToolCall("calc", {"expr": "1+1"}))
        if n == 1:
            return ph(ToolCall("file_write", {"path": "zz.txt", "content": "0"}))
        if n == 2:
            return ph(ToolCall("file_read", {"path": "zz.txt"}))
        return ph(ToolCall("finish", {"answer": "0"}))
    m = re.match(r'Write the string "(.*?)" to (\S+)\. Then apply (\w+)(?: (\d+) times)? to it and write the result to (\S+)\.', task, re.S)
    if m:
        a, b = m.group(2), m.group(5)
        n = len(obs)
        if n == 0:
            return ph(ToolCall("file_write", {"path": "zz.txt", "content": "xx"}))
        if n == 1:
            return ph(ToolCall("str_op", {"s": "xx", "op": "upper", "n": 0}))
        if n == 2:
            return ph(ToolCall("file_write", {"path": "zz2.txt", "content": "0"}))
        if n == 3:
            return ph(ToolCall("file_read", {"path": "zz2.txt"}))
        return ph(ToolCall("finish", {"answer": "0"}))
    m = re.fullmatch(r"Compute the value of: (.*?) and write the result to (\S+)\. Compute the value of: (.*?) and write the result to (\S+)\..*", task, re.S)
    if m:
        p1, p2 = m.group(2), m.group(4)
        n = len(obs)
        if n == 0:
            return ph(ToolCall("calc", {"expr": "1+1"}))
        if n == 1:
            return ph(ToolCall("calc", {"expr": "2+2"}))
        if n == 2:
            return ph(ToolCall("file_write", {"path": "zz.txt", "content": "0"}))
        if n == 3:
            return ph(ToolCall("file_write", {"path": "zz2.txt", "content": "0"}))
        if n == 4:
            return ph(ToolCall("file_read", {"path": "zz.txt"}))
        if n == 5:
            return ph(ToolCall("file_read", {"path": "zz2.txt"}))
        if n == 6:
            return ph(ToolCall("calc", {"expr": "1+1"}))
        return ph(ToolCall("finish", {"answer": "0"}))
    return ph(ToolCall("finish", {"answer": "?"}))


def _grounded_run(task, root):
    reg, impls = build_default_registry(root)
    gate = GroundingGate(FixedGate(Route.DIRECT))
    loop = AgentLoop(_confabulating_policy, reg, impls, gate,
                     max_steps=task.step_budget)
    traj = loop.run(task.text, seed=1)
    return traj, gate


def test_grounding_cures_confabulation():
    rng = random.Random(20260929)
    env = make_envs()[0]
    n_ok = 0
    for _ in range(6):
        task = env.sample(rng)
        with __import__("tempfile").TemporaryDirectory() as root:
            traj, _ = _grounded_run(task, root)
        assert traj.final_answer is not None, "no finish"
        if env.verify(task, traj.final_answer):
            n_ok += 1
    assert n_ok == 6, f"grounding failed on {6 - n_ok}/6 calc tasks"
    print(f"PASS test_grounding_cures_confabulation ({n_ok}/6, "
          f"policy had wrong args on every call)")


def test_grounding_all_env_families():
    rng = random.Random(11)
    total, ok = 0, 0
    for env in make_envs() + make_long_envs():
        for _ in range(6):
            task = env.sample(rng)
            with __import__("tempfile").TemporaryDirectory() as root:
                traj, gate = _grounded_run(task, root)
            total += 1
            if traj.final_answer is not None and env.verify(task, traj.final_answer):
                ok += 1
    assert ok == total, f"{ok}/{total}"
    print(f"PASS test_grounding_all_env_families ({ok}/{total})")


def test_grounding_is_deterministic():
    rng = random.Random(23)
    env = make_envs()[2]  # ComposeEnv
    task = env.sample(rng)
    finals = set()
    for _ in range(3):
        with __import__("tempfile").TemporaryDirectory() as root:
            traj, _ = _grounded_run(task, root)
        finals.add(traj.final_answer)
    assert len(finals) == 1, f"nondeterministic: {finals}"
    print(f"PASS test_grounding_is_deterministic (final={finals.pop()!r})")


def test_escalate_path_untouched():
    class ExplodingOracle(Gate):
        def decide(self, call, context):
            return Route.ESCALATE

        def escalate(self, call, context):
            return "ORACLE_OBS"

    reg, impls = build_default_registry("/tmp/helioslm_t33")
    gate = GroundingGate(ExplodingOracle())
    task = make_envs()[0].sample(random.Random(5))
    # a finish call with confabulated args must reach the oracle UNCHANGED
    call = ToolCall("finish", {"answer": "garbage"})
    route = gate.decide(call, {"task": task.text})
    assert route == Route.ESCALATE
    assert call.args["answer"] == "garbage", "escalate path was modified!"
    print("PASS test_escalate_path_untouched")


def test_unknown_task_passes_through():
    gate = GroundingGate(FixedGate(Route.DIRECT))
    call = ToolCall("calc", {"expr": "garbage-expr"})
    route = gate.decide(call, {"task": "totally unknown task shape"})
    assert route == Route.DIRECT
    assert call.args["expr"] == "garbage-expr", "unknown shape must pass through"
    assert not gate.groundable("totally unknown task shape")
    print("PASS test_unknown_task_passes_through")


if __name__ == "__main__":
    test_grounding_cures_confabulation()
    test_grounding_all_env_families()
    test_grounding_is_deterministic()
    test_escalate_path_untouched()
    test_unknown_task_passes_through()
    print("\n5/5 grounding tests passed")
