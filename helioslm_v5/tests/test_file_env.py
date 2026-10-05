"""T24 - v5.31: long-horizon file env oracles.

Covers: all three task families solved end-to-end by the scripted policy
through AgentLoop (the real loop, not a mock), budget discipline, replay
verification, and determinism.
Run from repo root: python3 helioslm_v5/tests/test_file_env.py
"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from envs import make_long_envs
from envs.file_env import FileLongHorizonEnv
from gate import FixedGate, Route
from loop import AgentLoop
from tools import build_default_registry
from trajectory import verify_replay


def _run(task, root):
    reg, impls = build_default_registry(root)
    from envs.file_env import scripted_file_policy
    loop = AgentLoop(scripted_file_policy, reg, impls,
                     FixedGate(Route.DIRECT), max_steps=task.step_budget)
    traj = loop.run(task.text, seed=1)
    return traj, reg, impls


def test_all_families_solved():
    rng = random.Random(41)
    env = FileLongHorizonEnv()
    seen = set()
    for _ in range(60):
        task = env.sample(rng)
        if task.family in seen:
            continue
        seen.add(task.family)
        with tempfile.TemporaryDirectory() as root:
            traj, reg, impls = _run(task, root)
            assert traj.final_answer is not None, \
                f"{task.family}: no final answer (steps={traj.n_steps})"
            assert env.verify(task, traj.final_answer), \
                f"{task.family}: wrong answer {traj.final_answer!r} != {task.answer!r}"
            assert len(traj.steps) <= task.step_budget, \
                f"{task.family}: budget exceeded {len(traj.steps)}>{task.step_budget}"
            verify_replay(traj, reg, impls)  # replay must not raise
        print(f"PASS family={task.family} steps={len(traj.steps)} "
              f"budget={task.step_budget}")
    assert seen == {"write_read", "write_transform", "accumulate"}, seen
    print("PASS test_all_families_solved")


def test_longer_budgets_than_toy():
    rng = random.Random(43)
    env = FileLongHorizonEnv()
    budgets = [env.sample(rng).step_budget for _ in range(30)]
    assert min(budgets) >= 8, f"expected long-horizon budgets, got {budgets}"
    print(f"PASS test_longer_budgets_than_toy (min={min(budgets)})")


def test_sampling_deterministic():
    t1 = FileLongHorizonEnv().sample(random.Random(99)).text
    t2 = FileLongHorizonEnv().sample(random.Random(99)).text
    assert t1 == t2, "same seed must reproduce the same task"
    print("PASS test_sampling_deterministic")


def test_replay_tamper_detected():
    rng = random.Random(47)
    env = FileLongHorizonEnv()
    task = next(t for t in (env.sample(rng) for _ in range(20))
                if t.family == "accumulate")
    with tempfile.TemporaryDirectory() as root:
        traj, reg, impls = _run(task, root)
        from trajectory import ReplayMismatch
        try:
            from trajectory import Trajectory
            bad = traj.steps[0].__class__(**{**traj.steps[0].__dict__,
                                             "observation": "bogus"})
            tampered = Trajectory(task=traj.task, seed=traj.seed,
                                  steps=[bad] + list(traj.steps[1:]),
                                  final_answer=traj.final_answer)
            verify_replay(tampered, reg, impls)
            raise SystemExit("tampered trajectory must not verify")
        except ReplayMismatch:
            pass
    print("PASS test_replay_tamper_detected")


if __name__ == "__main__":
    test_all_families_solved()
    test_longer_budgets_than_toy()
    test_sampling_deterministic()
    test_replay_tamper_detected()
    print("\n4/4 file-env tests passed")
