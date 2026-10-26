"""T39 - v5.37: grounding-aware replay oracles.

The recorded grounded trajectory re-verifies end to end: raw text ->
parse -> FRESH GroundingGate reproduces the recorded observations
exactly (tools are deterministic; the grammar is shared). Tampered
observations are still caught -- grounded runs are auditable like any
other (the v5.34 docstring deviation is now closed).
Run from repo root: python3 helioslm_v5/tests/test_grounded_replay.py
"""
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from envs import make_envs
from gate import FixedGate, Route
from grounding import GroundingGate, verify_grounded_replay
from loop import AgentLoop
from schema import ToolCall, render_tool_call
from tools import build_default_registry
from trajectory import ReplayMismatch, Trajectory


def _policy(prompt, seed, step):
    obs = __import__("re").findall(r"step \d+: (.*)", prompt)
    ph = lambda c: render_tool_call([c])
    if not obs:
        return ph(ToolCall("calc", {"expr": "1+1"}))   # wrong args on
    return ph(ToolCall("finish", {"answer": "0"}))      # purpose


def test_grounded_replay_roundtrip():
    rng = random.Random(7)
    env = make_envs()[0]
    n = 0
    for _ in range(6):
        task = env.sample(rng)
        with tempfile.TemporaryDirectory() as root:
            reg, impls = build_default_registry(root)
            gate = GroundingGate(FixedGate(Route.DIRECT))
            loop = AgentLoop(_policy, reg, impls, gate,
                             max_steps=task.step_budget)
            traj = loop.run(task.text, seed=1)
            verify_grounded_replay(traj, reg, impls)   # must not raise
            assert env.verify(task, traj.final_answer), \
                "grounding should have made it correct"
        n += 1
    print(f"PASS test_grounded_replay_roundtrip ({n} trajectories)")


def test_grounded_replay_catches_tamper():
    rng = random.Random(9)
    env = make_envs()[0]
    task = env.sample(rng)
    with tempfile.TemporaryDirectory() as root:
        reg, impls = build_default_registry(root)
        gate = GroundingGate(FixedGate(Route.DIRECT))
        loop = AgentLoop(_policy, reg, impls, gate, max_steps=task.step_budget)
        traj = loop.run(task.text, seed=1)
        bad = traj.steps[0].__class__(**{**traj.steps[0].__dict__,
                                         "observation": "bogus"})
        tampered = Trajectory(task=traj.task, seed=traj.seed,
                              steps=[bad] + list(traj.steps[1:]),
                              final_answer=traj.final_answer)
        try:
            verify_grounded_replay(tampered, reg, impls)
            raise SystemExit("tampered grounded trajectory verified")
        except ReplayMismatch:
            pass
    print("PASS test_grounded_replay_catches_tamper")


if __name__ == "__main__":
    test_grounded_replay_roundtrip()
    test_grounded_replay_catches_tamper()
    print("\n2/2 grounded-replay tests passed")
