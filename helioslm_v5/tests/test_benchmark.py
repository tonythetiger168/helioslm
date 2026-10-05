import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from benchmark import (make_env_oracle, run_modes, scripted_correct_policy,
                       export_score_stream)
from envs import make_envs


def test_benchmark_three_modes():
    rng = random.Random(3)
    tasks = [e.sample(rng) for e in make_envs() for _ in range(20)]
    verify = lambda t, a: a.strip() == t.answer.strip()
    res = run_modes(tasks, verify, scripted_correct_policy,
                    make_env_oracle(tasks),
                    confidence_fn=lambda p, s: 1.0)
    assert res["oracle"][0] == 1.0, f"oracle={res['oracle'][0]}"
    assert abs(res["direct"][0] - res["routed_tau0.5"][0]) < 1e-9, \
        f"direct={res['direct'][0]} routed={res['routed_tau0.5'][0]}"
    assert res["direct"][0] == 1.0, \
        "scripted policy must be perfect ??pipeline bug if not"
    export_score_stream(res["direct"][1], "/tmp/test_stream.jsonl")
    assert Path("/tmp/test_stream.jsonl").stat().st_size > 0


if __name__ == "__main__":
    test_benchmark_three_modes()
    print("PASS test_benchmark_three_modes\n\n1/1 tests passed")
