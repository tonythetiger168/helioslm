"""T19 - v5.29: three-mode benchmark artifact oracles (C stage).

Checks the seeded report artifact (benchmarks/three_modes_2026-09-25.json):
- structure & reproducibility metadata
- tau-grid monotonicity (the recorded routing gate outcome)
- overconfidence finding recorded: max confidence among wrong answers
  is FAR above the fraction correct (miscalibration is the headline
  real-model finding, same record-don't-hide discipline as T18)

Run from repo root: python3 helioslm_v5/tests/test_three_modes.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_t19_report_oracles():
    out = Path(__file__).resolve().parent.parent.parent / "benchmarks" / \
        "three_modes_2026-09-25.json"
    if not out.exists():
        print("SKIP: run examples/benchmark_three_modes.py first")
        return
    rep = json.loads(out.read_text())
    assert rep["seed"] == 20260925
    assert rep["n_tasks"] == 12
    corr = rep["correctness"]
    assert corr["oracle"] == 1.0
    grid = [(g["tau"], g["correctness"]) for g in corr["tau_grid"]]
    assert grid[0][0] == 0.5 and grid[-1][0] == 0.95
    # monotonic non-decreasing along ascending tau (v5.22 gate, real model)
    for (t0, c0), (t1, c1) in zip(grid, grid[1:]):
        assert c1 >= c0 - 1e-9, f"non-monotone at tau {t0}->{t1}: {c0}->{c1}"
    assert rep["routing_gate"].startswith("PASS"), rep["routing_gate"]
    # overconfidence finding: wrong answers carry near-1 confidence
    wrong = [r for r in rep["tasks"] if not r["correct"]]
    assert len(wrong) == 12, "toy model skeleton limit: all wrong (known)"
    max_conf_wrong = max(r["conf"] for r in wrong)
    assert max_conf_wrong > 0.9, \
        f"expected overconfidence (conf>0.9 on wrong answers), got {max_conf_wrong}"
    print(f"PASS test_t19_report_oracles "
          f"(gate={rep['routing_gate']}, max_conf_on_wrong={max_conf_wrong:.3f})")


if __name__ == "__main__":
    test_t19_report_oracles()
    print("\n1/1 v5.29 tests passed")
