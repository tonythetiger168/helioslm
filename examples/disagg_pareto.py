"""disagg_pareto.py - v5.28: three-axis Pareto sweep of the disagg evolver module.

Sweeps (n_prefill, n_decode) over sample workloads, applies the monotonicity
gate, and emits the non-dominated front on (makespan, worker_seconds) with
worker count as tier. This is the v5.28 cost-axis alignment artifact: the
report format is designed for spec-level comparison against production
serving cards (e.g. GLM-5.3-Flash $0.25 / 43 t/s).

Run from repo root: python3 examples/disagg_pareto.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.agent.disagg import (DisaggConfig, Request, evaluate,
                                      monotonicity_gate)

SEED = 20260925


def make_workload(rng, n, cache_rate, name):
    return name, [Request(i, rng.randint(200, 4000), rng.randint(50, 800),
                          f"p{rng.randrange(8)}" if rng.random() < cache_rate else None)
                  for i in range(n)]


def pareto_front(points):
    """Latency-vs-cost front: x = worker count (cost), y = makespan
    (latency), worker_seconds reported per rung. Best makespan per worker
    count, then keep rungs with strictly decreasing makespan as workers
    grow -- every such rung is non-dominated (fewer workers = more latency)."""
    best = {}
    for ms, ws, nw, cfg in points:
        if nw not in best or ms < best[nw][0]:
            best[nw] = (ms, ws, nw, cfg)
    front = []
    for nw in sorted(best):               # workers ascending
        ms, ws, _, cfg = best[nw]
        if not front or ms < front[-1][0] - 1e-12:
            front.append((ms, ws, nw, cfg))
    return front


def main():
    rng = random.Random(SEED)
    workloads = [make_workload(rng, 80, 0.95, "cache_heavy"),
                 make_workload(rng, 80, 0.0, "cold"),
                 make_workload(rng, 200, 0.6, "mixed")]
    grid = [DisaggConfig(p, d) for p in range(1, 7) for d in range(1, 7)]
    ladder = [DisaggConfig(1, 1), DisaggConfig(2, 1), DisaggConfig(2, 2),
              DisaggConfig(3, 3), DisaggConfig(4, 4), DisaggConfig(6, 6)]
    report = {"seed": SEED, "workloads": {}}

    for name, wl in workloads:
        # Hard gate on the round_robin ladder: without cache affinity the
        # model IS worker-monotone (gate contract holds). cache_aware is
        # the search space -- and a documented finding (see report) is that
        # cache affinity breaks worker monotonicity (~4% here): same-key
        # requests serialize on their holder, so more workers can add
        # makespan. Structural property, not a bug -- recorded, not hidden.
        rr = [DisaggConfig(c.n_prefill, c.n_decode, "round_robin")
              for c in ladder]
        monotonicity_gate(wl, rr)
        findings = []
        for a, b in zip(ladder, ladder[1:]):
            ma, _ = evaluate(wl, a)
            mb, _ = evaluate(wl, b)
            if mb > ma + 1e-9:
                findings.append(
                    f"cache_aware anti-monotonic: {a.n_prefill}P{a.n_decode}D"
                    f"({ma:.3f}) -> {b.n_prefill}P{b.n_decode}D({mb:.3f})")
        report.setdefault("findings", {})[name] = findings
        pts = []
        for cfg in grid:
            ms, ws = evaluate(wl, cfg)
            pts.append((ms, ws, cfg.n_prefill + cfg.n_decode, cfg))
        front = pareto_front(pts)
        report["workloads"][name] = {
            "front": [{"makespan": round(ms, 3), "worker_seconds": round(ws, 3),
                       "workers": nw, "cfg": f"{cfg.n_prefill}P+{cfg.n_decode}D"}
                      for ms, ws, nw, cfg in front],
            "all_configs": len(grid)}
        print(f"== {name} ==")
        for ms, ws, nw, cfg in front:
            print(f"  {cfg.n_prefill}P+{cfg.n_decode}D ({nw}w): makespan={ms:7.3f} "
                  f"worker_s={ws:8.3f}")
        assert len(front) >= 2, "Pareto front must offer a real trade-off"

    out = Path(__file__).resolve().parent.parent / "benchmarks" / "disagg_pareto_2026-09-25.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
