import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from disagg import (DisaggConfig, Request, evaluate, monotonicity_gate,
                    workload_fingerprint)


def _mk_workload(rng, n, cache_rate, shared_prefixes=8):
    return [Request(i, rng.randint(200, 4000), rng.randint(50, 800),
                    f"p{rng.randrange(shared_prefixes)}"
                    if rng.random() < cache_rate else None)
            for i in range(n)]


def test_t13_disagg_module():
    rng = random.Random(13)
    wl = _mk_workload(rng, 50, 0.9)
    cfg = DisaggConfig(2, 2)
    assert evaluate(wl, cfg) == evaluate(wl, cfg)
    assert evaluate(wl, cfg) == evaluate(list(wl), cfg)
    assert len(workload_fingerprint(wl)) == 16
    heavy = _mk_workload(rng, 80, 0.95)
    hit = evaluate(heavy, DisaggConfig(2, 2, "cache_aware"))[0]
    rr = evaluate(heavy, DisaggConfig(2, 2, "round_robin"))[0]
    assert hit <= rr + 1e-9
    assert hit < rr * 0.9, f"cache_aware no benefit: hit={hit:.3f} rr={rr:.3f}"
    cold = _mk_workload(rng, 60, 0.0)
    grid = [DisaggConfig(1, 1), DisaggConfig(2, 1),
            DisaggConfig(2, 2), DisaggConfig(4, 2)]
    monotonicity_gate(cold, grid)
    monotonicity_gate(heavy, grid)
    results = {c: evaluate(heavy, c) + evaluate(cold, c) for c in grid}
    lat = {c: v[0] + v[2] for c, v in results.items()}
    cost = {c: v[1] + v[3] for c, v in results.items()}
    assert min(lat, key=lat.get) != min(cost, key=cost.get) \
        or len(set(lat.values())) == 1, \
        "one config dominates everywhere ??evolver has nothing to search"


if __name__ == "__main__":
    test_t13_disagg_module()
    print("PASS test_t13_disagg_module\n\n1/1 tests passed")
