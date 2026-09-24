import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from attn_res import (AttnResMixer, determinism_gate, migration_gate,
                      run_stack)


def _toy_layers(rng, n, dim):
    ws = [[[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(dim)]
          for _ in range(n)]
    bs = [[rng.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(n)]
    return [lambda x, W=W, b=b: [sum(w * v for w, v in zip(row, x)) + bb
                                 for row, bb in zip(W, b)]
            for W, b in zip(ws, bs)]


def test_t12_attnres_oracles():
    rng = random.Random(12)
    dim, n = 8, 6
    fns = _toy_layers(rng, n, dim)
    x0 = [rng.uniform(-1, 1) for _ in range(dim)]
    migration_gate(fns, x0, n)
    mixer = AttnResMixer(n)
    mixer.set(2, [0.3, -0.1])
    mixer.set(4, [0.0, 0.5, 0.25, 0.0])
    determinism_gate(fns, x0, mixer)
    assert run_stack(fns, x0, mixer) != run_stack(fns, x0)
    m2 = AttnResMixer(2)
    m2.set(1, [1.0])
    const = [lambda x: [7.0] * dim, lambda x: [7.0] * dim]
    out = run_stack(const, [0.0] * dim, m2)
    assert all(v == 21.0 for v in out), out


if __name__ == "__main__":
    test_t12_attnres_oracles()
    print("PASS test_t12_attnres_oracles\n\n1/1 tests passed")
