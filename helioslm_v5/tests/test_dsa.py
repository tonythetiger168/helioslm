import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from dsa import dense_decode, dsa_gate, exact_topk, sparse_decode


def test_t11_dsa_two_layer_oracle():
    rng = random.Random(11)
    for _ in range(500):
        n, d = rng.randint(8, 512), rng.randint(1, 16)
        scores = [rng.gauss(0, 1) for _ in range(n)]
        for _ in range(4):
            scores[rng.randrange(n)] += 8.0
        V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        k = rng.randint(1, n)
        out_s, cert = sparse_decode(scores, V, k)
        if cert:
            dsa_gate(out_s, dense_decode(scores, V), cert, k, n)

    n, k = 64, 1
    scores = [10.0] + [9.999] * (n - 1)
    V = [[1.0] * 4] + [[0.0] * 4 for _ in range(n - 1)]
    _, cert = sparse_decode(scores, V, k)
    assert not cert, "certificate issued on a pseudo-max gap ??oracle broken"

    scores = [1.0, 2.0, 1.0, 2.0, 1.0]
    assert exact_topk(scores, 3) == exact_topk(scores, 3) == [1, 3, 0]

    V5 = [[float(kk)] for kk in range(5)]
    out_s, cert = sparse_decode(scores, V5, 5)
    assert cert and out_s == dense_decode(scores, V5)


if __name__ == "__main__":
    test_t11_dsa_two_layer_oracle()
    print("PASS test_t11_dsa_two_layer_oracle\n\n1/1 tests passed")
