"""T37 - v5.36: continuous Noul oracles.

The ternary Noul had a hard floor at 1/K on binary Choice confidence;
the official TypeSafe form (P(yes) in [0,1], no independent confidence)
expresses sub-50% uncertainty by construction. Oracles: type semantics,
training convergence + outcome-Brier on the probability itself, and
the FLOOR TEST (a low-confidence class the ternary form could never
represent).
Run from repo root: python3 helioslm_v5/tests/test_noul_continuous.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision import NOUL_NO, NOUL_UNKNOWN, NOUL_YES, Noul, DecisionError
from decision_head import DecisionHead
from trajectory import text_to_ids


def test_type_semantics():
    n = Noul("q", 0.8)
    assert n.answer == NOUL_YES and abs(n.confidence - 0.6) < 1e-9
    n = Noul("q", 0.2)
    assert n.answer == NOUL_NO and abs(n.confidence - 0.6) < 1e-9
    n = Noul("q", 0.5)
    assert n.answer == NOUL_UNKNOWN and n.confidence == 0.0
    try:
        Noul("q", 1.5)
        raise SystemExit("out-of-range accepted")
    except DecisionError:
        pass
    print("PASS test_type_semantics")


def _recs(n=40, seed=5, risky_p=0.2):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        ok = rng.random() < 0.5
        m = "ALPHA" if ok else "OMEGA"
        recs.append({"ids": torch.tensor(text_to_ids(f"{m} variant {i}")),
                     "answers": {"trust": "yes" if ok else "no",
                                 "trust__target_conf": 1.0 if ok else risky_p}})
    return recs


def test_convergence_and_brier_on_probability():
    torch.manual_seed(3)
    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    recs = _recs()
    head.fit(recs, epochs=1500, lr=1e-1)
    acc, ece = head.evaluate(recs)
    assert acc["trust"] > 0.9, acc
    assert ece["trust"] < 0.25, ece
    head.eval()
    with torch.no_grad():
        pr = [head.decide("trust", torch.tensor(
            text_to_ids(f"ALPHA variant {i}")))[0] for i in range(6)]
        pw = [head.decide("trust", torch.tensor(
            text_to_ids(f"OMEGA variant {i}")))[0] for i in range(6)]
    mr = sum(pr) / len(pr)
    mw = sum(pw) / len(pw)
    assert mw < 0.5 < mr, (mr, mw)
    print(f"PASS test_convergence_and_brier (P(yes) right {mr:.2f} "
          f"wrong {mw:.2f}, ece {ece['trust']:.3f})")


def test_floor_sub_half_uncertainty():
    """THE floor oracle: a low-confidence class targeted at P=0.2. The
    ternary form could not express this; the continuous form must."""
    torch.manual_seed(5)
    recs = _recs(n=30, seed=9, risky_p=0.2)
    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    head.fit(recs, epochs=1500, lr=1e-1)
    head.eval()
    ps = {}
    with torch.no_grad():
        for m in ("ALPHA", "OMEGA"):
            ps[m], _ = head.decide("trust", torch.tensor(
                text_to_ids(f"{m} variant 0")))
    assert 0.0 <= ps["OMEGA"] < 0.5, f"floor violated: {ps}"
    print(f"PASS test_floor_sub_half_uncertainty "
          f"(P(yes) RISKY={ps['OMEGA']:.3f} < 0.5 -- ternary-impossible)")


if __name__ == "__main__":
    test_type_semantics()
    test_convergence_and_brier_on_probability()
    test_floor_sub_half_uncertainty()
    print("\n3/3 continuous-Noul tests passed")
