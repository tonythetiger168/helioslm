"""T28 - v5.32: non-autoregressive DecisionHead oracles.

Covers: multi-question single-pass answering, learnability on separable
synthetic data, the Brier calibration term actually binding (ECE small
after fit), and batch-vs-single consistency.
Run from repo root: python3 helioslm_v5/tests/test_decision_head.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision import NOUL_NO, NOUL_YES
from decision_head import DecisionHead
from trajectory import text_to_ids


def _mk_records(n=200, seed=3):
    """Two synthetic states: state A -> route DIRECT + safe; state B ->
    route ESCALATE + irreversible. Char-level separable by construction
    (distinct keyword chars) so the honest minimum encoder can learn it."""
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        if i % 2 == 0:
            text = f"simple read only task number {i} check"
            recs.append({"ids": torch.tensor(text_to_ids(text)),
                         "answers": {"route": "DIRECT",
                                     "is_irreversible": NOUL_NO,
                                     "severity": 0.1}})
        else:
            text = f"DANGER delete wipe irreversible op {i} xyz"
            recs.append({"ids": torch.tensor(text_to_ids(text)),
                         "answers": {"route": "ESCALATE",
                                     "is_irreversible": NOUL_YES,
                                     "severity": 0.9}})
    return recs


def test_questions_single_pass():
    head = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE")),
                         "is_irreversible": ("noul", None),
                         "severity": ("score", None)})
    ids = torch.tensor(text_to_ids("DANGER delete wipe irreversible op"))
    out = head.forward(ids)
    assert set(out) == {"route", "is_irreversible", "severity"}
    ans, conf = out["route"]
    assert ans in ("DIRECT", "ESCALATE") and 0 <= conf <= 1
    assert out["is_irreversible"][0] in (NOUL_YES, NOUL_NO, "unknown")
    assert 0.0 <= out["severity"][0] <= 1.0
    print("PASS test_questions_single_pass")


def test_learns_separable_states():
    torch.manual_seed(5)
    head = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE")),
                         "is_irreversible": ("noul", None),
                         "severity": ("score", None)})
    recs = _mk_records()
    head.fit(recs, epochs=300, lr=5e-2)
    acc, ece = head.evaluate(recs)
    assert acc["route"] > 0.95, acc
    assert acc["is_irreversible"] > 0.95, acc
    assert ece["route"] < 0.15, ece
    # NOTE: on perfectly separable data acc==1.0, so Brier's target IS
    # 1.0 — confidence SHOULD be ~1.0 here. The Brier-binding oracle needs
    # noisy data (see test_brier_term_binds).
    print(f"PASS test_learns_separable_states acc={acc} ece={ece}")


def test_brier_term_binds():
    """Oracle for the OUTCOME-targeted Brier term (RLCD setting).

    FINDING (A/B, 2026-09-28): with target == train label correctness the
    Brier term is redundant with CE (both proper scoring rules -> same
    posterior confidences; observed identical to 3 decimals). The term
    bites only when records carry an explicit replay-verified
    `target_conf` that the softmax cannot memorize: here all records
    assert the head was right only 70% of the time on equivalent states,
    and confidence must move TOWARD 0.70 while accuracy stays perfect on
    the (separable) labels."""
    torch.manual_seed(5)
    # lambda must overcome CE's margin growth on separable data;
    # equilibrium conf scales with lambda (recorded, not hidden)
    head = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE"))},
                        brier_lambda=10.0)
    recs = _mk_records(n=160, seed=11)
    for r in recs:
        r["answers"]["route__target_conf"] = 0.70
    head.fit(recs, epochs=600, lr=5e-2)

    head.eval()
    with torch.no_grad():
        zs, _ = head._targets(recs)
        conf = torch.softmax(head.heads["route"](zs), dim=-1) \
            .max(dim=-1).values.mean()
    assert 0.55 < float(conf) < 0.80, \
        f"confidence did not track the outcome target 0.70: {float(conf):.3f}"
    acc, _ = head.evaluate(recs)
    assert acc["route"] > 0.95, "accuracy sacrificed — term must not break CE"
    print(f"PASS test_brier_term_binds (conf={float(conf):.3f} "
          f"target=0.70 acc={acc['route']:.3f})")


def test_batch_equals_single():
    torch.manual_seed(7)
    head = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE")),
                         "is_irreversible": ("noul", None)})
    head.fit(_mk_records(n=60, seed=9), epochs=150, lr=5e-2)
    head.eval()
    ids = torch.tensor(text_to_ids("simple read only task check"))
    both = head.forward(ids)
    one = head.decide("route", ids)
    assert one == both["route"]
    print("PASS test_batch_equals_single")


if __name__ == "__main__":
    test_questions_single_pass()
    test_learns_separable_states()
    test_brier_term_binds()
    test_batch_equals_single()
    print("\n4/4 decision-head tests passed")
