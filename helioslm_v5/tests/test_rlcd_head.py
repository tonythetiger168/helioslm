"""T35 - v5.35: RLCD head oracles on the mid failure profile.

Mirrors the real mid data shape: confidence pinned at ~0.9999 for BOTH
classes (the measured gap -- the claim carries no signal), outcomes
separable by a state marker. The head must recover the separation from
OUTCOMES alone (state-only input, no confidence features): conf on the
failure class drops while the model's own claim stays pinned.
Run from repo root: python3 helioslm_v5/tests/test_rlcd_head.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision_data import rlcd_reward_adjustment
from decision_head import DecisionHead
from trajectory import text_to_ids


def _records(n=40, seed=5):
    """even/odd i -> right/wrong classes; states marked textually."""
    recs = []
    for i in range(n):
        ok = i % 2 == 0
        marker = "ALPHA" if ok else "OMEGA"
        recs.append({"ids": torch.tensor(
            text_to_ids(f"{marker} compute task variant {i}")),
            "answers": {"trust": "yes" if ok else "no",
                        "trust__target_conf": 1.0 if ok else 0.0}})
    return recs, [r["answers"]["trust"] == "yes" for r in recs]


def test_model_conf_carries_no_signal_but_head_recovers():
    torch.manual_seed(3)
    recs, ok = _records()
    model_conf = [0.9999] * len(recs)          # the mid profile
    right_c = [c for c, o in zip(model_conf, ok) if o]
    wrong_c = [c for c, o in zip(model_conf, ok) if not o]
    assert right_c and wrong_c, "test needs both classes"

    # convergence budget recorded: the honest-minimum char mean-pool
    # encoder separates marked states but needs ~1500 epochs at lr 1e-1
    # (embedding organization is the bottleneck -- measured, not assumed)
    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    head.fit(recs, epochs=1500, lr=1e-1)
    acc, ece = head.evaluate(recs)
    head.eval()
    hc_r, hc_w = [], []
    with torch.no_grad():
        for rec, o in zip(recs, ok):
            p_yes, _ = head.decide("trust", rec["ids"])   # v5.36: (P, None)
            (hc_r if o else hc_w).append(p_yes)
    m_r, m_w = sum(hc_r)/len(hc_r), sum(hc_w)/len(hc_w)
    assert acc["trust"] > 0.9, acc
    assert m_w < 0.5 < m_r, f"head did not separate: right {m_r} wrong {m_w}"
    assert ece["trust"] < 0.3, ece  # measured 0.267 at ep1500
    print(f"PASS test_model_conf_carries_no_signal_but_head_recovers "
          f"(model 0.9999/0.9999 -> head {m_r:.2f}/{m_w:.2f}, ece "
          f"{ece['trust']:.3f})")


def test_rlcd_penalizes_confident_wrong():
    adj = rlcd_reward_adjustment([0.0, 0.0, 1.0, 1.0],
                                 [0.9999, 0.3, 0.9999, 0.3],
                                 [0.0, 0.0, 1.0, 1.0], lam=2.0)
    # confident-wrong pays ~2x the humble-wrong penalty
    assert adj[0] < adj[1] - 1.0, adj
    # confident-right keeps most of the reward; humble-right loses less
    assert adj[2] > adj[3], adj
    print(f"PASS test_rlcd_penalizes_confident_wrong {adj}")


if __name__ == "__main__":
    test_model_conf_carries_no_signal_but_head_recovers()
    test_rlcd_penalizes_confident_wrong()
    print("\n2/2 RLCD head tests passed")
