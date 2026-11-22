"""docs/CALIBRATION.md probe: measured DecisionHead calibration numbers.

Trains the non-autoregressive DecisionHead on a learnable routing task,
then reports calibration on (a) held-out in-distribution states and
(b) a distribution shift where the route rule is inverted — the regime
where calibration on wrong answers matters. Block (c) spends a bounded
budget of oracle-verified outcomes (the "verified decision" signal of
docs/cost_per_verified_decision.md) on the P2 OutcomeCalibrator and
reports post-warmup recalibrated confidence on the remaining shifted
records. Prints measured JSON only; it asserts nothing about the
values. Seeds are fixed so runs are reproducible bit-for-bit on CPU.
"""
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from helioslm_v5.agent.decision_head import DecisionHead  # noqa: E402
from helioslm_v5.agent.outcome_calibration import OutcomeCalibrator  # noqa: E402
from helioslm_v5.eval.calibration_board import auroc, brier  # noqa: E402
from helioslm_v5.eval.quant_calib import ece  # noqa: E402

rng = random.Random(20261123)
ROUTES = ("direct", "tool", "escalate")


def make_state(route, solvable, shift=False):
    """State text encodes the rule. Under shift=True the route rule is
    inverted (alpha<->gamma), so a trained head is systematically wrong
    with high confidence — the regime where calibration matters."""
    marker_route = {"direct": "escalate", "tool": "tool",
                    "escalate": "direct"}[route] if shift else route
    marker = {"direct": "alpha", "tool": "beta",
              "escalate": "gamma"}[marker_route]
    sv = "feasible" if solvable else "blocked"
    filler = " ".join(rng.choice(["task", "state", "node", "queue",
                                  "delta", "omega"]) for _ in range(6))
    text = f"{marker} {sv} {filler}"
    ids = torch.tensor([ord(c) % 1024 for c in text], dtype=torch.long)
    return {"ids": ids,
            "answers": {"route": route,
                        "solvable": "yes" if solvable else "no"}}


def make_split(n, shift=False):
    return [make_state(rng.choice(ROUTES), rng.random() < 0.6, shift)
            for _ in range(n)]


def conf_correct(head, records):
    confs, corr = [], []
    for r in records:
        ans, conf = head.decide("route", r["ids"])
        confs.append(conf)
        corr.append(1.0 if ans == r["answers"]["route"] else 0.0)
    return confs, corr


def main():
    torch.manual_seed(7)
    head = DecisionHead({"route": ("choice", ROUTES),
                         "solvable": ("noul", None)}, d=64)
    head.fit(make_split(400), epochs=200, lr=1e-2)
    held, shifted = make_split(200), make_split(200, shift=True)

    acc_h, ece_h = head.evaluate(held)
    acc_s, ece_s = head.evaluate(shifted)
    ch, rh = conf_correct(head, held)
    cs, rs = conf_correct(head, shifted)

    # P2 block: stream the SAME shifted records; the first `budget`
    # decisions get oracle-verified outcomes (the verified-decision
    # signal), the calibrator fits on exactly those, and everything
    # after the warmup is scored with recalibrated confidence. Answers
    # are never changed, so accuracy is identical by construction and
    # every point of AUROC/Brier lift is confidence ranking.
    budget = 40
    cal = OutcomeCalibrator(head.encode)
    for r in shifted[:budget]:
        ans, _conf = head.decide("route", r["ids"])
        cal.observe(r["ids"], ans == r["answers"]["route"])
    cal.fit()
    ca, ra = [], []
    for r in shifted[budget:]:
        ans, conf = head.decide("route", r["ids"])
        ca.append(cal.adjust(r["ids"], conf))
        ra.append(1.0 if ans == r["answers"]["route"] else 0.0)

    print(json.dumps({
        "train_records": 400,
        "heldout": {
            "n": 200,
            "route_acc": acc_h["route"], "route_ece": ece_h["route"],
            "route_auroc": auroc(ch, rh), "route_brier": brier(ch, rh),
            "solvable_acc": acc_h["solvable"],
            "solvable_mean_abs_err": ece_h["solvable"],
        },
        "shifted": {
            "n": 200,
            "route_acc": acc_s["route"], "route_ece": ece_s["route"],
            "route_auroc": auroc(cs, rs), "route_brier": brier(cs, rs),
            "solvable_acc": acc_s["solvable"],
            "solvable_mean_abs_err": ece_s["solvable"],
        },
        "shifted_outcome_recalibrated": {
            "budget_verified": budget,
            "scored_n": len(ra),
            "route_acc": sum(ra) / len(ra),
            "route_ece": ece(ca, [int(y) for y in ra]),
            "route_auroc": auroc(ca, ra), "route_brier": brier(ca, ra),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
