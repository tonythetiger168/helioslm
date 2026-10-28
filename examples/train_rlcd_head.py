"""train_rlcd_head.py - v5.35: RLCD-flavored calibration head on REAL
mid outcome data.

Source data: benchmarks/mid_agent_eval_v4.json (12 tasks, 0/12, conf
~0.9999) + mid_agent_eval_v5.json (12 tasks, 12/12, conf ~0.9999) --
the same checkpoint, same seed, before/after grounding. Outcomes are
env.verify results; confidence claims are the model's softmax max.

The head sees ONLY the task state (text). The mid confidence is not an
input -- it carries no signal (pinned at ~0.9999 either way). What the
head learns is P(correct | state) from OUTCOMES: the RLCD principle
(probabilities answer to outcomes, not preferences) applied at the
decision layer.

Honest scope (recorded): 24 real records is demonstration-grade, not
production-grade; the value is the closed pipeline (real artifact ->
outcome records -> calibrated head -> reported ECE), which any larger
run plugs into unchanged.

Usage: python examples/train_rlcd_head.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "helioslm_v5" / "agent"))

from decision_data import rlcd_reward_adjustment
from decision_head import DecisionHead
from trajectory import text_to_ids

BENCH = ROOT / "benchmarks"


def load_records():
    """Real outcome records from the mid before/after artifacts."""
    recs = []
    for name in ("mid_agent_eval_v4.json", "mid_agent_eval_v5.json"):
        d = json.loads((BENCH / name).read_text())
        for r in d["results"]:
            recs.append({"task": r["task"], "correct": bool(r["correct"]),
                         "conf": r["conf"]})
    return recs


def main():
    torch.manual_seed(3)
    recs = load_records()
    n_right = sum(r["correct"] for r in recs)
    print(f"records: {len(recs)} ({n_right} correct, "
          f"{len(recs) - n_right} wrong)")

    # the model's own confidence claims: the calibration gap, quantified
    conf_right = [r["conf"] for r in recs if r["correct"]]
    conf_wrong = [r["conf"] for r in recs if not r["correct"]]
    print(f"model conf: right mean {sum(conf_right)/len(conf_right):.4f} "
          f"| wrong mean {sum(conf_wrong)/len(conf_wrong):.4f} "
          "-- the claim carries no signal")

    # RLCD reward view: RLVR reward minus the calibration penalty
    adj = rlcd_reward_adjustment(
        rewards=[1.0 if r["correct"] else 0.0 for r in recs],
        confidences=[r["conf"] for r in recs],
        outcomes=[1.0 if r["correct"] else 0.0 for r in recs], lam=2.0)
    print(f"rlcd-adjusted rewards: "
          f"right {sum(adj[i] for i in range(len(adj)) if recs[i]['correct'])/n_right:.3f} "
          f"| wrong {sum(adj[i] for i in range(len(adj)) if not recs[i]['correct'])/(len(recs)-n_right):.3f}")

    # the head: learns P(correct | state) from outcomes only
    records = [{"ids": torch.tensor(text_to_ids(r["task"])),
                "answers": {"trust": "yes" if r["correct"] else "no",
                            "trust__target_conf": 1.0 if r["correct"] else 0.0}}
               for r in recs]
    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    # 1500 epochs: the char mean-pool encoder converges slowly (measured
    # in T35); the real-data set is small so the budget is cheap
    head.fit(records, epochs=1500, lr=1e-1)
    acc, ece = head.evaluate(records)
    # what the head believes on the failure-class states
    head.eval()
    conf_on_wrong, conf_on_right = [], []
    with torch.no_grad():
        for rec, r in zip(records, recs):
            _, conf = head.decide("trust", rec["ids"])
            (conf_on_right if r["correct"] else conf_on_wrong).append(conf)
    print(f"HEAD conf: right mean {sum(conf_on_right)/len(conf_on_right):.3f} "
          f"| wrong mean {sum(conf_on_wrong)/len(conf_on_wrong):.3f}")
    print(f"HEAD acc {acc['trust']:.3f} ece {ece['trust']:.3f}")
    out = {"n": len(recs), "model_conf_right": sum(conf_right)/len(conf_right),
           "model_conf_wrong": sum(conf_wrong)/len(conf_wrong),
           "head_conf_right": sum(conf_on_right)/len(conf_on_right),
           "head_conf_wrong": sum(conf_on_wrong)/len(conf_on_wrong),
           "head_acc": acc["trust"], "head_ece": ece["trust"],
           "note": "24 real records, demonstration-grade; head sees "
                   "state only, signal comes from outcomes (RLCD)"}
    (BENCH / "rlcd_head_summary.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmarks/rlcd_head_summary.json")


if __name__ == "__main__":
    main()
