"""eval_mid_trust.py - v5.37c: TrustGate x mid, the real acceptance.

Trains a trust head on the REAL failure outcomes from
benchmarks/mid_agent_eval_v4.json (the ungrounded run: 12/12 wrong,
every finish confabulated at ~0.9999 confidence), then drives the mid
checkpoint through AgentLoop behind TrustGate on FRESH eval tasks.

Expected: the head marks these state families as untrustworthy
(P(yes) low) -> TrustGate ESCALATEs -> the model is DENIED ungrounded
action on its own known-failure territory. That is the therapy pair:
grounding fixes what the model does; TrustGate refuses what the head
distrusts.

Honest scope: n=12 real records, demonstration-grade; the point is the
closed pipeline (artifact -> outcome records -> head -> gate behavior),
not the sample size.

Usage (local, after a train_mid_sft.py run + benchmarks present):
    python examples/eval_mid_trust.py
"""
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helioslm_v5" / "agent"))

from _mid_common import load_model_tok, make_model_fn
from decision_head import DecisionHead
from envs import make_envs, make_long_envs
from gate import Route
from helioslm_v5.agent.trajectory import text_to_ids   # noqa: F401 (path anchor)
from trust_gate import TrustGate
from loop import AgentLoop
from tools import build_default_registry

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"


def load_failure_records():
    """Real outcomes: ungrounded mid run -> every task wrong. trust=no
    for all -- these are the measured failure states."""
    d = json.loads((BENCH / "mid_agent_eval_v4.json").read_text())
    return [{"ids": torch.tensor(text_to_ids(r["task"])),
             "answers": {"trust": "no",
                         "trust__target_conf": 0.0}}
            for r in d["results"]]


def main():
    torch.manual_seed(11)
    model, tok, device = load_model_tok()
    state = {"conf": None}
    model_fn = make_model_fn(model, tok, device, max_new=128,
                             state=state)

    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    recs = load_failure_records()
    print(f"training trust head on {len(recs)} real failure records",
          flush=True)
    head.fit(recs, epochs=1500, lr=1e-1)
    head.eval()

    gate = TrustGate(head, hi=0.7, lo=0.3)
    rng = random.Random(777)   # fresh eval tasks (different seed)
    results = []
    with tempfile.TemporaryDirectory() as root:
        for env in make_envs() + make_long_envs():
            reg, impls = build_default_registry(root)
            for _ in range(3):
                task = env.sample(rng)
                loop = AgentLoop(model_fn, reg, impls, gate,
                                 max_steps=task.step_budget)
                traj = loop.run(task.text, seed=1)
                ok = (traj.final_answer is not None
                      and env.verify(task, traj.final_answer))
                results.append({"family": getattr(task, "family",
                                                  type(task).__name__),
                                "correct": ok,
                                "abstained": any(e["route"] == "ESCALATE"
                                                 for e in gate.log[-len(traj.steps):]),
                                "final": traj.final_answer})
                print(f"{results[-1]['family']} correct={ok} "
                      f"final={str(traj.final_answer)[:30]!r}", flush=True)
    abstains = [r for r in results if r["abstained"]]
    corr = [r for r in results if r["correct"]]
    summary = {
        "n": len(results),
        "correct": f"{len(corr)}/{len(results)}",
        "abstained": f"{len(abstains)}/{len(results)}",
        "trust_log_tail": gate.log[-8:],
        "interpretation": "TrustGate should abstain (ESCALATE) on the "
                          "failure-class states the head learned from "
                          "the v4 artifact; DIRECT only where trusted",
        "results": results}
    out = BENCH.parent / "checkpoints" / "mid_trust_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("results", "trust_log_tail")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
