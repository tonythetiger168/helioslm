"""eval_mid_trust.py - v5.37e: TrustGate x mid, the real acceptance.

Trains a trust head on the REAL failure outcomes from
benchmarks/mid_agent_eval_v4.json (the ungrounded run: 12/12 wrong,
every finish confabulated at ~0.9999 confidence), then drives the mid
checkpoint through CHAT SESSIONS behind TrustGate on FRESH eval tasks.

v5.37e FIX: the first version drove AgentLoop (Task:/step i: format) --
the v1 format-skew trap, third occurrence of that family. The mid SFT
data renders chat transcripts, so the harness must be ChatSession.
Parse failures never reach the gate under the wrong harness, which
silently voided the abstention measurement.

Expected: the head marks these state families untrustworthy -> TrustGate
ESCALATEs -> the session receives an abstention observation ("sent to
review") instead of letting the model act. Abstained tasks end with
final=None -- DO NOT ACT is the behavior, not a failure.

Usage (local):
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
from chat import ChatSession
from decision_head import DecisionHead
from envs import make_envs, make_long_envs
from helioslm_v5.agent.trajectory import text_to_ids   # path anchor
from trust_gate import TrustGate
from tools import build_default_registry

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"


def load_failure_records():
    """Real outcomes: ungrounded mid run -> every task wrong."""
    d = json.loads((BENCH / "mid_agent_eval_v4.json").read_text())
    return [{"ids": torch.tensor(text_to_ids(r["task"])),
             "answers": {"trust": "no",
                         "trust__target_conf": 0.0}}
            for r in d["results"]]


class ReviewOracle:
    """Stand-in for the human/vendor review path an abstention routes
    to in production."""
    def escalate(self, call, context):
        return "ABSTAINED (low trust): action routed to review"


def main():
    torch.manual_seed(11)
    model, tok, device = load_model_tok()
    state = {"conf": None}
    model_fn = make_model_fn(model, tok, device, max_new=128, state=state)

    head = DecisionHead({"trust": ("noul", None)}, brier_lambda=4.0)
    recs = load_failure_records()
    print(f"training trust head on {len(recs)} real failure records",
          flush=True)
    head.fit(recs, epochs=1500, lr=1e-1)
    head.eval()

    gate = TrustGate(head, hi=0.7, lo=0.3, inner=ReviewOracle())
    rng = random.Random(777)
    results = []
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(3):
            task = env.sample(rng)
            log_start = len(gate.log)
            session = ChatSession(model_fn, reg, impls, gate,
                                  max_steps=task.step_budget)
            final = session.send(task.text, seed=1)
            decisions = gate.log[log_start:]
            abstained = any(e["route"] == "ESCALATE" for e in decisions)
            ok = final is not None and env.verify(task, final)
            results.append({
                "family": getattr(task, "family", type(task).__name__),
                "correct": ok, "abstained": abstained,
                "p_trust_min": min((e["p_trust"] for e in decisions),
                                   default=None),
                "final": final})
            print(f"{results[-1]['family']} correct={ok} "
                  f"abstained={abstained} "
                  f"p_min={results[-1]['p_trust_min']} "
                  f"final={str(final)[:30]!r}", flush=True)
    abstains = [r for r in results if r["abstained"]]
    corr = [r for r in results if r["correct"]]
    summary = {
        "n": len(results),
        "correct": f"{len(corr)}/{len(results)}",
        "abstained": f"{len(abstains)}/{len(results)}",
        "interpretation": "on failure-class states the head learned from "
                          "the v4 artifact, TrustGate should ESCALATE "
                          "(final=None = do not act); DIRECT+finish only "
                          "where trusted",
        "results": results}
    out = ROOT / "checkpoints" / "mid_trust_eval.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
