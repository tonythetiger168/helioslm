"""benchmark_three_modes.py - v5.29: C-stage three-mode benchmark with the
real tool-tuned checkpoint. One HeliosLM-style "Intelligence Index"
methodology minimal implementation: direct / routed(tau) / oracle on one
task set, with the v5.22 routing-monotonicity gate checked on REAL model
confidence (mean generated-token logprob).

Key efficiency: the model runs ONCE per task; the tau curve is computed
offline by thresholding recorded confidences (escalated => oracle =>
correct). Output: benchmarks/three_modes_2026-09-25.json + console report.

Run from repo root: python3 examples/benchmark_three_modes.py
"""
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.agent.envs import make_envs
from helioslm_v5.agent.gate import GateViolation, routing_gate
from helioslm_v5.agent.tools import build_default_registry
from helioslm_v5.agent.loop import AgentLoop
from helioslm_v5.agent.gate import FixedGate, OracleGate, Route
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

BOS, EOS, MAXC = 1023, 1022, 1021
TASKS_PER_ENV = 4
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SEED = 20260925


def make_model_fn(model):
    """Greedy generation with marker early-stop; returns (text, confidence)."""
    state = {}

    def model_fn(prompt, seed, step):
        ids = torch.tensor([[BOS] + [ord(c) for c in prompt][-1500:]])
        gen, logps = [], []
        for _ in range(96):
            with torch.no_grad():
                logits, _, _ = model(ids)
            probs = torch.log_softmax(logits[0, -1, :MAXC].float(), dim=-1)
            nxt = probs.argmax(-1, keepdim=True)
            logps.append(float(probs[int(nxt)]))
            if int(nxt) == EOS:
                break
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            gen.append(int(nxt))
            if "".join(chr(t) for t in gen[-8:]).endswith("@@end@@"):
                break
        text = "".join(chr(t) for t in gen)
        # geometric-mean token probability, normalized to [0,1] for tau
        state["conf"] = math.exp(sum(logps) / max(len(logps), 1))
        return text
    return model_fn, state


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / \
        "tool_tuned_v5.27.pt"
    model = HeliosLMv5(HeliosLMv5Config(size="lite")).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    envs = make_envs()
    records = []
    t0 = time.time()
    for env in envs:
        model_fn, state = make_model_fn(model)
        with tempfile.TemporaryDirectory() as tmp:
            reg, impls = build_default_registry(tmp)
            loop = AgentLoop(model_fn, reg, impls,
                             FixedGate(Route.DIRECT), max_steps=4)
            for i in range(TASKS_PER_ENV):
                rng = random.Random(f"three:{env.__class__.__name__}:{i}")
                task = env.sample(rng)
                state["conf"] = 0.0
                traj = loop.run(task.text, seed=SEED + i)
                ok = bool(traj.final_answer and env.verify(task, traj.final_answer))
                records.append({"env": env.__class__.__name__, "correct": ok,
                                "conf": state["conf"],
                                "finished": traj.final_answer is not None})
                print(f"  {env.__class__.__name__} {i}: correct={ok} "
                      f"conf={state['conf']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    n = len(records)
    grid = []
    for tau in TAUS:
        direct_frac = sum(1 for r in records if r["conf"] >= tau) / n
        correct = sum(1 for r in records
                      if (r["conf"] >= tau and r["correct"])
                      or r["conf"] < tau) / n      # escalated => oracle => correct
        grid.append((tau, correct))
        print(f"  tau={tau:.2f}: direct={direct_frac:.2f} "
              f"correctness={correct:.3f}")
    try:
        routing_gate(grid)
        gate = "PASS"
    except GateViolation as e:
        gate = f"VIOLATION: {e}"
    print("routing monotonicity gate:", gate)

    direct = sum(1 for r in records if r["correct"]) / n
    report = {"seed": SEED, "n_tasks": n,
              "model": "tool_tuned_v5.27 (ep2 weights)",
              "correctness": {"direct": direct, "oracle": 1.0,
                              "tau_grid": [{"tau": t, "correctness": c}
                                           for t, c in grid]},
              "routing_gate": gate,
              "tasks": records}
    out = Path(__file__).resolve().parent.parent / "benchmarks" / \
        "three_modes_2026-09-25.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\ndirect={direct:.3f} oracle=1.000 (by construction)")
    print("report:", out)


if __name__ == "__main__":
    main()
