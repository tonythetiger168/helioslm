"""swe_agent_rl_train.py — roadmap Phase AC, AC2: multi-turn agent RL on
the mini SWE sandbox with the repo's GRPO update.

Trains a tiny HeliosLM policy (macro actions, see
helioslm_v5/src/training/swe_agent_rl.py) on the first 10 SWE tasks and
evaluates greedy pass rate on those tasks AND on the 10 held-out tasks,
before and after training, over 3 shuffled candidate orders each. The
reward is the real verifier (visible + hidden tests) run in the sandbox.

Usage:
  python examples/swe_agent_rl_train.py --output benchmarks/swe_agent_rl_2026-10-16.json
  python examples/swe_agent_rl_train.py --updates 3      # smoke
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    GRPOConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.eval import swe_bench as sb  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training import swe_agent_rl as ar  # noqa: E402
from helioslm_v5.src.training.grpo import GRPOTrainer  # noqa: E402


def policy_config(args):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=2, vocab_size=ar.VOCAB,
        hidden_size=128, intermediate_size=256,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=4, num_activated_experts=2,
                      expert_hidden_size=128),
        use_attention_residuals=False, use_hyper_connections=False,
        grpo=GRPOConfig(group_size=args.group, epsilon=0.2,
                        kl_coef=args.kl, lr=args.lr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--tasks-per-update", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = policy_config(args)
    model = HeliosLMv5(cfg)
    ref = copy.deepcopy(model)
    trainer = GRPOTrainer(model, ref, cfg)
    train_tasks, held_tasks = sb.TASKS[:10], sb.TASKS[10:]

    before = {"train": ar.evaluate(model, train_tasks),
              "held_out": ar.evaluate(model, held_tasks)}
    print("before", before, flush=True)

    rng = random.Random(1000 + args.seed)
    curve = []
    t0 = time.time()
    for u in range(args.updates):
        tasks = rng.sample(train_tasks, args.tasks_per_update)
        samples = []
        model.eval()
        for t in tasks:
            for _ in range(args.group):
                samples.append(ar.rollout(model, t, rng))
        rewards = torch.tensor([s["reward"] for s in samples])
        stats = trainer._learn_from_samples(samples, None, rewards=rewards)
        mean_steps = sum(s["steps"] for s in samples) / len(samples)
        curve.append([u + 1, round(rewards.mean().item(), 4),
                      round(mean_steps, 2)])
        print(f"update {u + 1}: reward {rewards.mean():.3f} steps "
              f"{mean_steps:.2f} loss {stats['loss']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    after = {"train": ar.evaluate(model, train_tasks),
             "held_out": ar.evaluate(model, held_tasks)}
    model.eval()
    example = ar.rollout(model, held_tasks[0], random.Random(7), greedy=True)
    print("after", after, "example trace", example["trace"], flush=True)
    results = {"before": before, "after": after, "curve": curve,
               "example_held_out_trace": example["trace"]}
    out = {
        "experiment": "AC2 multi-turn agent RL (macro actions, GRPO)",
        "setting": {"train_tasks": [t["id"] for t in train_tasks],
                    "held_out_tasks": [t["id"] for t in held_tasks],
                    "candidates": "reference fix + 2 broken patches, "
                                  "shuffled per episode",
                    "reward": "1 if DONE and verifier (visible + hidden) "
                              "passes, else 0",
                    "eval": "greedy, 3 shuffled candidate orders per task",
                    "max_steps": ar.MAX_STEPS},
        "train": vars(args) | {"output": None},
        "results": results,
        "results_sha256": hashlib.sha256(
            json.dumps(results, sort_keys=True).encode()).hexdigest(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
