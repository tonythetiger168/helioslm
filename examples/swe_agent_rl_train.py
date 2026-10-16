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
    ap.add_argument("--false-done-penalty", type=float, default=1.0,
                    help="reward for DONE with a failing verifier is -this")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sft-demos", type=int, default=0,
                    help="scripted demos per TRAIN task for the SFT warm "
                         "start (0 = no SFT, RL from scratch)")
    ap.add_argument("--sft-steps", type=int, default=0)
    ap.add_argument("--sft-lr", type=float, default=1e-3)
    ap.add_argument("--output")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    cfg = policy_config(args)
    model = HeliosLMv5(cfg)
    ref = copy.deepcopy(model)
    trainer = GRPOTrainer(model, ref, cfg)
    train_tasks, held_tasks = sb.TASKS[:10], sb.TASKS[10:]

    before = {"train": ar.evaluate(model, train_tasks),
              "held_out": ar.evaluate(model, held_tasks)}
    print("before", before, flush=True)

    t0 = time.time()
    after_sft = None
    if args.sft_demos and args.sft_steps:
        # Light SFT warm start (behaviour cloning of the careful script on
        # the TRAIN tasks only), then RL — the SFT -> online-RL order of
        # Llama 4 / Olmo 3. Held-out tasks never contribute demos.
        drng = random.Random(500 + args.seed)
        demos = [ar.scripted_demo(t, drng) for t in train_tasks
                 for _ in range(args.sft_demos)]
        assert all(d["passed"] for d in demos)
        sft_opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr)
        model.train()
        for step in range(args.sft_steps):
            loss = ar.sft_loss(model, demos)
            sft_opt.zero_grad()
            loss.backward()
            sft_opt.step()
            if (step + 1) % max(1, args.sft_steps // 5) == 0:
                print(f"sft step {step + 1}: nll {loss.item():.4f}", flush=True)
        after_sft = {"train": ar.evaluate(model, train_tasks),
                     "held_out": ar.evaluate(model, held_tasks)}
        print("after_sft", after_sft, flush=True)
        # KL anchor and old policy start from the SFT model
        trainer.ref_model.load_state_dict(model.state_dict())

    rng = random.Random(1000 + args.seed)
    curve = []
    for u in range(args.updates):
        tasks = rng.sample(train_tasks, args.tasks_per_update)
        samples = []
        model.eval()
        for t in tasks:
            for _ in range(args.group):
                samples.append(ar.rollout(
                    model, t, rng,
                    false_done_penalty=args.false_done_penalty))
        rewards = torch.tensor([s["reward"] for s in samples])
        stats = trainer._learn_from_samples(samples, None, rewards=rewards)
        mean_steps = sum(s["steps"] for s in samples) / len(samples)
        pass_rate = sum(s["passed"] for s in samples) / len(samples)
        curve.append([u + 1, round(rewards.mean().item(), 4),
                      round(pass_rate, 4), round(mean_steps, 2)])
        print(f"update {u + 1}: reward {rewards.mean():.3f} pass "
              f"{pass_rate:.3f} steps "
              f"{mean_steps:.2f} loss {stats['loss']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    after = {"train": ar.evaluate(model, train_tasks),
             "held_out": ar.evaluate(model, held_tasks)}
    model.eval()
    example = ar.rollout(model, held_tasks[0], random.Random(7), greedy=True)
    print("after", after, "example trace", example["trace"], flush=True)
    results = {"before": before, "after_sft": after_sft, "after": after,
               "curve": curve,
               "example_held_out_trace": example["trace"]}
    out = {
        "experiment": "AC2 multi-turn agent RL (macro actions, GRPO; "
                      "optional SFT warm start)",
        "setting": {"train_tasks": [t["id"] for t in train_tasks],
                    "held_out_tasks": [t["id"] for t in held_tasks],
                    "candidates": "reference fix + 2 broken patches, "
                                  "shuffled per episode",
                    "reward": "+1 DONE with verifier (visible + hidden) "
                              "passing, -false_done_penalty DONE failing, "
                              "0 out of steps",
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
