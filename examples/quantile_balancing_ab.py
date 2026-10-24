"""quantile_balancing_ab.py — Stable LatentMoE study: Kimi K3 Quantile
Balancing vs the other aux-free bias rules on the U7 harness.

One LatentMoE layer (32 experts, top-4, latent routing) routes a skewed
token stream: every token carries a fixed offset along a random
direction, so the bias-free router prefers a few experts. Each update
routes ``micro`` micro-batches in training mode (statistics are recorded)
and then calls ``update_bias()``. All arms use the same weights and the
same stream; only ``balance_strategy`` differs:

  heuristic           bias -= lambda * sign(load - mean)
  smebu               soft-clamped momentum (v5.72)
  quantile            v5.5: bias from the last 512 margins, not centred
  quantile_balancing  K3: bias solved from every token's margin since the
                      last update (histogram), mean-centred

Scenarios:
  stationary  the U7 stream (one direction, 512 tokens per update);
  drift       the skew direction is replaced by a new one halfway;
  micro       4 micro-batches of 256 tokens per update (1024 tokens; the
              v5.5 window keeps the last 512).

Reported per (scenario, arm): MaxVio = max_i (load_i - mean) / mean,
measured on each update's own tokens BEFORE that update (so it is the
imbalance the current bias produces): mean and variance over the second
half, mean over the first 20 updates, the mean over the 20 updates after
the drift, and |mean bias| at the end (drift of the bias itself).

Usage:
  python examples/quantile_balancing_ab.py --output benchmarks/quantile_balancing_ab_2026-10-21.json
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval import multiseed  # noqa: E402
from evastar.configs.config_v5 import EvastarV5Config, MoEConfig  # noqa: E402
from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402

N_EXPERTS, TOP_K = 32, 4
STRATEGIES = ("heuristic", "smebu", "quantile", "quantile_balancing")
SCENARIOS = {"stationary": (1, 512, False), "drift": (1, 512, True),
             "micro": (4, 256, False)}


def build(strategy, lam, seed):
    cfg = EvastarV5Config(size="lite", moe=MoEConfig(
        num_experts=N_EXPERTS, num_activated_experts=TOP_K,
        device_group_size=1, balance_strategy=strategy,
        bias_update_rate=lam, latent_dim=64))
    torch.manual_seed(seed)
    return DeviceLimitedMoE(cfg).train(), cfg.hidden_size


def run(strategy, scenario, lam, updates, skew, seed):
    micro, tokens, drift = SCENARIOS[scenario]
    moe, hidden = build(strategy, lam, seed)
    gen = torch.Generator().manual_seed(seed + 1)

    def direction():
        d = torch.randn(hidden, generator=gen)
        return d / d.norm() * skew

    offset = direction()
    second = direction()
    for p in moe.parameters():
        p.requires_grad_(False)
    maxvio = []
    for u in range(updates):
        if drift and u == updates // 2:
            offset = second
        for _ in range(micro):
            moe(torch.randn(1, tokens, hidden, generator=gen) + offset)
        load = moe.expert_load.clone()
        mean = load.mean()
        maxvio.append(((load - mean) / mean).max().item())
        moe.update_bias()
    half = updates // 2
    tail = maxvio[half:]
    return {"scenario": scenario, "strategy": strategy,
            "maxvio_first20_mean": round(statistics.fmean(maxvio[:20]), 4),
            "maxvio_tail_mean": round(statistics.fmean(tail), 4),
            "maxvio_tail_var": round(statistics.pvariance(tail), 6),
            "maxvio_after_drift20_mean": round(
                statistics.fmean(maxvio[half:half + 20]), 4),
            "bias_abs_mean_end": round(moe.route_bias.mean().abs().item(), 6)}


def experiment(args):
    """One run at args.seed; returns the payload (not written)."""
    t0 = time.time()
    results = []
    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            row = run(strategy, scenario, args.lam, args.updates, args.skew,
                      args.seed)
            results.append(row)
            print(json.dumps(row), flush=True)
    out = {"experiment": "Stable LatentMoE: K3 Quantile Balancing vs sign / "
                         "SMEBU / v5.5 quantile bias rules, one LatentMoE "
                         "layer, skewed token stream",
           "setup": {"experts": N_EXPERTS, "top_k": TOP_K, "latent_dim": 64,
                     "scenarios": {k: {"micro_batches": v[0],
                                       "tokens_per_micro": v[1],
                                       "drift_at_half": v[2]}
                                   for k, v in SCENARIOS.items()},
                     **{k: v for k, v in vars(args).items() if k not in ("output", "seeds")}},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--skew", type=float, default=3.0)
    multiseed.add_seed_args(ap, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    out = multiseed.run(args, experiment)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
