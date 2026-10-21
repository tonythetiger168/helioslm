"""smebu_balance_ab.py — roadmap U7: SMEBU (Arcee Trinity) vs the sign-based
aux-free bias update, load-balance A/B.

One MoE layer (32 experts, top-4, lite width) routes a skewed token
stream: every token carries a fixed offset along a random direction, so
the bias-free router prefers a few experts. Each step routes a fresh
batch in training mode (loads are recorded) and then calls
``update_bias()``. Both arms use the same weights, the same token
stream and the same step size lambda; only the update rule differs:

  heuristic: bias -= lambda * sign(load - mean)
  smebu:     v = (mean - load) / mean
             m <- beta m + (1 - beta) tanh(kappa v),  bias += lambda m

Reported per (lambda, arm): MaxVio = max_i (load_i - mean) / mean per
step; its mean and variance over the second half of the run (the
roadmap criterion is that SMEBU lowers the variance); the mean over the
first 50 steps (convergence); and how often a bias update flips sign.

Usage:
  python examples/smebu_balance_ab.py --output benchmarks/smebu_balance_ab_2026-10-16.json
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

from helioslm_v5.eval import multiseed  # noqa: E402
from helioslm_v5.configs.config_v5 import HeliosLMv5Config, MoEConfig  # noqa: E402
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402

N_EXPERTS, TOP_K = 32, 4


def build(strategy, lam, seed):
    cfg = HeliosLMv5Config(size="lite", moe=MoEConfig(
        num_experts=N_EXPERTS, num_activated_experts=TOP_K,
        device_group_size=1, balance_strategy=strategy,
        bias_update_rate=lam))
    torch.manual_seed(seed)
    return DeviceLimitedMoE(cfg).train(), cfg.hidden_size


def run(strategy, lam, steps, batch, skew, seed):
    moe, hidden = build(strategy, lam, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    direction = torch.randn(hidden, generator=gen)
    direction = direction / direction.norm() * skew
    maxvio, flips, prev = [], 0, None
    for p in moe.parameters():
        p.requires_grad_(False)
    # grad mode stays ON: the MoE only records loads when
    # torch.is_grad_enabled() (training statistics); no parameter needs
    # grad, so no graph is built.
    for _ in range(steps):
        x = torch.randn(1, batch, hidden, generator=gen) + direction
        moe(x)
        load = moe.expert_load.clone()
        mean = load.mean()
        maxvio.append(((load - mean) / mean).max().item())
        before = moe.route_bias.clone()
        moe.update_bias()
        delta = moe.route_bias - before
        if prev is not None:
            flips += int(((delta * prev) < 0).sum())
        prev = delta
    tail = maxvio[steps // 2:]
    return {"strategy": strategy, "lambda": lam,
            "maxvio_first50_mean": round(statistics.fmean(maxvio[:50]), 4),
            "maxvio_tail_mean": round(statistics.fmean(tail), 4),
            "maxvio_tail_var": round(statistics.pvariance(tail), 6),
            "bias_sign_flips_per_step": round(flips / (steps - 1), 3)}


def experiment(args):
    """One run at args.seed; returns the payload (not written)."""
    t0 = time.time()
    results = []
    for lam in args.lambdas:
        for strategy in ("heuristic", "smebu"):
            row = run(strategy, lam, args.steps, args.batch, args.skew, args.seed)
            results.append(row)
            print(json.dumps(row), flush=True)
    out = {"experiment": "U7 SMEBU vs sign-based aux-free bias update, "
                         "skewed token stream, one MoE layer",
           "setup": {"experts": N_EXPERTS, "top_k": TOP_K,
                     "smebu_beta": 0.5, "smebu_kappa": 2.0,
                     **{k: v for k, v in vars(args).items() if k not in ("output", "seeds")}},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=512)
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
