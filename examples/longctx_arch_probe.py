"""longctx_arch_probe.py — OS7: long-context recall of four attention
layouts on the same data at the same size (RULER-lite).

Arms (EvastarV5, identical width / depth / MoE / seed / data; the
attention layout is the only change):

- ``dense``: every layer global MLA (reference);
- ``swa_global``: MiMo-style sliding window ``--window`` with a global
  layer at 0 and 3 (``global_attention_every=4``) and learnable sinks;
- ``k3_hybrid``: Kimi-K3 order KDA, KDA, KDA, MLA (per-channel decay);
- ``dsa``: dense MLA with DSA-style top-``--topk`` sparse attention in
  prefill too (head-mean indexer, absorbed mode).

CSA / HCA (DeepSeek V4) are not layers of EvastarV5; their comparison
stays in examples/repo_longctx_probe.py.

Each arm trains on a mix of the RULER-lite tasks
(evastar/eval/ruler_lite.py) at ``--train-len`` tokens, then is
evaluated per task at 1x, 2x and 4x that length (needles uniformly
placed, so distance grows with length).

Usage:
  python examples/longctx_arch_probe.py --device cuda --output benchmarks/longctx_arch_<date>.json
  python examples/longctx_arch_probe.py --steps 20 --train-len 64 --arms dense   # smoke
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import (  # noqa: E402
    AttentionConfig, EvastarV5Config, HybridAttentionConfig, MoEConfig,
    MTPConfig)
from evastar.eval import ruler_lite as rl  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

ARMS = ("dense", "swa_global", "k3_hybrid", "dsa")
LAYERS = 4


def arm_config(arm, hidden=128, window=32, topk=64):
    att = {"dense": AttentionConfig(),
           "swa_global": AttentionConfig(sliding_window=window,
                                         global_attention_every=4,
                                         learnable_sink=True),
           "k3_hybrid": AttentionConfig(),
           "dsa": AttentionConfig(sparse_top_k=topk, sparse_prefill=True)}[arm]
    hyb = (HybridAttentionConfig(enabled=True, full_attention_every=4,
                                 first_layer_full=False, per_channel_decay=True)
           if arm == "k3_hybrid" else HybridAttentionConfig(enabled=False))
    return EvastarV5Config(
        size="lite", num_hidden_layers=LAYERS, vocab_size=rl.VOCAB,
        hidden_size=hidden, intermediate_size=2 * hidden, attention=att,
        hybrid_attention=hyb, mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=4, num_activated_experts=2,
                      expert_hidden_size=hidden),
        use_attention_residuals=False, use_hyper_connections=False)


def train_arm(arm, args):
    cfg = arm_config(arm, args.hidden, args.window, args.topk)
    torch.manual_seed(args.seed)
    model = EvastarV5(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.steps, 1), pct_start=0.1)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    t0, losses = time.time(), []
    for step in range(args.steps):
        task = rl.TASKS[step % len(rl.TASKS)]
        ids, ans = rl.make_batch(task, args.batch, args.train_len, gen)
        logits = model(ids.to(args.device))[0][:, -1]
        loss = F.cross_entropy(logits, ans.to(args.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if args.log_every and (step + 1) % args.log_every == 0:
            print(f"  {arm} step {step + 1}: loss "
                  f"{sum(losses[-50:]) / len(losses[-50:]):.3f}", flush=True)
    acc = {task: {str(m * args.train_len): round(rl.accuracy(
        model, task, m * args.train_len, args.eval_batches, args.batch), 4)
        for m in (1, 2, 4)} for task in rl.TASKS}
    return {"arm": arm,
            "params": sum(p.numel() for p in model.parameters()),
            "train_loss_last50": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4),
            "accuracy": acc,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-len", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--output")
    args = ap.parse_args()
    results = []
    for arm in args.arms:
        r = train_arm(arm, args)
        print(json.dumps(r), flush=True)
        results.append(r)
    out = {"experiment": "OS7 long-context layouts, RULER-lite",
           "chance": round(1 / rl.N_VALUES, 4), "device": args.device,
           "args": vars(args), "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
