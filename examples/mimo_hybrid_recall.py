"""mimo_hybrid_recall.py — roadmap B2-1 + B2-2: does a MiMo-style 5:1
sliding-window/global interleave keep long-range recall that all-SWA
loses, and what does the learnable sink change?

TASK (trained, synthetic key-value recall, MQAR-style)
------------------------------------------------------
Each sequence is ``[k1 v1 ... k8 v8] [64 filler tokens] [kq vq] x 8``:
8 key-value bindings, filler, then the 8 keys queried again in a random
order. Loss and accuracy are taken ONLY on the 8 query-value positions
(chance = 1/16). The query sits >= 64 tokens after its binding.

ARMS (identical width/depth/seed/data; attention layout is the only change)
---------------------------------------------------------------------------
- ``dense``: all 6 layers global (reference).
- ``all_swa``: all 6 layers window 8. Stacked reach is ~6 * 7 = 42
  tokens < 64, so this arm structurally cannot see the binding and
  should sit near chance — the negative control.
- ``hybrid_5to1``: ``global_attention_every=6`` -> layers G S S S S G.
- ``hybrid_5to1_sink``: the same plus ``learnable_sink`` (B2-2).

Per arm: query accuracy on a held-out eval stream (seed 0; train stream
seed 1000), final train loss, parameter count, and attendable keys per
query at the eval length (the KV a windowed layer actually reads; the
current cache is NOT trimmed, so this is an access count, not memory).

Usage:
  python examples/mimo_hybrid_recall.py --output benchmarks/mimo_hybrid_2026-10-15.json
  python examples/mimo_hybrid_recall.py --steps 50 --arms dense   # smoke
  # one process per arm, then combine:
  python examples/mimo_hybrid_recall.py --arms dense --threads 1 --output d.json
  python examples/mimo_hybrid_recall.py --merge d.json s.json ... --output all.json
"""
import argparse
import hashlib
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
from evastar.src.model_v5 import EvastarV5  # noqa: E402

N_PAIRS = 8
N_KEYS = 16
KEY0, VAL0, FILL0, VOCAB = 4, 20, 36, 64
FILLER = 64
SEQ = 4 * N_PAIRS + FILLER          # 96
WINDOW = 8
LAYERS = 6
ARMS = ("dense", "all_swa", "hybrid_5to1", "hybrid_5to1_sink")
# Q1 (v5.63): linear-attention hybrid (GDA layers at 1, 3, 4; MLA at
# 0, 2, 5 with full_attention_every=3) without / with the short conv.
EXTRA_ARMS = ("gda_hybrid", "gda_hybrid_conv")
ALL_ARMS = ARMS + EXTRA_ARMS


def arm_config(arm):
    if arm in EXTRA_ARMS:
        return EvastarV5Config(
            size="lite", num_hidden_layers=LAYERS, vocab_size=VOCAB,
            hidden_size=128, intermediate_size=256,
            hybrid_attention=HybridAttentionConfig(
                enabled=True, full_attention_every=3,
                short_conv_kernel=4 if arm == "gda_hybrid_conv" else 0),
            mtp=MTPConfig(enabled=False),
            moe=MoEConfig(num_experts=4, num_activated_experts=2,
                          expert_hidden_size=128),
            use_attention_residuals=False, use_hyper_connections=False)
    att = {"dense": AttentionConfig(),
           "all_swa": AttentionConfig(sliding_window=WINDOW),
           "hybrid_5to1": AttentionConfig(sliding_window=WINDOW,
                                          global_attention_every=6),
           "hybrid_5to1_sink": AttentionConfig(sliding_window=WINDOW,
                                               global_attention_every=6,
                                               learnable_sink=True)}[arm]
    return EvastarV5Config(
        size="lite", num_hidden_layers=LAYERS, vocab_size=VOCAB,
        hidden_size=128, intermediate_size=256, attention=att,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=4, num_activated_experts=2,
                      expert_hidden_size=128),
        use_attention_residuals=False, use_hyper_connections=False)


def make_batch(batch, gen):
    """Returns (ids [B, SEQ], answer positions [8], answers [B, 8])."""
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:N_PAIRS]
                        for _ in range(batch)]) + KEY0
    vals = torch.randint(0, N_KEYS, (batch, N_PAIRS), generator=gen) + VAL0
    filler = torch.randint(FILL0, VOCAB, (batch, FILLER), generator=gen)
    order = torch.stack([torch.randperm(N_PAIRS, generator=gen)
                         for _ in range(batch)])
    qk = keys.gather(1, order)
    qv = vals.gather(1, order)
    bind = torch.stack([keys, vals], -1).flatten(1)
    query = torch.stack([qk, qv], -1).flatten(1)
    ids = torch.cat([bind, filler, query], 1)
    ans_pos = torch.arange(2 * N_PAIRS + FILLER + 1, SEQ, 2)
    return ids, ans_pos, qv


def attendable_keys(cfg):
    """Keys each layer may read for the LAST query, summed over layers."""
    w = cfg.attention.sliding_window
    every = cfg.attention.global_attention_every
    total = 0
    if cfg.hybrid_attention.enabled:      # GDA layers read a fixed state
        from evastar.src.model_v5 import _is_full_attention_layer
        return sum(SEQ for i in range(cfg.num_hidden_layers)
                   if _is_full_attention_layer(cfg, i))
    for i in range(cfg.num_hidden_layers):
        is_global = w is None or (every is not None
                                  and (i == 0 or i % every == every - 1))
        total += SEQ if is_global else min(w, SEQ)
    return total


def evaluate(model, n_batches, batch):
    gen = torch.Generator().manual_seed(0)
    hit = tot = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            ids, pos, ans = make_batch(batch, gen)
            logits = model(ids)[0]
            pred = logits[:, pos - 1].argmax(-1)
            hit += (pred == ans).sum().item()
            tot += ans.numel()
    model.train()
    return hit / tot


def train_arm(arm, args):
    cfg = arm_config(arm)
    torch.manual_seed(args.seed)
    model = EvastarV5(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.steps, 1), pct_start=0.1)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    losses, curve = [], []
    t0 = time.time()
    for step in range(args.steps):
        ids, pos, ans = make_batch(args.batch, gen)
        logits = model(ids)[0]
        loss = F.cross_entropy(logits[:, pos - 1].reshape(-1, VOCAB),
                               ans.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if args.eval_every and (step + 1) % args.eval_every == 0:
            acc = evaluate(model, 2, 64)
            curve.append([step + 1, round(acc, 4)])
            print(f"  {arm} step {step + 1}: loss "
                  f"{sum(losses[-50:]) / len(losses[-50:]):.3f} acc {acc:.3f}",
                  flush=True)
    acc = evaluate(model, args.eval_batches, 64)
    sinks = None
    if cfg.attention.learnable_sink:
        sinks = [[round(v, 4) for v in layer.attention.attn_sink.tolist()]
                 for layer in model.layers]
    return {
        "arm": arm,
        "layout": "".join(
            "L" if getattr(layer.attention, "is_recurrent_attention", False)
            else "G" if layer.attention.sliding_window is None else "S"
            for layer in model.layers),
        "params": sum(p.numel() for p in model.parameters()),
        "query_acc": round(acc, 4),
        "train_loss_last50": round(sum(losses[-50:]) / max(len(losses[-50:]), 1), 4),
        "attendable_keys_last_query": attendable_keys(cfg),
        "curve": curve,
        "final_sink_logits": sinks,
    }, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ALL_ARMS)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--merge", nargs="+", metavar="JSON",
                    help="combine per-arm outputs (same settings) into one")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        out = parts[0]
        for p in parts[1:]:
            assert p["task"] == out["task"] and p["train"] == out["train"], p
            out["results"] += p["results"]
            out["wall_seconds"].update(p["wall_seconds"])
        out["results"].sort(key=lambda r: ALL_ARMS.index(r["arm"]))
        out["results_sha256"] = hashlib.sha256(json.dumps(
            out["results"], sort_keys=True).encode()).hexdigest()
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)
        return

    results, wall = [], {}
    for arm in args.arms:
        r, secs = train_arm(arm, args)
        results.append(r)
        wall[arm] = round(secs, 1)
        print(f"{arm}: acc {r['query_acc']} loss {r['train_loss_last50']} "
              f"layout {r['layout']} keys {r['attendable_keys_last_query']} "
              f"({secs:.0f}s)", flush=True)
    canon = json.dumps(results, sort_keys=True).encode()
    out = {
        "experiment": "B2-1/B2-2 MiMo hybrid recall",
        "task": {"pairs": N_PAIRS, "keys": N_KEYS, "filler": FILLER,
                 "seq": SEQ, "window": WINDOW, "layers": LAYERS,
                 "chance": 1 / N_KEYS, "train_seed": 1000 + args.seed,
                 "eval_seed": 0},
        "train": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                  "seed": args.seed, "eval_batches": args.eval_batches},
        "results": results,
        "results_sha256": hashlib.sha256(canon).hexdigest(),
        "wall_seconds": wall,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
