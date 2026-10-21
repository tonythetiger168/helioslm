"""kv_share_ab.py — roadmap U6: cross-layer KV sharing (Gemma 3n/4),
small-model perplexity A/B.

Arms: ``attention.kv_share_last_n`` = N over an all-MLA lite-width model
(8 layers by default). Every arm uses the same seed, so all weights an arm
keeps start from the same init as the N = 0 arm (sharing is applied after
init), and the same char-level corpus batches in the same order (the repo
corpus of train_toy_checkpoint.py). The last 10% of the corpus is held
out; validation is the mean next-char loss over fixed held-out windows.

Per arm it reports:
  - params;
  - KV cache bytes per token MEASURED from a real prefill cache (unique
    tensor storages, so a shared layer's aliased entry counts once) and
    its ratio to N = 0, which must equal (L - N) / L exactly;
  - final train loss (mean of the last 20 steps), held-out val loss and
    perplexity.

Usage:
  python examples/kv_share_ab.py --output benchmarks/kv_share_ab_2026-10-16.json
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from helioslm_v5.eval import multiseed  # noqa: E402
from helioslm_v5.configs.config_v5 import (  # noqa: E402
    AttentionConfig, HeliosLMv5Config, HybridAttentionConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402


def config(n_share, layers):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=layers,
        attention=AttentionConfig(kv_share_last_n=n_share),
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False))


@torch.no_grad()
def cache_bytes_per_token(model, vocab, seq=64):
    ids = torch.randint(0, vocab, (1, seq), generator=torch.Generator().manual_seed(0))
    _, _, past = model.eval()(ids, use_cache=True)
    seen, total = set(), 0
    for entry in past:
        for t in entry:
            key = t.untyped_storage().data_ptr()
            if key not in seen:
                seen.add(key)
                total += t.untyped_storage().nbytes()
    return total / seq


def experiment(args):
    """One run at args.seed; returns the payload (not written)."""

    from train_toy_checkpoint import build_corpus, encode
    vocab = config(0, args.layers).vocab_size
    ids = torch.tensor(encode(build_corpus(), vocab), dtype=torch.long)
    cut = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:cut], ids[cut:]
    vgen = torch.Generator().manual_seed(args.seed + 999)
    starts = torch.randint(0, len(val_ids) - args.seq - 1,
                           (args.val_windows,), generator=vgen)
    val = torch.stack([val_ids[i:i + args.seq + 1] for i in starts])

    results = []
    t0 = time.time()
    for n in args.shares:
        cfg = config(n, args.layers)
        torch.manual_seed(args.seed)
        model = HeliosLMv5(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        kv_bpt = cache_bytes_per_token(model, vocab)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        gen = torch.Generator().manual_seed(args.seed)
        losses = []
        for step in range(args.steps):
            ix = torch.randint(0, len(train_ids) - args.seq - 2,
                               (args.batch,), generator=gen)
            data = torch.stack([train_ids[i:i + args.seq + 1] for i in ix])
            logits = model(data[:, :-1])[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   data[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f"N={n} step {step + 1}: loss {loss.item():.3f}", flush=True)
        model.eval()
        with torch.no_grad():
            vl = model(val[:, :-1])[0]
            val_loss = F.cross_entropy(vl.reshape(-1, vl.shape[-1]),
                                       val[:, 1:].reshape(-1)).item()
        row = {"kv_share_last_n": n, "params": n_params,
               "kv_bytes_per_token": kv_bpt,
               "train_loss_last20": round(sum(losses[-20:]) / 20, 4),
               "val_loss": round(val_loss, 4),
               "val_ppl": round(math.exp(val_loss), 3)}
        results.append(row)
        print(json.dumps(row), flush=True)

    base = results[0]["kv_bytes_per_token"]
    for r in results:
        r["kv_ratio_vs_first"] = r["kv_bytes_per_token"] / base
    out = {"experiment": "U6 cross-layer KV sharing (Gemma 3n/4), char-LM "
                         "perplexity A/B at lite width",
           "model": {"layers": args.layers, "all_mla": True, "mtp": False},
           "data": {"corpus": "train_toy_checkpoint.build_corpus (char ids)",
                    "train_tokens": len(train_ids), "val_tokens": len(val_ids),
                    "val_windows": args.val_windows},
           "train": {k: v for k, v in vars(args).items() if k not in ("output", "seeds")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--shares", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    multiseed.add_seed_args(ap, default=0)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--output")
    args = ap.parse_args()
    out = multiseed.run(args, experiment)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
