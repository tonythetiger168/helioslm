"""nope_temperature_ab.py — roadmap U3: Llama 4 NoPE attention temperature,
length-generalization A/B.

One all-NoPE MLA model is trained on key-value recall (MQAR-style) at
sequence length 64. With ``nope_temp_floor_scale`` = 64 the temperature is
exactly 1.0 at every training position, so training is identical with or
without it: the A/B is purely the inference-time scaling, evaluated on
the SAME weights at 1x, 2x and 4x the training length (Llama 4's use: the
temperature only kicks in beyond the floor scale).

Task: 8 key-value bindings, then filler, then the 8 keys queried in a
random order; accuracy on the 8 answers (chance 1/16). Train stream seed
1000, eval stream seed 0. Lengths grow only by filler, so the query-to-
binding distance grows with the length.

Usage:
  python examples/nope_temperature_ab.py --output benchmarks/nope_temperature_ab_2026-10-16.json
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

from evastar.eval import multiseed  # noqa: E402
from evastar.configs.config_v5 import (  # noqa: E402
    AttentionConfig, EvastarV5Config, HybridAttentionConfig, MoEConfig,
    MTPConfig)
from evastar.src.model_v5 import EvastarV5  # noqa: E402

N_PAIRS, N_KEYS = 8, 16
KEY0, VAL0, FILL0, VOCAB = 4, 20, 36, 64
TRAIN_SEQ = 64
EVAL_SEQS = (64, 128, 256)


def config(temp_scale):
    return EvastarV5Config(
        size="lite", num_hidden_layers=4, vocab_size=VOCAB, hidden_size=128,
        intermediate_size=256,
        attention=AttentionConfig(
            nope=True, nope_attn_temperature=temp_scale > 0,
            nope_temp_floor_scale=TRAIN_SEQ, nope_temp_scale=temp_scale),
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=4, num_activated_experts=2,
                      expert_hidden_size=128),
        use_attention_residuals=False, use_hyper_connections=False)


def make_batch(batch, seq, gen):
    filler = seq - 4 * N_PAIRS
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:N_PAIRS]
                        for _ in range(batch)]) + KEY0
    vals = torch.randint(0, N_KEYS, (batch, N_PAIRS), generator=gen) + VAL0
    fill = torch.randint(FILL0, VOCAB, (batch, filler), generator=gen)
    order = torch.stack([torch.randperm(N_PAIRS, generator=gen)
                         for _ in range(batch)])
    qk, qv = keys.gather(1, order), vals.gather(1, order)
    ids = torch.cat([torch.stack([keys, vals], -1).flatten(1), fill,
                     torch.stack([qk, qv], -1).flatten(1)], 1)
    pos = torch.arange(2 * N_PAIRS + filler + 1, seq, 2)
    return ids, pos, qv


@torch.no_grad()
def evaluate(model, seq, n_batches=8, batch=32):
    gen = torch.Generator().manual_seed(0)
    hit = tot = 0
    for _ in range(n_batches):
        ids, pos, ans = make_batch(batch, seq, gen)
        pred = model(ids)[0][:, pos - 1].argmax(-1)
        hit += (pred == ans).sum().item()
        tot += ans.numel()
    return round(hit / tot, 4)


def experiment(args):
    """One run at args.seed; returns the payload (not written)."""

    torch.manual_seed(args.seed)
    model = EvastarV5(config(0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        ids, pos, ans = make_batch(args.batch, TRAIN_SEQ, gen)
        logits = model(ids)[0]
        loss = F.cross_entropy(logits[:, pos - 1].reshape(-1, VOCAB),
                               ans.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0:
            print(f"step {step + 1}: loss {loss.item():.3f}", flush=True)
    state = model.state_dict()

    results = []
    for scale in [0.0] + list(args.scales):
        m = EvastarV5(config(scale))
        m.load_state_dict(state)
        m.eval()
        row = {"nope_temp_scale": scale,
               "acc": {str(s): evaluate(m, s) for s in EVAL_SEQS}}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"experiment": "U3 NoPE attention temperature (Llama 4), "
                         "length generalization on one set of weights",
           "task": {"pairs": N_PAIRS, "keys": N_KEYS, "chance": 1 / N_KEYS,
                    "train_seq": TRAIN_SEQ, "eval_seqs": list(EVAL_SEQS),
                    "floor_scale": TRAIN_SEQ, "train_seed": 1000 + args.seed,
                    "eval_seed": 0},
           "train": {k: v for k, v in vars(args).items() if k not in ("output", "seeds")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    multiseed.add_seed_args(ap, default=0)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--output")
    args = ap.parse_args()
    out = multiseed.run(args, experiment)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
