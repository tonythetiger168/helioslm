"""thinking_budget_ab.py — roadmap U2: thinking-budget control, accuracy vs
budget on a toy math task, with and without truncation training.

Task: N digits; the answer is their sum mod 10. Each training sequence is

  BOS d1 .. dN = OPEN s1 .. sN CLOSE a EOS

where s_k = (d1 + .. + dk) mod 10 is the running sum (the "thinking"
trace) and a = sN. With a budget b < N the thinking span is forced closed
after s_b, so the model has to finish the remaining N - b additions in
one step: accuracy should rise with the budget.

Two arms, same init, same batches: "full" trains on complete traces only;
"truncation" replaces each trace, with probability 0.5, by a uniformly
truncated one (``random_truncation``, Nemotron-style). Evaluation is
greedy generation from ``BOS digits = OPEN`` with
``ThinkingBudget(budget=b)`` for every b in 0..N, on a held-out stream;
every generated sample is checked to think at most b tokens.

Usage:
  python examples/thinking_budget_ab.py --output benchmarks/thinking_budget_ab_2026-10-17.json
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    AttentionConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig,
    MTPConfig)
from helioslm_v5.src.inference.thinking_budget import (  # noqa: E402
    ThinkingBudget, random_truncation)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
D0, EQ, OPEN, CLOSE = 4, 14, 15, 16
VOCAB = 32
N = 8


def config():
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=4, vocab_size=VOCAB, hidden_size=128,
        intermediate_size=256, pad_token_id=PAD, bos_token_id=BOS,
        eos_token_id=EOS,
        attention=AttentionConfig(),
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=4, num_activated_experts=2,
                      expert_hidden_size=128),
        use_attention_residuals=False, use_hyper_connections=False)


def example(rng):
    d = [rng.randrange(10) for _ in range(N)]
    s, run = [], 0
    for x in d:
        run = (run + x) % 10
        s.append(run)
    seq = ([BOS] + [D0 + x for x in d] + [EQ, OPEN] + [D0 + x for x in s]
           + [CLOSE, D0 + s[-1], EOS])
    return seq, s[-1]


def batch(rng, size, p_trunc):
    seqs = [random_truncation(example(rng)[0], OPEN, CLOSE, p_trunc, rng)
            for _ in range(size)]
    L = max(len(x) for x in seqs)
    ids = torch.full((size, L), PAD)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
    return ids


def train(p_trunc, steps, bsz, lr, seed):
    torch.manual_seed(seed)
    model = HeliosLMv5(config())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = random.Random(1000 + seed)
    model.train()
    for step in range(steps):
        ids = batch(rng, bsz, p_trunc)
        logits = model(ids[:, :-1])[0]
        tgt = ids[:, 1:].clone()
        tgt[:, :N + 2] = -100            # loss only after "= OPEN"
        tgt[tgt == PAD] = -100
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0:
            print(f"p={p_trunc} step {step + 1}: loss {loss.item():.3f}", flush=True)
    return model.eval()


@torch.no_grad()
def evaluate(model, budget, n_eval):
    rng = random.Random(0)
    exs = [example(rng) for _ in range(n_eval)]
    prompts = torch.tensor([e[0][:N + 3] for e in exs])     # BOS d.. = OPEN
    out = model.generate(prompts, max_new_tokens=N + 3, temperature=0,
                         thinking_budget=ThinkingBudget(OPEN, CLOSE, budget))
    hit = 0
    for row, (_, ans) in zip(out[:, N + 3:].tolist(), exs):
        assert CLOSE in row, "thinking span never closed"
        k = row.index(CLOSE)
        assert k <= budget, f"thought {k} tokens with budget {budget}"
        hit += int(k + 1 < len(row) and row[k + 1] == D0 + ans)
    return round(hit / n_eval, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--output")
    args = ap.parse_args()
    t0 = time.time()
    results = []
    for arm, p in (("full", 0.0), ("truncation", 0.5)):
        model = train(p, args.steps, args.batch, args.lr, args.seed)
        row = {"arm": arm, "p_truncate": p,
               "acc_by_budget": {str(b): evaluate(model, b, args.n_eval)
                                 for b in range(N + 1)}}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"experiment": "U2 thinking-budget control: accuracy vs forced "
                         "budget on running-sum mod 10",
           "task": {"digits": N, "chance": 0.1, "train_seed": 1000 + args.seed,
                    "eval_seed": 0},
           "train": {k: v for k, v in vars(args).items() if k != "output"},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
