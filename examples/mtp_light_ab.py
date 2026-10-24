"""mtp_light_ab.py — roadmap B2-5: MiMo-style light MTP head vs the
default MTP block, on the same frozen main model.

The committed toy checkpoint (lite, char-level) supplies a trained main
model. Its MTP head is discarded; a fresh head of each style is trained
with the MTP loss only (main model frozen, hidden states detached) on the
same corpus batches, same steps, same seed. Then, per style:

- head parameters (fusion projection + transformer block);
- greedy speculative decoding vs plain greedy on 8 prompts: the
  token-identity GATE must hold, plus acceptance rate, accepted drafts per
  round and new tokens per verification round;
- MTP-head forward latency for one inference step (CPU, telemetry).

Usage:
  python examples/mtp_light_ab.py --output benchmarks/mtp_light_ab_2026-10-16.json
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
sys.path.insert(0, str(_REPO / "examples"))

from evastar.eval import multiseed  # noqa: E402
from evastar.configs.config_v5 import EvastarV5Config, MTPConfig  # noqa: E402
from evastar.src.inference.mtp import MTPDecoder, MTPModule  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

CKPT = _REPO / "checkpoints" / "toy_v5.13.pt"
PROMPTS = ["HeliosLM", "The model", "import torch", "def forward",
           "# v5.16", "Attention(", "KV cache", "speculative"]
STYLES = ("default", "light")


def load_main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = EvastarV5Config(size="lite")
    model = EvastarV5(cfg)
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), cfg


def fresh_head(model, style, window, seed):
    cfg = EvastarV5Config(size="lite",
                           mtp=MTPConfig(style=style, light_window=window))
    torch.manual_seed(seed)
    head = MTPModule(cfg, 0, embed_tokens=model.embed_tokens,
                     lm_head=model.lm_head)
    return head, cfg


def head_params(head):
    return sum(p.numel() for n, p in head.named_parameters()
               if not n.startswith(("embed_tokens", "lm_head")))


def train_head(model, head, ids, args):
    params = [p for n, p in head.named_parameters()
              if not n.startswith(("embed_tokens", "lm_head"))]
    opt = torch.optim.Adam(params, lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    head.train()
    losses = []
    for _ in range(args.steps):
        ix = torch.randint(0, len(ids) - args.seq - 2, (args.batch,),
                           generator=gen)
        x = torch.stack([ids[i:i + args.seq] for i in ix])
        with torch.no_grad():
            _, hidden, _ = model(x)
        logits, _ = head.forward_with_hidden(hidden, x)
        usable = x.shape[1] - 2
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            x[:, 2:2 + usable].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    head.eval()
    return sum(losses[-20:]) / len(losses[-20:])


@torch.no_grad()
def evaluate(model, head, cfg, max_new):
    from train_toy_checkpoint import encode
    model.mtp_modules = torch.nn.ModuleList([head])
    drafted = accepted = rounds = 0
    gate = True
    for p in PROMPTS:
        x = torch.tensor([encode(p, cfg.vocab_size)], dtype=torch.long)
        plain = model.generate(x, max_new_tokens=max_new, temperature=0)
        r = MTPDecoder(model, model.mtp_modules, cfg).generate(
            x, max_new_tokens=max_new, temperature=0)
        gate &= torch.equal(plain[0], r.sequences[0][:plain.shape[1]])
        drafted += r.num_drafted
        accepted += r.num_accepted
        rounds += r.num_rounds
    h = torch.randn(1, 1, cfg.hidden_size)
    t = torch.tensor([[5]])
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        head.forward_with_hidden(h, t)
        times.append(time.perf_counter() - t0)
    return {"gate_greedy_equal": gate,
            "acceptance_rate": round(accepted / drafted, 4),
            "accepted_per_round": round(accepted / rounds, 4),
            "tokens_per_round": round(len(PROMPTS) * max_new / rounds, 4)}, \
        statistics.median(times) * 1e6


def experiment(args):
    """One run at args.seed; returns the payload (not written)."""
    torch.set_num_threads(1)

    from train_toy_checkpoint import build_corpus, encode
    model, cfg = load_main()
    ids = torch.tensor(encode(build_corpus(), cfg.vocab_size), dtype=torch.long)

    results, latency = [], {}
    for style in STYLES:
        head, hcfg = fresh_head(model, style, args.window, args.seed)
        loss = train_head(model, head, ids, args)
        ev, us = evaluate(model, head, hcfg, args.max_new)
        if not ev["gate_greedy_equal"]:
            sys.exit(f"greedy-equivalence gate FAILED for {style}")
        row = {"style": style, "head_params": head_params(head),
               "ffn_size": head.transformer.linear1.out_features,
               "train_window": head.window, "mtp_loss_last20": round(loss, 4),
               **ev}
        results.append(row)
        latency[style] = round(us, 1)
        print(json.dumps(row), f"latency {us:.0f}us", flush=True)
    out = {"experiment": "B2-5 light MTP head A/B (frozen toy main model)",
           "train": {k: v for k, v in vars(args).items() if k not in ("output", "seeds")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "head_step_latency_us_median": latency}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=16)
    multiseed.add_seed_args(ap, default=0)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--output")
    args = ap.parse_args()
    out = multiseed.run(args, experiment)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
