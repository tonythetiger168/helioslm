"""Train the HeliosLM toy checkpoint (v5.13).

Trains the lite config on a small character-level corpus built from the
repo's own docs (offline-guaranteed), with an auxiliary MTP loss so the
draft head learns to propose tokens (addresses the "MTP acceptance
requires trained weights" limitation end-to-end on CPU).

Char tokenizer: token id == ord(ch); only chars with ord < vocab_size
(lite: 1024, covers all of ASCII) are kept.

Usage:
    python examples/train_toy_checkpoint.py                 # full run
    python examples/train_toy_checkpoint.py --steps 50      # smoke run

Outputs:
    checkpoints/toy_v5.13.pt    model state_dict + meta (fp32, ~34MB)
    checkpoints/toy_v5.13.json  train/val loss, samples, MTP acceptance
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def build_corpus() -> str:
    """Collect all *.py + *.md in the repo (~10x the docs-only corpus)."""
    parts = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__",
                                                "checkpoints")]
        for fn in files:
            if fn.endswith((".py", ".md")):
                fp = os.path.join(root, fn)
                try:
                    parts.append(open(fp, encoding="utf-8").read())
                except OSError:
                    pass
    text = "\n\n".join(parts)
    assert len(text) > 50000, f"corpus too small ({len(text)} chars)"
    return text


def encode(text: str, vocab_size: int):
    return [ord(c) for c in text if ord(c) < vocab_size]


def decode(ids) -> str:
    return "".join(chr(int(i)) for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mtp-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "checkpoints",
                                                  "toy_v5.13.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"toy checkpoint: {n_params/1e6:.2f}M params, vocab={cfg.vocab_size}")

    ids = torch.tensor(encode(build_corpus(), cfg.vocab_size), dtype=torch.long)
    # Held-out val windows from a FIXED generator (the corpus tail is one
    # file's ending — unrepresentative of the mixture).
    vgen = torch.Generator().manual_seed(args.seed + 999)
    n_val = 512
    val_starts = torch.randint(0, len(ids) - n_val - 1, (4,), generator=vgen)
    val_windows = torch.cat([ids[i:i + n_val] for i in val_starts])
    train_ids = ids
    print(f"corpus: {len(ids)} tokens ({len(train_ids)} train / {n_val} val)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    model.train()

    def batch(gen):
        ix = torch.randint(0, len(train_ids) - args.seq - 2, (args.batch,),
                           generator=gen)
        return torch.stack([train_ids[i:i + args.seq + 1] for i in ix])

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        data = batch(gen)
        x, y = data[:, :-1], data[:, 1:]
        logits, hidden, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total = loss
        if model.mtp_modules is not None:
            mtp_loss = 0.0
            for mtp in model.mtp_modules:
                d = mtp.module_index + 1
                usable = x.shape[1] - d - 1
                if usable <= 0:
                    continue
                mlogits, _ = mtp.forward_with_hidden(hidden, x)
                tgt = x[:, d + 1: d + 1 + usable].reshape(-1)
                mtp_loss = mtp_loss + torch.nn.functional.cross_entropy(
                    mlogits.reshape(-1, mlogits.shape[-1]), tgt)
            total = total + args.mtp_weight * mtp_loss
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            ml = mtp_loss.item() if model.mtp_modules else float("nan")
            print(f"step {step:4d}  lm {loss.item():.3f}  mtp {ml:.3f}  "
                  f"({time.time() - t0:.0f}s)")

    # ---- validation -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        vx = val_windows[:256].unsqueeze(0)[:, :-1]
        vy = val_windows[:256].unsqueeze(0)[:, 1:]
        vlogits, _, _ = model(vx)
        val_loss = torch.nn.functional.cross_entropy(
            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1)).item()
    print(f"val loss: {val_loss:.3f}")

    # ---- samples + MTP acceptance ---------------------------------------
    samples = {}
    for prompt in ["HeliosLM", "# HeliosLM v5.13"]:
        pids = torch.tensor([encode(prompt, cfg.vocab_size)])
        with torch.no_grad():
            out = model.generate(pids, max_new_tokens=120, temperature=0)[0]
        samples[prompt] = decode(out[len(pids[0]):])
        print(f"--- prompt: {prompt!r}\n{samples[prompt]}\n")

    acc = None
    pids = torch.tensor([encode("HeliosLM", cfg.vocab_size)])
    if model.mtp_modules is not None:
        from helioslm_v5.src.inference.mtp import MTPDecoder
        with torch.no_grad():
            d = MTPDecoder(model, model.mtp_modules, cfg)
            r = d.generate(pids, max_new_tokens=32, temperature=0)
        acc = r.acceptance_rate
        print(f"MTP acceptance: {acc:.3f} "
              f"({r.num_accepted}/{r.num_drafted} accepted, {r.num_rounds} rounds)")
    else:
        print("MTP disabled in config — skipping acceptance probe")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config_size": "lite",
        "tokenizer": "char-ascii",
        "vocab_size": cfg.vocab_size,
        "meta": {"steps": args.steps, "val_loss": val_loss,
                 "train_seconds": round(time.time() - t0)},
    }, args.out)
    meta_path = args.out.replace(".pt", ".json")
    with open(meta_path, "w") as f:
        json.dump({"val_loss": val_loss, "samples": samples,
                   "mtp_acceptance": acc,
                   "params_M": round(n_params / 1e6, 2)}, f, indent=2)
    print(f"saved -> {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
