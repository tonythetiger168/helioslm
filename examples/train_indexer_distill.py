"""Distill-train the learned lightning indexer on the toy checkpoint
(limitations round 2026-10-07).

The v5.21-L learned indexer (attention/lightning_indexer.py) ships with a
DSA-style distillation loss (MLA.indexer_distill_loss) but every existing
checkpoint predates it — so ``indexer="learned"`` rows in
benchmarks/bench_sparse_mtp.py were refused loudly (untrained indexer =
noise). This script closes that gap end to end:

1. Loads the toy checkpoint into a learned-indexer config
   (sparse_top_k + sparse_indexer="learned"). The checkpoint carries no
   indexer.* weights, so the load is strict-except-indexer — ANY other
   missing/unexpected key fails loudly (a silently misloaded trunk would
   poison every downstream number).
2. Distills each MLA layer's indexer against that layer's own teacher
   (the detached head-mean of the true absorbed score terms — exactly the
   v5.8 free indexer) on windows of the repo-source corpus (the same
   distribution the toy checkpoint was trained on; sampling real text,
   not random token ids, keeps the teacher meaningful).
3. Reports per-run loss and teacher top-k overlap BEFORE vs AFTER
   (training evidence, not assumed improvement).
4. Saves a checkpoint with indexer weights; bench_sparse_mtp accepts it
   for indexer="learned" rows.

Honesty notes:
- The teacher IS the head-mean free indexer, so a fully distilled
  learned indexer reproduces head-mean selections (teacher ceiling);
  claims are scoped to fidelity-to-teacher, never to "better attention".
- All non-indexer parameters are frozen: the trunk (and its MTP head)
  is bit-preserved, so dense outputs of the saved checkpoint are
  bit-identical to the source checkpoint's.
- Overlap is measured with the same top-k the sweep uses; near-tied
  teacher scores cap how high overlap can go (see test comments).

Usage:
    python examples/train_indexer_distill.py \
        --checkpoint checkpoints/toy_v5.13.pt \
        --out checkpoints/toy_v5.13_indexer.pt --steps 400
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5


def load_with_new_indexer(model, state_dict):
    """Strict load that tolerates ONLY missing indexer.* keys (the trunk
    checkpoint predates the learned indexer). Any other mismatch fails
    loudly — a silently misloaded trunk invalidates the whole run."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if ".indexer." not in k]
    if bad_missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the model trunk: missing "
            f"(non-indexer) {bad_missing[:5]}, unexpected "
            f"{unexpected[:5]} — refusing to distill on a misloaded trunk"
        )
    n_idx = len([k for k in missing if ".indexer." in k])
    return n_idx


def _mla_layers(model):
    layers = [
        layer.attention for layer in model.layers
        if hasattr(layer, "attention")
        and getattr(layer.attention, "sparse_indexer", None) == "learned"
    ]
    if not layers:
        raise ValueError(
            "no MLA layer with sparse_indexer='learned' found — build the "
            "model with attention.sparse_indexer='learned' and "
            "attention.sparse_top_k set before distilling"
        )
    return layers


def _capture_inputs(model):
    """Forward hooks returning {mla_module: detached input hidden}."""
    store = {}
    handles = []

    def make_hook(module):
        def hook(mod, args, kwargs):
            store[mod] = args[0].detach()
        return hook

    for mla in _mla_layers(model):
        handles.append(mla.register_forward_pre_hook(
            make_hook(mla), with_kwargs=True))
    return store, handles


@torch.no_grad()
def _hidden_per_layer(model, ids):
    store, handles = _capture_inputs(model)
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    return store


def teacher_overlap(model, ids, top_k):
    """Mean top-k overlap between each learned indexer's selection and its
    teacher's (head-mean true score terms), over all layers/queries."""
    store = _hidden_per_layer(model, ids)
    overlaps = []
    with torch.no_grad():
        for mla, hidden in store.items():
            pos, _ = mla._resolve_positions(hidden, 0, None)
            qn, qr, c, kr = mla._project_new_tokens(hidden, pos)
            c = c.unsqueeze(1)
            w_uk, _ = mla._w_uk_w_uv()
            qa = torch.einsum("bhqd,hdc->bhqc", qn, w_uk)
            teacher = (qa.mean(1, keepdim=True) @ c.transpose(-2, -1)
                       + qr.mean(1, keepdim=True) @ kr.transpose(-2, -1))
            student = mla.indexer(hidden, c)
            tk = teacher.topk(top_k, -1).indices
            sk = student.topk(top_k, -1).indices
            overlaps.append(
                (tk.unsqueeze(-1) == sk.unsqueeze(-2)
                 ).any(-1).float().mean().item())
    return sum(overlaps) / len(overlaps)


def distill_indexer(model, corpus_ids, steps=400, seq_len=32, batch_size=4,
                    lr=3e-3, top_k=4, seed=1234, log_every=0):
    """Distill every learned-indexer layer against its own teacher.

    All non-indexer parameters are frozen (trunk bit-preserved). Returns a
    stats dict with initial/final loss and teacher overlap (measured on a
    fixed eval batch, so the before/after comparison is apples-to-apples).
    """
    mla_layers = _mla_layers(model)
    for name, p in model.named_parameters():
        p.requires_grad_(".indexer." in name)
    idx_params = [p for n, p in model.named_parameters() if ".indexer." in n]
    opt = torch.optim.Adam(idx_params, lr=lr)
    gen = torch.Generator().manual_seed(seed)

    eval_ids = _sample_windows(corpus_ids, 2, seq_len + 1, gen)
    loss0 = _distill_loss(model, mla_layers, eval_ids).item()
    ov0 = teacher_overlap(model, eval_ids, top_k)

    model.train()
    t0 = time.time()
    for step in range(steps):
        ids = _sample_windows(corpus_ids, batch_size, seq_len + 1, gen)
        loss = _distill_loss(model, mla_layers, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps} loss {loss.item():.4f}")
    model.eval()

    loss1 = _distill_loss(model, mla_layers, eval_ids).item()
    ov1 = teacher_overlap(model, eval_ids, top_k)
    return {
        "layers": len(mla_layers),
        "steps": steps,
        "loss_initial": round(loss0, 4),
        "loss_final": round(loss1, 4),
        "teacher_overlap_initial": round(ov0, 4),
        "teacher_overlap_final": round(ov1, 4),
        "wall_seconds": round(time.time() - t0, 1),
    }


def _distill_loss(model, mla_layers, ids):
    """Sum of per-layer indexer distill losses on one batch (grad-enabled;
    trunk activations computed under no_grad and detached)."""
    store = _hidden_per_layer(model, ids)
    return torch.stack([
        mla.indexer_distill_loss(store[mla]) for mla in mla_layers
    ]).sum()


def _sample_windows(corpus_ids, n, width, gen):
    hi = len(corpus_ids) - width
    if hi <= 0:
        raise ValueError(
            f"corpus too small for window width {width} "
            f"({len(corpus_ids)} ids)"
        )
    starts = torch.randint(0, hi, (n,), generator=gen)
    return torch.stack([
        torch.tensor(corpus_ids[s:s + width], dtype=torch.long)
        for s in starts.tolist()
    ])


def build_corpus_ids(vocab_size):
    """Repo-source corpus as char ids (same source + filter as
    examples/train_toy_checkpoint.py)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "__pycache__", "checkpoints")]
        for fn in files:
            if fn.endswith((".py", ".md")):
                try:
                    with open(os.path.join(dirpath, fn),
                              encoding="utf-8") as f:
                        parts.append(f.read())
                except OSError:
                    pass
    text = "\n\n".join(parts)
    if len(text) < 50000:
        raise ValueError(f"corpus too small ({len(text)} chars)")
    return [ord(c) for c in text if ord(c) < vocab_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/toy_v5.13.pt")
    ap.add_argument("--out", default="checkpoints/toy_v5.13_indexer.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sparse-top-k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"source checkpoint not found: {args.checkpoint}"
        )
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.sparse_top_k = args.sparse_top_k
    cfg.attention.sparse_indexer = "learned"
    torch.manual_seed(args.seed)
    model = HeliosLMv5(cfg).eval()
    n_idx = load_with_new_indexer(model, ckpt["state_dict"])
    print(f"loaded trunk; {n_idx} indexer params freshly initialized")

    corpus = build_corpus_ids(cfg.vocab_size)
    print(f"corpus: {len(corpus)} ids; distilling {args.steps} steps ...")
    stats = distill_indexer(model, corpus, steps=args.steps,
                            seq_len=args.seq_len, batch_size=args.batch_size,
                            lr=args.lr, top_k=args.sparse_top_k,
                            seed=args.seed, log_every=100)
    print("distill stats:", stats)

    payload = dict(ckpt)  # preserve original meta fields
    payload["state_dict"] = model.state_dict()
    payload["indexer_distill"] = stats
    payload["indexer_note"] = (
        "learned lightning indexer distilled against the head-mean "
        "teacher (examples/train_indexer_distill.py); trunk weights "
        "bit-preserved (frozen during distill)"
    )
    torch.save(payload, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
