"""hca_ab_passkey.py — B1-2: MLA-quality dense vs HCA high-compression,
with and without sparse top-k rescue. A bitwise-reproducible A/B at toy
scale.

WHAT THIS MEASURES
------------------
The quality breakpoint of SEQUENCE compression. MLA's KV cache is
lossless up to floating-point reassociation (the repo's own suite
verifies absorbed == expanded at atol 1e-4), so a plain full-causal
attention arm is the MLA-quality reference. HCA (kv_compress.py) is the
opposite trade: the far past is mean-pooled into one slot per `chunk`
tokens (8x/32x/128x compression), which destroys exact per-token
information. The rescue arm adds a sparse top-k exact-token retrieval
over the compressed region.

METHOD (constructive oracle — no training, no dataset)
------------------------------------------------------
Embedding-space passkey probe with IDENTITY-wired projections
(q = k = v = x, out = identity; rescue indexer = identity). Tokens are
random unit vectors scaled to amplitude A; a planted pair token is the
superposition (u_key + w_val)/sqrt(2) * A * boost; the final-position
query is u_key * A. The attention mechanism that preserves the far past
puts its largest score on the planted token; the readout argmaxes the
output vector over the candidate value directions. Norms are boosted
(boost > 1) because self-attention is the argmax by Cauchy-Schwarz —
the planted token must out-score the query itself.

This isolates the ARCHITECTURAL bottleneck from training dynamics:
every arm sees identical geometry, identical weights, identical seeds.
It answers "can this attention pattern, at its best, carry an exact
token from the far past to the readout" — not "does SGD find the
circuit". The trained-circuit variant is deliberately out of scope for
this script (CPU budget; see examples/train_indexer_distill.py for the
indexer-training machinery).

RLCDAlignBench note: the gated dataset (sumleo/RLCDAlignBench) is not
in the repo and not used here; when access lands, the same arms can be
scored zero-shot with eval/mini_bench.py's choice-loglikelihood
harness. Until then this probe is the CPU-reproducible substitute —
the JSON is tagged accordingly so the numbers can never be quoted as
RLCDAlignBench results.

BITWISE DISCIPLINE
------------------
All randomness flows through torch.Generator with fixed seeds; results
carry a sha256 over the canonical JSON of the accuracy table. Re-run
with the same torch build to reproduce byte-identical numbers.

Usage:
  python examples/hca_ab_passkey.py                          # defaults
  python examples/hca_ab_passkey.py --output benchmarks/hca_ab_2026-10-12.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evastar.eval import multiseed  # noqa: E402
from evastar.src.attention.kv_compress import HighCompressionAttention

SCORING_TAG = "constructive-oracle-passkey-readout"
PROBE_VERSION = 1


class DenseCausalAttention(nn.Module):
    """Full causal MHA — the MLA-quality reference arm.

    Justification (recorded): MLA's absorbed cache is mathematically
    identical to expanded dense attention up to fp reassociation (the
    repo suite verifies atol 1e-4), so plain full attention is the
    honest stand-in for MLA quality. Sequence compression — the thing
    B1-2 measures — is what the HCA arms add.
    """

    def __init__(self, dim: int, n_heads: int = 2):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        H, dh = self.n_heads, self.dh
        q, k, v = self.qkv(x).view(B, L, 3, H, dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


def _wire_identity(m: nn.Module, dim: int) -> nn.Module:
    """Identity-wire qkv/out (and the rescue indexer when present) so
    every arm sees the SAME geometry; only the attention pattern differs."""
    with torch.no_grad():
        eye = torch.eye(dim)
        m.qkv.weight.copy_(torch.cat([eye] * 3, dim=0))
        m.qkv.bias.zero_()
        m.out.weight.copy_(eye)
        m.out.bias.zero_()
        if hasattr(m, "rescue_indexer"):
            m.rescue_indexer.weight.copy_(eye)
            m.rescue_indexer.bias.zero_()
    return m.eval()


def _trial(g: torch.Generator, seq_len: int, n_pairs: int, dim: int,
           amp: float, boost: float):
    """One passkey instance in embedding space. Returns
    (x, correct_idx, candidate_value_vectors)."""
    x = torch.randn(1, seq_len, dim, generator=g)
    x = x / x.norm(dim=-1, keepdim=True) * amp
    ukeys = torch.nn.functional.normalize(
        torch.randn(n_pairs, dim, generator=g), dim=-1)
    wvals = torch.nn.functional.normalize(
        torch.randn(n_pairs, dim, generator=g), dim=-1)
    slots = torch.randperm(seq_len - 8, generator=g)[:n_pairs] + 2
    for kp, vp, s in zip(ukeys, wvals, slots):
        x[0, s] = (kp + vp) / math.sqrt(2) * amp * boost
    qi = int(torch.randint(n_pairs, (1,), generator=g))
    x[0, -1] = ukeys[qi] * amp
    return x, qi, wvals


def probe_accuracy(make_module, seq_len: int, n_pairs: int, dim: int,
                   amp: float, boost: float, n_trials: int,
                   seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    correct = 0
    with torch.no_grad():
        for _ in range(n_trials):
            x, qi, wvals = _trial(g, seq_len, n_pairs, dim, amp, boost)
            out = make_module()(x)[0, -1]
            scores = (wvals * amp) @ out
            correct += int(scores.argmax().item() == qi)
    return correct / n_trials


def run(args) -> dict:
    blocks = []
    for L in args.seq_lens:
        arms = {"dense": lambda: _wire_identity(
            DenseCausalAttention(args.dim, n_heads=args.heads), args.dim)}
        for chunk in args.chunks:
            for rescue in (0, args.rescue_topk):
                name = f"hca-{chunk}x" + (f"+r{rescue}" if rescue else "")
                arms[name] = (lambda c=chunk, r=rescue: _wire_identity(
                    HighCompressionAttention(
                        args.dim, n_heads=args.heads, window=args.window,
                        chunk=c, rescue_topk=r), args.dim))
        accuracy = {}
        for name, make in arms.items():
            accuracy[name] = probe_accuracy(make, L, args.pairs, args.dim,
                                            args.amp, args.boost,
                                            args.trials, args.seed)
        blocks.append({"seq_len": L, "accuracy": accuracy})

    payload = {
        "tag": "hca-ab-passkey",
        "scoring": SCORING_TAG,
        "probe_version": PROBE_VERSION,
        "date": args.date,
        "config": {
            "dim": args.dim, "heads": args.heads, "window": args.window,
            "pairs": args.pairs, "amp": args.amp, "boost": args.boost,
            "trials": args.trials, "seed": args.seed,
            "chunks": args.chunks, "rescue_topk": args.rescue_topk,
            "seq_lens": args.seq_lens,
        },
        "torch": torch.__version__,
        "results": blocks,
        "cache_accounting": {
            f"hca-{c}x": HighCompressionAttention(
                args.dim, n_heads=args.heads, window=args.window,
                chunk=c, rescue_topk=0).get_compression_stats(
                    max(args.seq_lens))
            for c in args.chunks},
        "notes": [
            "dense arm = MLA-quality reference (absorbed cache is "
            "lossless up to fp reassociation, verified by the repo suite)",
            "identity-wired constructive oracle: no training, no dataset; "
            "NOT an RLCDAlignBench result (gated dataset not in repo)",
        ],
    }
    canon = json.dumps(payload["results"], sort_keys=True).encode()
    payload["results_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def main():
    ap = argparse.ArgumentParser(
        description="B1-2: dense (MLA-quality) vs HCA high-compression "
                    "+/- sparse top-k rescue — bitwise-reproducible "
                    "constructive passkey probe")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--chunks", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--rescue-topk", type=int, default=4)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--amp", type=float, default=24.0)
    ap.add_argument("--boost", type=float, default=1.6)
    multiseed.add_seed_args(ap, default=0)
    ap.add_argument("--date", default="2026-10-12")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    payload = multiseed.run(args, run)
    if args.seeds is not None:
        print(json.dumps(payload["aggregate"], indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(payload, indent=2) + "\n",
                                         encoding="utf-8")
            print(f"wrote {args.output}")
        return payload

    # console table
    names = ["dense"] + [f"hca-{c}x" for c in args.chunks] + \
            [f"hca-{c}x+r{args.rescue_topk}" for c in args.chunks]
    print(f"\nB1-2 A/B — constructive passkey readout "
          f"(trials={args.trials}, seed={args.seed}, {SCORING_TAG})")
    print(f"{'arm':<16}" + "".join(f"L={L:>6}" for L in args.seq_lens))
    for name in names:
        row = "".join(
            f"{b['accuracy'][name]:>8.3f}" if name in b["accuracy"]
            else f"{'-':>8}" for b in payload["results"])
        print(f"{name:<16}{row}")
    print(f"\nchance level: {1.0 / args.pairs:.3f}")
    print(f"results sha256: {payload['results_sha256']}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    return payload


if __name__ == "__main__":
    main()
