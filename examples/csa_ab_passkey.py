"""csa_ab_passkey.py — roadmap Phase B: dense vs CSA (compress, then
top-k select) on the B1-2 constructive passkey probe.

Same probe, same seeds and same dense reference arm as
examples/hca_ab_passkey.py (imported, not copied), so the CSA rows sit
next to the HCA rows of benchmarks/hca_ab_2026-10-12.md. Every arm is
identity-wired (q = k = v = x, out = identity, indexer = identity): the
numbers measure what the attention PATTERN can carry from the far past,
not what training would find. Not an RLCDAlignBench result.

Arms: dense, and CSA at compression ``ratio`` in {4, 16} with top-k in
{8, 32} compressed entries per query.

Run from repo root:
    python examples/csa_ab_passkey.py --output benchmarks/csa_ab_2026-10-15.json
    python examples/csa_ab_passkey.py --check    # verify the committed file
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from hca_ab_passkey import (  # noqa: E402
    SCORING_TAG, DenseCausalAttention, _trial, _wire_identity,
    probe_accuracy)
from helioslm_v5.src.attention.kv_compress import (  # noqa: E402
    CompressedSparseAttention)

DEFAULT_OUT = "benchmarks/csa_ab_2026-10-15.json"


def _wire_csa(m, dim):
    _wire_identity(m, dim)
    with torch.no_grad():
        m.indexer.weight.copy_(torch.eye(dim))
        m.indexer.bias.zero_()
    return m.eval()


def selection_hit_rate(make_module, ratio, seq_len, args):
    """Fraction of trials where the final query's top-k selection
    contains the chunk holding the queried planted token. Separates
    "the indexer missed it" from "the compressed entry could not carry
    it" (same trial stream as probe_accuracy)."""
    g = torch.Generator().manual_seed(args.seed)
    hits = 0
    with torch.no_grad():
        for _ in range(args.trials):
            x, _qi, _w = _trial(g, seq_len, args.pairs, args.dim,
                                args.amp, args.boost)
            target = int((x[0, :-1] @ x[0, -1]).argmax())
            _, picks = make_module()(x, return_selection=True)
            hits += int(target // ratio in picks[0, -1].tolist())
    return hits / args.trials


def run(args):
    blocks = []
    for L in args.seq_lens:
        arms = {"dense": lambda: _wire_identity(
            DenseCausalAttention(args.dim, n_heads=args.heads), args.dim)}
        for ratio in args.ratios:
            for k in args.topks:
                arms[f"csa-{ratio}x@k{k}"] = (
                    lambda r=ratio, kk=k: _wire_csa(CompressedSparseAttention(
                        args.dim, n_heads=args.heads, ratio=r, topk=kk),
                        args.dim))
        accuracy = {name: probe_accuracy(make, L, args.pairs, args.dim,
                                         args.amp, args.boost, args.trials,
                                         args.seed)
                    for name, make in arms.items()}
        hit_rate = {name: selection_hit_rate(
            make, int(name.split("-")[1].split("x")[0]), L, args)
            for name, make in arms.items() if name.startswith("csa")}
        blocks.append({"seq_len": L, "accuracy": accuracy,
                       "selection_hit_rate": hit_rate})
    blob = json.dumps(blocks, sort_keys=True, separators=(",", ":"))
    return {
        "experiment": "Phase B v4_csa: dense vs CSA compress-then-select",
        "scoring": SCORING_TAG,
        "verification_tier": "measured",
        "date": args.date,
        "torch": torch.__version__,
        "params": {k: getattr(args, k) for k in (
            "seq_lens", "ratios", "topks", "trials", "pairs", "dim",
            "heads", "amp", "boost", "seed")},
        "chance": 1.0 / args.pairs,
        "note": "identity-wired constructive oracle (no training, no "
                "dataset); same probe and dense arm as hca_ab_2026-10-12",
        "results": blocks,
        "results_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--ratios", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--topks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--amp", type=float, default=24.0)
    ap.add_argument("--boost", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--output", default=None)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare with the committed results")
    args = ap.parse_args()
    payload = run(args)

    names = list(payload["results"][0]["accuracy"])
    print(f"{'arm':<16}" + "".join(f"L={b['seq_len']:>6}"
                                   for b in payload["results"]))
    for name in names:
        print(f"{name:<16}" + "".join(f"{b['accuracy'][name]:>8.3f}"
                                      for b in payload["results"]))
    print(f"chance {payload['chance']:.3f}; results sha256 "
          f"{payload['results_sha256']}")

    if args.check:
        committed = json.loads((_REPO / DEFAULT_OUT).read_text())
        if committed["results_sha256"] != payload["results_sha256"]:
            sys.exit("csa A/B results differ from the committed artifact")
        print("csa A/B results match the committed artifact")
    elif args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n",
                                     encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
