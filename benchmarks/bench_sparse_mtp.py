"""Sparse top-k x MTP end-to-end acceptance measurement (limitations round
2026-10-06).

The v5.16 break-even sweep (bench_spec_breakeven.py) measures MTP draft
net gain across cache states but keeps attention dense. This script adds
the missing axis: the trunk runs DSA-style sparse top-k decode (v5.8 /
v5.21-L learned-indexer line) and we measure, end to end on the toy
checkpoint, what sparsity costs the MTP draft loop:

    acceptance            MTP acceptance rate under the sparse trunk
    tok_s_out             measured wall-clock output tokens/s (CPU)
    gain_vs_dense_draft0  tok/s vs the dense, draft-off baseline
    output_overlap_vs_dense  fraction of generated positions whose token
                          matches the dense draft=1 run (sparse top-k is
                          an APPROXIMATION — divergence is reported, never
                          hidden)

Rows follow a colibri-P3-compatible superset schema:

    {"model", "sparse_top_k", "indexer", "draft_depth", "acceptance",
     "tok_s_out", "gain_vs_dense_draft0", "n_tokens",
     "output_overlap_vs_dense"}

Oracle / honesty notes:
- ``sparse_top_k >= kv_len`` is bit-identical to dense attention, so the
  k >= L row MUST match the dense draft=1 row on output ids AND
  acceptance exactly (enforced by test_sparse_mtp_acceptance_sweep).
- With k < kv_len the trunk sees fewer keys; identical outputs are a
  (legitimate, measured) property of THIS checkpoint's greedy margins,
  not a guarantee — the schema reports overlap so any divergence shows
  up as data.
- ``indexer="learned"`` rows are refused loudly unless the checkpoint
  carries indexer.* weights: the toy checkpoint predates the learned
  indexer (v5.21-L line), and measuring an UNTRAINED indexer would
  produce noise, not signal.

Usage:
    python benchmarks/bench_sparse_mtp.py \
        --checkpoint checkpoints/toy_v5.13.pt --ks 4,16 --iters 3
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.inference.mtp import MTPDecoder
from helioslm_v5.src.model_v5 import HeliosLMv5

PROMPTS = [
    "HeliosLM", "The model", "import torch", "def forward",
    "# v5.16", "Attention(", "KV cache", "speculative",
]

SCHEMA_KEYS = (
    "model", "sparse_top_k", "indexer", "draft_depth", "acceptance",
    "tok_s_out", "gain_vs_dense_draft0", "n_tokens",
    "output_overlap_vs_dense",
)


def encode(prompt, vocab=1024):
    return torch.tensor([[ord(c) for c in prompt if ord(c) < vocab]])


def _build_model(state_dict, sparse_top_k):
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.sparse_top_k = sparse_top_k
    model = HeliosLMv5(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def sweep_sparse_mtp(checkpoint, ks, max_new=32, iters=3,
                     indexer="head_mean"):
    """Run the sparse x MTP sweep; returns (rows, details).

    ``details[(k, prompt_idx)]`` holds the generated suffix ids (list of
    ints) of the draft=1 run, so tests can assert the k >= L bitwise
    oracle without trusting wall-clock fields.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"sparse x MTP sweep needs the toy checkpoint; not found: "
            f"{checkpoint}"
        )
    if indexer not in ("head_mean", "learned"):
        raise ValueError(
            f"indexer must be 'head_mean' or 'learned', got {indexer!r}"
        )
    for k in ks:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                f"sparse ks must be positive integers, got {k!r}"
            )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    if indexer == "learned" and not any(
            "indexer" in name for name in state_dict):
        raise ValueError(
            "indexer='learned' requires a checkpoint carrying indexer.* "
            "weights; this checkpoint predates the learned indexer "
            "(v5.21-L line) — measuring an untrained indexer would "
            "produce noise, not signal. Train with indexer_distill_loss "
            "first or use indexer='head_mean'"
        )

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(iters)]

    def _run_draft1(model, cfg, ids):
        dec = MTPDecoder(model, model.mtp_modules, cfg)
        t0 = time.perf_counter()
        with torch.no_grad():
            r = dec.generate(ids, max_new_tokens=max_new, temperature=0)
        dt = time.perf_counter() - t0
        n_new = r.sequences.shape[1] - ids.shape[1]
        suffix = r.sequences[0, ids.shape[1]:].tolist()
        return suffix, r.acceptance_rate, n_new / dt

    rows = []
    details = {}

    # Dense baselines (draft off = tok/s reference; draft on = overlap
    # and acceptance reference).
    model_d, cfg_d = _build_model(state_dict, None)
    dense_d0_toks = []
    dense_d1 = []
    for i, prompt in enumerate(prompts):
        ids = encode(prompt)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_d.generate(ids, max_new_tokens=max_new,
                                   temperature=0)
        dense_d0_toks.append((out.shape[1] - ids.shape[1])
                             / (time.perf_counter() - t0))
        suffix, acc, tok_s = _run_draft1(model_d, cfg_d, ids)
        dense_d1.append((suffix, acc, tok_s))
        details[(None, i)] = suffix
    base_toks = statistics.median(dense_d0_toks)
    rows.append({
        "model": "helioslm-toy-8.5M",
        "sparse_top_k": None,
        "indexer": indexer,
        "draft_depth": 0,
        "acceptance": None,
        "tok_s_out": round(base_toks, 3),
        "gain_vs_dense_draft0": None,
        "n_tokens": max_new,
        "output_overlap_vs_dense": None,
    })
    rows.append({
        "model": "helioslm-toy-8.5M",
        "sparse_top_k": None,
        "indexer": indexer,
        "draft_depth": 1,
        "acceptance": round(statistics.mean(x[1] for x in dense_d1), 4),
        "tok_s_out": round(statistics.median(x[2] for x in dense_d1), 3),
        "gain_vs_dense_draft0": round(
            statistics.median(x[2] for x in dense_d1) / base_toks - 1.0, 3),
        "n_tokens": max_new,
        "output_overlap_vs_dense": 1.0,  # dense vs dense by definition
    })

    # Sparse rows (draft on).
    for k in ks:
        model_s, cfg_s = _build_model(state_dict, k)
        per_prompt = []
        for i, prompt in enumerate(prompts):
            ids = encode(prompt)
            suffix, acc, tok_s = _run_draft1(model_s, cfg_s, ids)
            details[(k, i)] = suffix
            dense_suffix = details[(None, i)]
            n = min(len(suffix), len(dense_suffix))
            overlap = (sum(1 for a, b in zip(suffix[:n], dense_suffix[:n])
                           if a == b) / n) if n else 1.0
            per_prompt.append((acc, tok_s, overlap))
        med_toks = statistics.median(x[1] for x in per_prompt)
        rows.append({
            "model": "helioslm-toy-8.5M",
            "sparse_top_k": k,
            "indexer": indexer,
            "draft_depth": 1,
            "acceptance": round(statistics.mean(x[0] for x in per_prompt),
                                4),
            "tok_s_out": round(med_toks, 3),
            "gain_vs_dense_draft0": round(med_toks / base_toks - 1.0, 3),
            "n_tokens": max_new,
            "output_overlap_vs_dense": round(
                statistics.mean(x[2] for x in per_prompt), 4),
        })
    return rows, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/toy_v5.13.pt")
    ap.add_argument("--ks", default="4,16,1000000",
                    help="comma-separated sparse_top_k values "
                         "(10**6 = effectively dense, the oracle row)")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--indexer", default="head_mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    if not os.path.exists(args.checkpoint):
        print(f"skipped (no checkpoint at {args.checkpoint})")
        return
    rows, _ = sweep_sparse_mtp(args.checkpoint, ks, args.max_new,
                               args.iters, args.indexer)

    print("\n=== sparse top-k x MTP acceptance sweep (P3-superset JSONL) ===")
    for r in rows:
        print(json.dumps(r))

    out = args.out or ("sweep_sparse_mtp_"
                       + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
