"""Sparse top-k x vLLM-style ENGINE end-to-end serving measurement
(limitations round 2026-10-08).

bench_sparse_mtp.py (2026-10-06) measures sparse top-k through the
MTPDecoder draft loop. The other half of the residual "sparse x
MTP/engine acceptance" item is the SERVING engine: continuous batching,
watermark-aligned padded prefills and batched [B, 1] decode steps. This
script measures that half on the toy checkpoint:

    solo_prefix_match   fraction of requests whose engine output is an
                        exact prefix of the SAME model's solo greedy
                        reference (the serving-correctness oracle; the
                        engine stops at EOS while generate() freezes,
                        so prefix — not equality — is the contract)
    tok_s_out           measured wall-clock output tokens/s of the whole
                        engine run (CPU, batched)
    gain_vs_dense_engine  tok/s vs the dense engine row
    output_overlap_vs_dense_engine  per-request fraction of generated
                        positions matching the DENSE engine run (sparse
                        top-k is an APPROXIMATION — divergence is
                        reported, never hidden)
    watermark_pad_requests  how many requests were admitted with a
                        watermark pad prefix (prompt_pad > 0). The pad
                        K/V entries are masked out of the sparse top-k
                        selection (mla.py masked_fill before topk), so
                        padded batched decode must stay exact — this
                        column proves the padded path was exercised.

ENGINE DRAFT-PATH FINDING (loud, 2026-10-08): VLLMEngine has NO
draft/MTP/speculative interface — decoding is one trunk forward per
step per cache-length group (vllm_engine.py step/_decode_batch). A
combined engine x draft measurement would require adding a draft path
to the engine first; until then the draft half of the residual item
lives in bench_sparse_mtp.py (MTPDecoder) and cannot be re-expressed
here. test_engine_sparse_serving asserts the absence structurally so
this note cannot silently go stale.

Rows follow the colibri-P3-compatible style of bench_sparse_mtp.py:

    {"model", "engine", "sparse_top_k", "indexer", "n_requests",
     "watermark_pad_requests", "tok_s_out", "gain_vs_dense_engine",
     "n_tokens", "output_overlap_vs_dense_engine", "solo_prefix_match"}

Oracle / honesty notes:
- ``sparse_top_k >= kv_len`` never activates the sparse path, so the
  k >= L engine row MUST match the dense engine row on output ids
  bitwise (enforced by test_engine_sparse_serving).
- With k < kv_len identical outputs are a (legitimate, measured)
  property of THIS checkpoint's greedy margins, not a guarantee — the
  schema reports overlap so any divergence shows up as data.
- ``indexer="learned"`` rows are refused loudly unless the checkpoint
  carries indexer.* weights (same gate as bench_sparse_mtp.py — an
  untrained indexer produces noise, not signal).

Usage:
    python benchmarks/bench_engine_sparse.py \
        --checkpoint checkpoints/toy_v5.13.pt --ks 4,16 --reqs 4
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_sparse_mtp import PROMPTS, _build_model, encode
from helioslm_v5.src.inference.vllm_engine import VLLMEngine

SCHEMA_KEYS = (
    "model", "engine", "sparse_top_k", "indexer", "n_requests",
    "watermark_pad_requests", "tok_s_out", "gain_vs_dense_engine",
    "n_tokens", "output_overlap_vs_dense_engine", "solo_prefix_match",
)

ENGINE_NAME = "vllm_watermark"


def _engine_run(model, config, prompts_ids, max_new):
    """One timed engine run; returns (outputs, n_padded, tok_s).

    outputs[i] is the generated id list for prompt i (submission order);
    n_padded counts requests admitted with a watermark pad prefix.
    """
    engine = VLLMEngine(model, config, block_size=8, max_num_blocks=256)
    req_ids = [engine.add_request(p, max_new_tokens=max_new,
                                  temperature=0.0)
               for p in prompts_ids]
    t0 = time.perf_counter()
    results = engine.run()
    dt = time.perf_counter() - t0
    outputs = [results[rid] for rid in req_ids]
    n_padded = sum(1 for r in engine.finished_requests if r.prompt_pad > 0)
    n_tokens = sum(len(o) for o in outputs)
    return outputs, n_padded, (n_tokens / dt if dt > 0 else 0.0)


def _solo_prefix_match(model, prompts_ids, outputs, max_new):
    """Fraction of requests whose engine output is an exact prefix of the
    same model's solo greedy reference (EOS-stop vs freeze contract)."""
    n_ok = 0
    with torch.no_grad():
        for ids, got in zip(prompts_ids, outputs):
            ref = model.generate(torch.tensor([ids]),
                                 max_new_tokens=max_new, temperature=0)
            ref_new = ref[0, len(ids):].tolist()
            if ref_new[:len(got)] == got and len(got) >= 1:
                n_ok += 1
    return n_ok / max(1, len(outputs))


def sweep_engine_sparse(checkpoint, ks, max_new=16, n_reqs=4,
                        indexer="head_mean"):
    """Run the engine-level sparse serving sweep; returns (rows, details).

    ``details[(k, i)]`` holds the engine-generated id list of request i
    (``k is None`` = dense engine row), so tests can assert the k >= L
    bitwise oracle and determinism without trusting wall-clock fields.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"engine sparse sweep needs the toy checkpoint; not found: "
            f"{checkpoint}"
        )
    if indexer not in ("head_mean", "learned"):
        raise ValueError(
            f"indexer must be 'head_mean' or 'learned', got {indexer!r}"
        )
    if not isinstance(n_reqs, int) or n_reqs < 2:
        raise ValueError(
            f"n_reqs must be an integer >= 2 (batched serving with "
            f"watermark padding is the point of this sweep), got "
            f"{n_reqs!r}"
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

    # Distinct prompt lengths in the first admission wave are what makes
    # the watermark pad prefix (and its masked top-k interaction) actually
    # fire; PROMPTS entries all differ in length.
    prompts_ids = [encode(PROMPTS[i % len(PROMPTS)]).tolist()[0]
                   for i in range(n_reqs)]

    rows = []
    details = {}

    # Dense engine baseline.
    model_d, cfg_d = _build_model(state_dict, None, indexer)
    out_d, pad_d, toks_d = _engine_run(model_d, cfg_d, prompts_ids, max_new)
    for i, o in enumerate(out_d):
        details[(None, i)] = o
    rows.append({
        "model": "helioslm-toy-8.5M",
        "engine": ENGINE_NAME,
        "sparse_top_k": None,
        "indexer": indexer,
        "n_requests": n_reqs,
        "watermark_pad_requests": pad_d,
        "tok_s_out": round(toks_d, 3),
        "gain_vs_dense_engine": None,
        "n_tokens": max_new,
        "output_overlap_vs_dense_engine": None,
        "solo_prefix_match": round(
            _solo_prefix_match(model_d, prompts_ids, out_d, max_new), 4),
    })

    # Sparse engine rows.
    for k in ks:
        model_s, cfg_s = _build_model(state_dict, k, indexer)
        out_s, pad_s, toks_s = _engine_run(
            model_s, cfg_s, prompts_ids, max_new)
        overlaps = []
        for i, o in enumerate(out_s):
            details[(k, i)] = o
            ref = details[(None, i)]
            n = min(len(o), len(ref))
            overlaps.append(
                (sum(1 for a, b in zip(o[:n], ref[:n]) if a == b) / n)
                if n else 1.0)
        rows.append({
            "model": "helioslm-toy-8.5M",
            "engine": ENGINE_NAME,
            "sparse_top_k": k,
            "indexer": indexer,
            "n_requests": n_reqs,
            "watermark_pad_requests": pad_s,
            "tok_s_out": round(toks_s, 3),
            "gain_vs_dense_engine": round(toks_s / toks_d - 1.0, 3),
            "n_tokens": max_new,
            "output_overlap_vs_dense_engine": round(
                sum(overlaps) / len(overlaps), 4),
            "solo_prefix_match": round(
                _solo_prefix_match(model_s, prompts_ids, out_s, max_new),
                4),
        })
    return rows, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/toy_v5.13.pt")
    ap.add_argument("--ks", default="4,16,1000000",
                    help="comma-separated sparse_top_k values "
                         "(10**6 = effectively dense, the oracle row)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--reqs", type=int, default=4)
    ap.add_argument("--indexer", default="head_mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    if not os.path.exists(args.checkpoint):
        print(f"skipped (no checkpoint at {args.checkpoint})")
        return
    rows, _ = sweep_engine_sparse(args.checkpoint, ks, args.max_new,
                                  args.reqs, args.indexer)

    print("\n=== sparse top-k x ENGINE serving sweep (P3-style JSONL) ===")
    for r in rows:
        print(json.dumps(r))

    out = args.out or ("sweep_engine_sparse_"
                       + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
