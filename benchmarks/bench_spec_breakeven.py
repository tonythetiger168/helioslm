"""Speculative-decoding break-even sweep (v5.16, roadmap #6 component 2).

Measures MTP draft net gain across cache states and emits rows in the
exact JSONL schema proposed to the colibri project (issue P3), so numbers
from the two engines are directly comparable:

    {"model", "draft_depth", "cache_state", "acceptance",
     "tok_s_out", "gain_vs_draft0", "n_tokens"}

Also measures the expert-cache side of the break-even curve (Part B):
hit rate of the v5.15 disk-tier expert store at varying resident budgets —
the x-axis that decides whether drafting pays.

Usage:
    python benchmarks/bench_spec_breakeven.py \
        --checkpoint checkpoints/toy_v5.13.pt --iters 3 --out sweep.jsonl

Part A requires the toy checkpoint (MTP head trained via aux loss).
Part B builds a small synthetic MoE (no checkpoint needed).
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
from helioslm_v5.src.inference.expert_store import (attach_streaming_store,
                                                    detach_streaming_store)
from helioslm_v5.src.inference.mtp import MTPDecoder
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

PROMPTS = [
    "HeliosLM", "The model", "import torch", "def forward",
    "# v5.16", "Attention(", "KV cache", "speculative",
]
CACHE_STATES = ["cold", "warm"]


def encode(prompt, vocab=1024):
    return torch.tensor([[ord(c) for c in prompt if ord(c) < vocab]])


def bench_toy_model(ckpt_path, max_new, iters):
    """Part A: draft on/off x cache states on the toy checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    base_times = {}   # cache_state -> median tok/s with draft off
    for cache_state in CACHE_STATES:
        # warm the session prefix (a stand-in for .coli_usage warm state)
        warm_ids = encode("HeliosLM is a pure PyTorch implementation of")
        with torch.no_grad():
            model(warm_ids)
        for depth, use_draft in ((0, False), (1, True)):
            tok_s, acc = [], []
            for it in range(iters):
                prompt = PROMPTS[(it + depth) % len(PROMPTS)]
                ids = encode(prompt)
                t0 = time.perf_counter()
                if use_draft:
                    dec = MTPDecoder(model, model.mtp_modules,
                                     HeliosLMv5Config(size="lite"))
                    r = dec.generate(ids, max_new_tokens=max_new, temperature=0)
                    n = len(r.output_token_ids) if hasattr(r, "output_token_ids") \
                        else max_new
                    acc.append(r.acceptance_rate)
                else:
                    out = model.generate(ids, max_new_tokens=max_new,
                                         temperature=0)
                    n = out.shape[1] - ids.shape[1]
                dt = time.perf_counter() - t0
                tok_s.append(n / dt)
            med = statistics.median(tok_s)
            if depth == 0:
                base_times[cache_state] = med
            rows.append({
                "model": "helioslm-toy-8.5M",
                "draft_depth": depth,
                "cache_state": cache_state,
                "acceptance": (statistics.mean(acc) if acc else None),
                "tok_s_out": round(med, 3),
                "gain_vs_draft0": round(med / base_times[cache_state] - 1.0, 3)
                if cache_state in base_times else None,
                "n_tokens": max_new,
            })
    return rows


def bench_expert_hit_curve(budgets):
    """Part B: disk-tier store hit rate vs resident budget (the x-axis)."""
    torch.manual_seed(1616)
    cfg = HeliosLMv5Config(size="lite")
    cfg.hidden_size = 32
    cfg.moe.num_experts = 8
    cfg.moe.num_shared_experts = 0
    cfg.moe.num_activated_experts = 2
    cfg.moe.expert_hidden_size = 64
    moe = DeviceLimitedMoE(cfg).eval()

    expert_bytes = 64 * 32 * 3 * 4   # one expert fp32 (w13 + w2)
    rows = []
    for n_resident in budgets:
        st = attach_streaming_store(moe, budget_bytes=n_resident * expert_bytes)
        for trial in range(8):
            xt = torch.randn(2, 5, 32,
                             generator=torch.Generator().manual_seed(100 + trial))
            moe(xt)
        s = st.stats()
        rows.append({
            "resident_experts": n_resident,
            "budget_bytes": n_resident * expert_bytes,
            "hit_rate": round(s["hit_rate"], 3),
            "misses": s["misses"],
            "evictions": s["evictions"],
            "ram_saving_vs_dense": round(s["ram_saving_vs_dense"], 3),
        })
        detach_streaming_store(moe, st)
    return rows



def best_draft_per_cache_state(rows):
    """Pure: pick the draft depth with the highest tok_s_out per cache_state.

    Rows follow the colibri-P3 JSONL schema. Deterministic — unit-testable
    without wall timing.
    """
    best = {}
    for r in rows:
        cs = r["cache_state"]
        if cs not in best or r["tok_s_out"] > best[cs]["tok_s_out"]:
            best[cs] = r
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/toy_v5.13.pt")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budgets", default="1,2,4,8",
                    help="resident-expert counts for Part B")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Part A skipped (no checkpoint at {args.checkpoint})")
        part_a = []
    else:
        part_a = bench_toy_model(args.checkpoint, args.max_new, args.iters)

    part_b = bench_expert_hit_curve([int(b) for b in args.budgets.split(",")])

    print("\n=== Part A: MTP draft net gain (colibri-P3 JSONL schema) ===")
    for r in part_a:
        print(json.dumps(r))
    print("\n=== Part B: expert-store hit rate vs residency budget ===")
    for r in part_b:
        print(json.dumps(r))

    out = args.out or "sweep_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"
    with open(out, "w") as f:
        for r in part_a + part_b:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
