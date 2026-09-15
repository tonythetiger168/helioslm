"""HeliosLM CPU benchmark suite (v5.10).

Measures two things, dependency-free beyond torch:

1. **Analytic KV-cache memory** — bytes per token (MLA absorbed vs the
   expanded/MHA reference) plus the fixed recurrent state of hybrid
   Gated-Delta layers. These are exact structural counts derived from the
   config, not estimates.
2. **Wall-clock generation** — prefill and decode tokens/s on CPU for the
   ``lite`` config (and optional custom overrides), greedy decoding.

Usage:
    python benchmarks/bench_cpu.py                # table + JSON results
    python benchmarks/bench_cpu.py --chart out.png  # + KV-cache chart

Results are written to ``benchmarks/results_<date>.json`` so trends can be
tracked across versions (run weekly — see the repo's scheduled review).
"""

import argparse
import datetime
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

DTYPE_BYTES = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
}


# --------------------------------------------------------------------------
# Analytic cache accounting
# --------------------------------------------------------------------------

def _is_gda_layer(cfg: HeliosLMv5Config, layer_idx: int) -> bool:
    hy = cfg.hybrid_attention
    if not hy.enabled:
        return False
    return not (layer_idx == 0 or layer_idx % hy.full_attention_every
                == hy.full_attention_every - 1)


def mla_bytes_per_token(cfg: HeliosLMv5Config) -> int:
    """Absorbed-mode MLA cache: c_kv latent + shared k_rope, per layer."""
    a = cfg.attention
    elt = 1 if a.kv_cache_dtype == "fp8" else 2  # bf16 default
    if a.use_absorption:
        return (a.kv_latent_dim + a.rope_head_dim) * elt
    # expanded fallback: per-head K_nope + V + shared rope
    return (a.num_attention_heads * (a.no_rope_head_dim + a.v_head_dim)
            + a.rope_head_dim) * elt


def mha_reference_bytes_per_token(cfg: HeliosLMv5Config) -> int:
    """MHA reference: every head caches K (head_dim) + V (v_head_dim)."""
    a = cfg.attention
    return (a.num_attention_heads * (a.head_dim + a.v_head_dim)) * 2


def gda_state_bytes(cfg: HeliosLMv5Config) -> int:
    """Fixed recurrent state per hybrid Gated-Delta layer (per sequence).

    Delta-rule state is (head_dim x head_dim) per linear head.
    """
    hy = cfg.hybrid_attention
    return hy.linear_num_heads * hy.linear_head_dim * hy.linear_head_dim * 2


def cache_bytes(cfg: HeliosLMv5Config, seq_len: int) -> dict:
    """Total cache bytes at ``seq_len`` for this config."""
    hy = cfg.hybrid_attention
    n_layers = cfg.num_hidden_layers
    n_gda = sum(1 for i in range(n_layers) if _is_gda_layer(cfg, i))
    n_mla = n_layers - n_gda
    mla_total = mla_bytes_per_token(cfg) * n_mla * seq_len
    gda_total = gda_state_bytes(cfg) * n_gda if n_gda else 0
    mha_total = mha_reference_bytes_per_token(cfg) * n_layers * seq_len
    return {
        "mla_layers": n_mla,
        "gda_layers": n_gda,
        "helioslm_bytes": mla_total + gda_total,
        "mha_reference_bytes": mha_total,
        "savings_vs_mha": 1.0 - (mla_total + gda_total) / mha_total,
    }


# --------------------------------------------------------------------------
# Wall-clock generation
# --------------------------------------------------------------------------

def bench_generation(cfg: HeliosLMv5Config, prompt_len: int = 32,
                     new_tokens: int = 32, warmup: int = 1) -> dict:
    """Greedy generate on CPU; split prefill vs decode wall time."""
    model = HeliosLMv5(cfg)
    model.eval()
    prompt = torch.randint(0, min(cfg.vocab_size, 32000), (1, prompt_len))

    with torch.no_grad():
        for _ in range(warmup):
            model.generate(prompt[:, :8], max_new_tokens=4, temperature=0)

        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=new_tokens, temperature=0)
        total = time.perf_counter() - t0

    n_prompt = prompt.shape[1]
    n_new = out.shape[1] - n_prompt
    # Prefill ~= time to first token is not separately instrumented; we
    # approximate split by a second run stopping after 1 token.
    with torch.no_grad():
        t0 = time.perf_counter()
        model.generate(prompt, max_new_tokens=1, temperature=0)
        prefill = time.perf_counter() - t0
    decode = max(total - prefill, 1e-9)
    params = sum(p.numel() for p in model.parameters())

    return {
        "prompt_len": n_prompt,
        "new_tokens": n_new,
        "params_M": round(params / 1e6, 2),
        "param_bytes_MB": round(params * 4 / 1e6, 1),
        "prefill_s": round(prefill, 3),
        "prefill_tok_s": round(n_prompt / prefill, 1),
        "decode_s": round(decode, 3),
        "decode_tok_s": round((n_new - 1) / decode, 2),
        "total_s": round(total, 3),
    }


# --------------------------------------------------------------------------
# Chart (optional; requires matplotlib)
# --------------------------------------------------------------------------

def plot_kv_cache(cfg: HeliosLMv5Config, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [1024, 4096, 16384, 65536, 131072]
    helios, mha = [], []
    for L in seqs:
        r = cache_bytes(cfg, L)
        helios.append(r["helioslm_bytes"] / 1e6)
        mha.append(r["mha_reference_bytes"] / 1e6)

    x = range(len(seqs))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], mha, width=0.4, label="MHA reference",
           color="#888899")
    ax.bar([i + 0.2 for i in x], helios, width=0.4,
           label=f"HeliosLM ({'hybrid MLA+GDA' if cfg.hybrid_attention.enabled else 'MLA'})",
           color="#4c9be0")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{L//1024}k" for L in seqs])
    ax.set_ylabel("KV cache (MB)")
    ax.set_xlabel("sequence length")
    ax.set_title(f"HeliosLM v5.10 KV-cache memory vs MHA "
                 f"({cfg.size} config, {cfg.num_hidden_layers} layers)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="lite", choices=["lite", "full"])
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--chart", default=None, help="write KV-cache chart PNG")
    ap.add_argument("--analytic-only", action="store_true",
                    help="skip wall-clock generation (use for the 'full' size)")
    ap.add_argument("--out", default=None, help="override JSON output path")
    args = ap.parse_args()

    cfg = HeliosLMv5Config(size=args.size)
    print(f"HeliosLM {cfg.model_name}  (size={cfg.size}, "
          f"{cfg.num_hidden_layers} layers, hybrid={cfg.hybrid_attention.enabled})")
    print("=" * 72)

    # --- analytic cache table -------------------------------------------
    print(f"{'seq_len':>8} | {'HeliosLM':>12} | {'MHA ref':>12} | {'savings':>8}")
    print("-" * 72)
    cache_rows = {}
    for L in (1024, 8192, 65536, 131072):
        r = cache_bytes(cfg, L)
        cache_rows[L] = r
        print(f"{L:>8} | {r['helioslm_bytes']/1e6:>10.1f}MB | "
              f"{r['mha_reference_bytes']/1e6:>10.1f}MB | "
              f"{r['savings_vs_mha']*100:>7.1f}%")

    # --- wall clock -------------------------------------------------------
    gen = None
    if not args.analytic_only:
        print("-" * 72)
        gen = bench_generation(cfg, args.prompt_len, args.new_tokens)
        print(f"params: {gen['params_M']}M ({gen['param_bytes_MB']}MB fp32)")
        print(f"prefill: {gen['prefill_s']}s  ({gen['prefill_tok_s']} tok/s, "
              f"{gen['prompt_len']} tokens)")
        print(f"decode:  {gen['decode_s']}s  ({gen['decode_tok_s']} tok/s, "
              f"{gen['new_tokens'] - 1} tokens)")
    print("=" * 72)

    result = {
        "date": datetime.date.today().isoformat(),
        "version": cfg.model_name,
        "size": cfg.size,
        "cache": {str(k): v for k, v in cache_rows.items()},
        "generation": gen,
    }
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{datetime.date.today().isoformat()}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results -> {out}")

    if args.chart:
        plot_kv_cache(cfg, args.chart)
        print(f"chart   -> {args.chart}")


if __name__ == "__main__":
    main()
