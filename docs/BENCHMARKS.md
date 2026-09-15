# HeliosLM Benchmarks

All numbers reproducible on CPU:

```bash
python benchmarks/bench_cpu.py                          # lite: analytic + wall-clock
python benchmarks/bench_cpu.py --size full --analytic-only   # full: analytic only
python benchmarks/bench_cpu.py --chart out.png          # + KV-cache chart
```

Results are written to `benchmarks/results_<date>.json` so trends can be
tracked across versions. Suite introduced in **v5.10** (2026-09-15).

## KV-cache memory (analytic structural counts, bf16 unless noted)

| Config | seq_len | HeliosLM | MHA reference | savings |
|---|---|---|---|---|
| full (48L, hybrid MLA+GDA) | 1k | 52.0 MB | 2013.3 MB | 97.4% |
| full | 8k | 159.4 MB | 16.1 GB | 99.0% |
| full | 64k | 1018.2 MB | 128.8 GB | 99.2% |
| full | 128k | 2000 MB | 257.7 GB | **99.2%** |
| lite (2L, MLA only) | 128k | 37.7 MB | 134.2 MB | 71.9% |

**Why hybrid savings grow with context:** Gated-Delta layers hold a fixed
recurrent state (`linear_num_heads × head_dim² × bf16` = 32×128×128×2 ≈ 1 MB
per layer) regardless of sequence length — only the 12 MLA layers (48 layers,
`full_attention_every=4`) scale with `seq_len`. At 128k, 36 of 48 layers
contribute zero per-token cache.

MLA absorbed mode stores `kv_latent_dim + rope_head_dim` = 512 + 64 values per
token per MLA layer (vs `num_heads × (head_dim + v_head_dim)` = 64×320 for the
MHA reference). With `kv_cache_dtype="fp8"` the MLA side halves again.

## Generation on CPU (lite, 8.5M params, greedy)

| phase | time | throughput |
|---|---|---|
| prefill (32 tok) | 0.011 s | 2938 tok/s |
| decode (31 tok) | 0.176 s | 176 tok/s |

Lite config is a smoke-test scale; numbers exist to catch regressions, not to
represent the full config (which cannot run wall-clock benchmarks on CPU).

## Methodology notes

- **Analytic counts are exact** — derived from config dims, not sampling.
- Prefill time = wall time of a 1-token generation pass; decode = remaining
  time of a 32-token pass (approximation documented in `bench_cpu.py`).
- MHA reference = every head caches K (`head_dim`) + V (`v_head_dim`) at
  bf16 — the design MLA replaces.
