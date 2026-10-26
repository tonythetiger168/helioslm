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


## Fused vs unfused AWQ/GPTQ/MXFP4 benchmark (Issue #2)

`benchmarks/bench_quant.py` times one quantized linear layer with the fused
group-wise dequant x matmul (`mod.fused = True`, v5.11) against the
unfused path that dequantizes the whole weight first (`mod.fused = False`).
Each method and setting runs in its **own process**. Before timing, the
fused output is checked against the unfused output on the same quantized
weights (`atol = rtol = 1e-4`); a mismatch aborts the run.

```bash
# Laptop-sized CPU smoke run
python benchmarks/bench_quant.py --in-features 256 --out-features 512 --repeats 3 --out /tmp/quant_smoke.json

# The measured runs below
python benchmarks/bench_quant.py --in-features 4096 --out-features 14336 --group-size 128 \
    --batch 1 --warmup 3 --repeats 11 --out benchmarks/quant_fused_b1_2026-10-21.json
```

**Measured 2026-10-21** (4096 x 14336 projection, AWQ / GPTQ group 128,
MXFP4 block 32; 4-vCPU Intel Xeon @ 2.80GHz, Python 3.13, torch 2.14 CPU;
median of 11 runs after 3 warm-up runs; artifacts
`benchmarks/quant_fused_b{1,16}_2026-10-21.json`):

| Method | Batch | Unfused (ms) | Fused (ms) | Speedup | Max abs error vs unfused |
|---|---|---|---|---|---|
| AWQ | 1 | 926 | 125 | **7.4x** | 7.7e-7 |
| GPTQ | 1 | 1245 | 104 | **11.9x** | 8.3e-7 |
| MXFP4 | 1 | 1604 | 708 | **2.3x** | 1.1e-6 |
| AWQ | 16 | 722 | 194 | **3.7x** | 7.7e-7 |
| GPTQ | 16 | 1112 | 159 | **7.0x** | 8.3e-7 |
| MXFP4 | 16 | 1534 | 738 | **2.1x** | 1.1e-6 |

- The fused path is faster for every method at both batch sizes. It
  dequantizes one group at a time and never materialises the full
  weight; the unfused path builds the whole fp32 weight on every call.
- The gain shrinks at batch 16: the matmul does more work per call, so
  the dequantization cost the fused path saves is a smaller share.
- MXFP4 gains least. Its fused path takes 4096 / 32 = 128 block steps
  per call, each a code lookup plus a 32-column matmul; AWQ and GPTQ
  take 32 steps of 128 columns.
- Noise: the container is shared. The batch-1 unfused AWQ samples range
  from 705 to 2674 ms (see `samples_ms`); the medians are robust to this,
  and the fastest unfused sample is still 4.6x the slowest fused one.
- Peak RSS is about the same for both paths (AWQ / GPTQ ~1.4-1.5 GiB,
  MXFP4 ~3.6 GiB): the process peak is set by building the quantized layer
  from an fp32 weight, not by the forward pass.
- Scope: CPU and PyTorch only. This says nothing about GPU kernels
  (auto-gptq / exllama / Marlin) or FP8.
- Checked by `helioslm_v5/tests/test_bench_quant.py`: each median follows
  from its samples, every parity error is <= 1e-4, and every speedup > 1.5.

## Multi-seed A/B runs (Issue #55)

Every `examples/*_ab*.py` script accepts `--seeds` in place of `--seed`:

```bash
# legacy single-seed payload (unchanged)
python examples/smebu_balance_ab.py --seed 0 --output /tmp/smebu_s0.json

# paired multi-seed run: every arm runs once per seed
python examples/smebu_balance_ab.py --seeds 0 1 2 --output /tmp/smebu_3seeds.json
```

- `--seed` and `--seeds` are mutually exclusive; seeds must be distinct
  non-negative integers.
- The multi-seed payload holds `per_seed` (each seed's full single-seed
  payload, bit-identical to a `--seed` run), `aggregate` (numbers that
  differ across seeds as mean / sample std / min / max / n),
  `environment` and `per_seed_results_sha256`.
- Paired differences between arms are taken within a seed from
  `per_seed`.
- `csa_ab_passkey.py --check` stays single-seed (it compares the
  committed artifact).
- The committed A/B artifacts are single-seed. A multi-seed claim needs
  its own `--seeds` artifact.

