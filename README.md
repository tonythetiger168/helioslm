# HeliosLM — A Hackable DeepSeek-V3/K3-Style LLM Stack in Pure PyTorch

A from-scratch PyTorch reference implementation of a modern LLM stack: MLA attention with weight absorption, sigmoid-gated MoE with auxiliary-loss-free load balancing, hybrid linear attention, speculative decoding, FP8 training, a DualPipe schedule simulation, and a vLLM-style serving engine. Built to be **read, modified, and verified** — every core path is unit-tested and many are checked with bitwise-equivalence tests. Everything runs on CPU.

> **One-liner:** If you want to understand (or hack on) how DeepSeek-V3/K3-class models actually work — without needing a GPU cluster first — this repo is for you.

[![CI](https://github.com/tonythetiger168/helioslm/actions/workflows/ci.yml/badge.svg)](https://github.com/tonythetiger168/helioslm/actions)
![Tests](https://img.shields.io/badge/tests-48%20unit%20%2B%209%20integration-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch%20(pure)-ee4c2c)

![Demo](docs/demo.gif)

## Who is this for?

| You are... | What HeliosLM gives you |
|---|---|
| **A learner** who wants to understand MLA, MoE routing, DualPipe, speculative decoding | Annotated, review-hardened PyTorch with 48 unit tests that act as executable documentation |
| **A researcher** who wants a stack to modify, ablate, and extend quickly | Single-process, CPU-iterable training + serving code — change one file, run one test |
| **A practitioner** evaluating serving/quantization techniques | vLLM-style paged engine, GPTQ/AWQ/FP8/MXFP4 quantization, MTP speculative decoding — all inspectable |

**Honest positioning:** this is a correctness-focused reference implementation, not a throughput-optimized production engine (see [Known Limitations](#known-limitations)).

## Quick Start

```bash
pip install torch
python -m helioslm_v5.tests.test_v5     # 48 unit tests
python integration_test_v51.py          # 9 end-to-end integration tests
```

```python
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

model = HeliosLMv5(HeliosLMv5Config(size="lite"))   # CPU-friendly
out = model.generate([[1, 2, 3]], max_new_tokens=20, temperature=0)
print(out)
```

<!-- TODO: paste actual generate() output here, e.g.:
```
generated: [[1, 2, 3, ...]]
```
Nothing sells an LLM repo like showing it produce tokens. -->

## Capabilities at a glance

| Area | Implementation |
| --- | --- |
| **Attention** | MLA with **weight absorption** — latent-only KV cache, **−97.7% memory vs MHA** (full config), verified equivalent to the expanded path (<1e-4). Hybrid linear attention: Gated Delta Rule layers interleaved with MLA, fixed-size recurrent state cache, decode ≡ one-shot (<2e-7). RoPE scaling: linear / NTK / YaRN. DSA-style sparse top-k decode over the latent cache (k≥L exactly dense). Sliding-window attention with StreamingLLM sinks (O(W) decode), per-head QK-norm, Gemma-style logit soft-capping. |
| **MoE** | Sigmoid-gated fine-grained experts with **auxiliary-loss-free** load balancing (selection-only bias, heuristic or quantile updates). **LatentMoE**: routed experts in a shared latent space. **SiTU-GLU** tanh soft-capped activation. |
| **Cross-layer** | **Attention Residuals** — per-layer gated injection of accumulated lower-layer attention outputs, threaded through DualPipe (gradient-exact, bitwise-verified). |
| **Speculative decoding** | DeepSeek-style MTP with **strict verification** (residual (p−q)₊ resampling), batch support, O(1) cache-truncation rollback; hybrid recurrent-state rollback via restore+replay. |
| **Serving** | vLLM-style engine: paged KV accounting, copy-on-write forks, watermark-aligned continuous batching. |
| **Training** | FP8 trainer (native float8 + STE, E5M2 gradient hooks, AdamW master weights), DualPipe schedule simulation (recompute-based, gradient-exact), GRPO (real sampling, k3 KL, answer-extraction rewards), Muon optimizer (Newton–Schulz orthogonalized momentum, optional per-head blocks), QAT straight-through fake-quant training. |
| **Certified sparsity** | `attention/certified_sparse_decode.py`: top-δ decode with a runtime output certificate; pseudo-max gap g detected per step — FFD's empirical Fig-5 assumption becomes an executable check |
| **Adaptation** | `AdaptationLoop`: transferred harnesses must be re-accepted under the target workload's gate or dropped; cold/prefix-free targets shed draft+pool, matching the v5.16 break-even data |
| **Harness evolution** | `inference/harness_evolver.py`: ModularRSI-style module-wise search over draft/pool/tier configs; the temp-0 bitwise gate makes 'latency evolves, answers never change' an enforced invariant (1.41x modeled speedup, 0 gate rejects) |
| **Prefix pool** | `inference/prefix_pool.py`: blake2b(token-block + config fingerprint) keyed KV snapshots, LRU, exact-length past; pooled greedy == from-scratch bitwise |
| **Stream scoring** | `eval/score_stream.py`: score any engine's (prompt, output) JSONL under a reference model; A/B compare with bootstrap CI — the audit-side complement to serving engines |
| **Spec telemetry** | `bench_spec_breakeven.py`: draft x cache-state sweep in the colibri-P3 schema; acceptance + expert hit-rate per decode context, `best_draft_per_cache_state()` picker |
| **Expert streaming** | `expert_store.py`: routed experts tiered to a memory-mapped file, LRU residency with hit/miss/eviction telemetry; streaming forward is bitwise-identical to dense (oracle-verified, roadmap #6) |
| **Toy checkpoint** | `checkpoints/toy_v5.13.pt` — 8.5M char-level model trained on the repo's own source in ~10 CPU-minutes (`examples/train_toy_checkpoint.py`); `generate()` / harness / MTP run against trained weights |
| **Quantization** | **True GPTQ** (Hessian OBS with error compensation, optional act-order), AWQ with activation-aware grid search, native FP8, MXFP4 — all with `from_linear` real-weight packing. |
| **Eval** | Log-likelihood harness (`helioslm_v5/eval/harness.py`): `loglikelihood` / `multiple_choice` / `run_harness` + built-in synthetic tasks (v5.11), token-id based, lm-eval-harness spirit |
| **Multimodal** | NaViT vision encoder (row/col position decomposition, mixed-resolution packing), streaming audio encoder (causal, sliding-window memory, bit-equivalent to one-shot). |

## Why HeliosLM vs. alternatives?

| | HeliosLM | `transformers` | `vLLM` | nanoGPT-style |
|---|---|---|---|---|
| Purpose | Understand + hack the full stack | Run pre-trained models | Max serving throughput | Learn basics |
| Runs on CPU end-to-end | ✅ | partial | ❌ | ✅ |
| Training + serving + quantization in one repo | ✅ | ❌ | ❌ | ❌ |
| Bitwise/strict correctness checks on core paths | ✅ | — | — | — |
| Production throughput | ❌ (by design) | — | ✅ | ❌ |

## Benchmarks

![KV cache](docs/benchmarks_kv_cache.png)

The full config holds **99.2% less KV-cache memory than MHA** at 128k context
(2.0 GB vs 257.7 GB): 36 of 48 layers are Gated-Delta linear attention with a
fixed ~1 MB recurrent state, and the 12 MLA layers store only the compressed
latent (512 + 64 values/token). Full numbers and methodology:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) — reproducible on CPU via
`python benchmarks/bench_cpu.py`.

## Repository layout

```
helioslm_v5/          # source (configs, src/{attention,moe,inference,training,vision,audio,quantization}, tests)
docs/                 # code review reports + per-version fix reports
integration_test_v51.py
CHANGELOG.md          # full version history (v5.0 → v5.9)
```

## Roadmap

See the [GitHub Project board](https://github.com/tonythetiger168/helioslm/projects) for the live plan. Highlights:

- [ ] Pre-trained toy checkpoint (small corpus, few hours of training) so people can `load` and chat immediately
- [ ] Fused quantization kernels
- [ ] CUDA end-to-end verification (paths are currently static-checked; CPU-verified)
- [ ] Hugging Face Hub integration for configs/checkpoints
- [ ] Example notebooks: "Train a tiny HeliosLM on your laptop" / "Add a new attention variant in 30 lines"

## Contributing

Contributions are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Issues labeled [`good first issue`](https://github.com/tonythetiger168/helioslm/labels/good%20first%20issue) are the best entry points.

## Known Limitations

All five previously known limitations are resolved as of v5.14. Remaining
hardware-dependent item: CUDA end-to-end verification (CPU-verified paths
are static-checked) — tracked as a community issue.

## Version history

Headlines (full details in [CHANGELOG.md](helioslm_v5/CHANGELOG.md)):

- **v5.21** — Certified top-δ sparse decode: per-step output bound + pseudo-max gap detection (ICML 2026 FFD's empirical assumption, made executable)
- **v5.20** — Pareto-aware integration (memory axis) + cross-workload adaptation loop — ModularRSI gaps 2/3 closed at the inference layer
- **v5.19** — Evolvable serving harness: ModularRSI-style module-wise search (draft/pool/tier) behind a deterministic bitwise oracle gate
- **v5.18** — Content-addressed KV prefix pool: cross-session prefix reuse, fingerprint-guarded, bit-exact oracle (roadmap #6 complete)
- **v5.17** — Standalone token-stream scorer: quality-gate any engine's output (A/B compare + bootstrap CI, colibri-container-style)
- **v5.16** — Speculation break-even sweep: colibri-P3-compatible JSONL schema, first honest data point (draft pays only when warm)
- **v5.15** — Disk-tier expert store: bit-exact streaming oracle, mmap + LRU, router/shared stay resident (roadmap #6)
- **v5.14** — Multi-process DualPipe (one stage per process, phased queue protocol, gradient-exact vs single-process)
- **v5.13** — CPU-trained toy char-level checkpoint (MTP aux loss, acceptance 1.00 on greedy) + train_toy_checkpoint.py
- **v5.12** — Batched equal-length prefill in the engine (hybrid recurrent-state models included)
- **v5.11** — Fused dequant×matmul kernels (AWQ/GPTQ/MXFP4), MXFP4 decode fix, eval loglikelihood harness
- **v5.10** — CPU benchmark suite: analytic KV-cache accounting + wall-clock generation, BENCHMARKS.md
- **v5.9** — Gemma-style attention + final logit soft-capping, per-head QK-norm, sliding-window attention with StreamingLLM sinks
- **v5.8** — YaRN RoPE scaling, DSA-style sparse top-k decode, per-head Muon, GPTQ act-order
- **v5.7** — RoPE scaling (linear/NTK), FP8 latent KV cache, Hyper-Connections, QAT training
- **v5.6** — Hybrid packed-sequence training, MXFP4, Muon optimizer
- **v5.5** — K3-aligned feature wave: hybrid Gated-Delta attention, LatentMoE, quantile balancing, attention residuals, SiTU-GLU
- **v5.0–v5.4** — Review-hardened core: MLA weight absorption, true GPTQ, strict speculative sampling, NaViT + audio encoders

## License

Apache License 2.0 — see [LICENSE](LICENSE).
