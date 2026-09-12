# HeliosLM v5.4 — DeepSeek-Style LLM

A PyTorch reference implementation of a DeepSeek-V3-style LLM stack, built and hardened through four rounds of adversarial code review (see `docs/`).

[![CI](https://github.com/tonythetiger168/helioslm/actions/workflows/ci.yml/badge.svg)](https://github.com/tonythetiger168/helioslm/actions)

## Features

| Area | Implementation |
|---|---|
| **Attention** | MLA (Multi-head Latent Attention) with **weight absorption** — latent-only KV cache, −97.7% memory vs MHA (full config), verified equivalent to expanded path (<1e-4) |
| **MoE** | Sigmoid-gated, fine-grained experts with **auxiliary-loss-free** load balancing (selection-only bias + load-error heuristic updates) |
| **Speculative decoding** | DeepSeek-style MTP with **strict verification** (residual `(p−q)₊` resampling), batch support, O(1) cache-truncation rollback |
| **Serving** | vLLM-style engine: paged KV accounting, copy-on-write forks, watermark-aligned continuous batching |
| **Training** | FP8 trainer (native float8 + STE, E5M2 gradient hooks, AdamW master weights), DualPipe schedule simulation (recompute-based, gradient-exact), GRPO (real sampling, k3 KL, answer-extraction rewards) |
| **Quantization** | **True GPTQ** (Hessian OBS with error compensation), AWQ with activation-aware grid search, native FP8 — all with `from_linear` real-weight packing |
| **Multimodal** | NaViT vision encoder (row/col position decomposition, mixed-resolution packing), streaming audio encoder (causal, sliding-window memory, bit-equivalent to one-shot) |

## Quick Start

```bash
pip install torch
python -m helioslm_v5.tests.test_v5        # 23 unit tests
python integration_test_v51.py             # 9 end-to-end integration tests
```

```python
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

model = HeliosLMv5(HeliosLMv5Config(size="lite"))   # CPU-friendly
out = model.generate([[1, 2, 3]], max_new_tokens=20, temperature=0)
```

## Repository Layout

```
helioslm_v5/          # v5.4 source (configs, src/{attention,moe,inference,training,vision,audio,quantization}, tests)
docs/                 # code review reports + per-version fix reports (v5.0 → v5.4)
integration_test_v51.py
```

## Version History

- **v5.1** — Full review fixes: 14 CRITICAL + 25 MAJOR (broken MLA cache decode, gradient-less DualPipe, stub quantization, ...)
- **v5.2** — MLA weight absorption, true GPTQ, MTP batching, batched engine decode
- **v5.3** — v5.2 review fixes: strict speculative sampling, AWQ calibration repair, NaN/edge hardening
- **v5.4** — Packed-sequence document isolation, engine edge cases, full hardening sweep

See `helioslm_v5/CHANGELOG.md` and `docs/` for details.

## Known Limitations

Single-process DualPipe simulation; non-fused quantization kernels; CPU-verified (CUDA paths static-checked); MTP acceptance requires trained weights. See `docs/V5.4_FIXES_REPORT.md`.
