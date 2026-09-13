# HeliosLM v5.6 — DeepSeek/K3-Style LLM

A PyTorch reference implementation of a DeepSeek-V3-style LLM stack, built and hardened through four rounds of adversarial code review plus a v5.5 feature wave aligned with Kimi-K3-class architecture mechanisms (see `docs/`).

[![CI](https://github.com/tonythetiger168/helioslm/actions/workflows/ci.yml/badge.svg)](https://github.com/tonythetiger168/helioslm/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

| Area | Implementation |
|---|---|
| **Attention** | MLA (Multi-head Latent Attention) with **weight absorption** — latent-only KV cache, −97.7% memory vs MHA (full config), verified equivalent to expanded path (<1e-4); **hybrid linear attention** (v5.5): Gated Delta Rule layers interleaved with MLA, fixed-size recurrent state cache, decode ≡ one-shot (<2e-7) |
| **MoE** | Sigmoid-gated, fine-grained experts with **auxiliary-loss-free** load balancing (selection-only bias + heuristic or **quantile** updates); **LatentMoE** (v5.5): routed experts in shared latent space (down→dispatch→up), shared experts full-width; **SiTU-GLU** (v5.5) tanh soft-capped activation |
| **Cross-layer** | **Attention Residuals** (v5.5): per-layer gated injection of accumulated lower-layer attention outputs, threaded through DualPipe (gradient-exact, bitwise-verified) |
| **Speculative decoding** | DeepSeek-style MTP with **strict verification** (residual `(p−q)₊` resampling), batch support, O(1) cache-truncation rollback; hybrid recurrent-state rollback via restore+replay (v5.5) |
| **Serving** | vLLM-style engine: paged KV accounting, copy-on-write forks, watermark-aligned continuous batching; hybrid-aware batching fallback (v5.5) |
| **Training** | FP8 trainer (native float8 + STE, E5M2 gradient hooks, AdamW master weights), DualPipe schedule simulation (recompute-based, gradient-exact), GRPO (real sampling, k3 KL, answer-extraction rewards) |
| **Quantization** | **True GPTQ** (Hessian OBS with error compensation), AWQ with activation-aware grid search, native FP8 — all with `from_linear` real-weight packing |
| **Multimodal** | NaViT vision encoder (row/col position decomposition, mixed-resolution packing), streaming audio encoder (causal, sliding-window memory, bit-equivalent to one-shot) |

## Quick Start

```bash
pip install torch
python -m helioslm_v5.tests.test_v5        # 34 unit tests
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
helioslm_v5/          # v5.5 source (configs, src/{attention,moe,inference,training,vision,audio,quantization}, tests)
docs/                 # code review reports + per-version fix reports (v5.0 → v5.5)
integration_test_v51.py
```

## Version History

- **v5.1** — Full review fixes: 14 CRITICAL + 25 MAJOR (broken MLA cache decode, gradient-less DualPipe, stub quantization, ...)
- **v5.2** — MLA weight absorption, true GPTQ, MTP batching, batched engine decode
- **v5.3** — v5.2 review fixes: strict speculative sampling, AWQ calibration repair, NaN/edge hardening
- **v5.4** — Packed-sequence document isolation, engine edge cases, full hardening sweep
- **v5.5** — K3-aligned feature wave: hybrid Gated-Delta linear attention, LatentMoE, quantile balancing, attention residuals, SiTU-GLU (34 unit tests)
- **v5.6** (2026-09-13) — Daily-analysis improvement round: hybrid packed-sequence training support (doc-boundary state reset), MXFP4 quantization, Muon optimizer, test-suite calibration (36 unit tests)

See `helioslm_v5/CHANGELOG.md` and `docs/` for details.

## Known Limitations

Single-process DualPipe simulation; non-fused quantization kernels; CPU-verified (CUDA paths static-checked); MTP acceptance requires trained weights; hybrid (linear-attention) models now SUPPORT packed sequences via doc-boundary state reset (v5.6); hybrid models use unpadded prefill batching in the engine. See `docs/V5.5_FEATURES_REPORT.md`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
