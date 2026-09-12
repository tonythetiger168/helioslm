# HeliosLM v5.4 - DeepSeek-Style Architecture

Reference LLM implementation with DeepSeek-V3-style efficiency techniques.
All modules below are implemented and exercised by a CPU test suite with
numerical assertions (see Quick Start). v5.1 followed a full code review:
every module was rewritten so that the behavior below is what the code
actually does — unsupported performance claims have been removed. v5.2
builds on that baseline: MLA weight absorption (the real KV-cache saving),
true GPTQ Hessian error compensation, batched MTP speculative decoding and
batched engine decode steps, and a bounded sliding-window audio memory.
v5.3 is a fix release on the v5.2 code review: absorbed-mode left-padding
NaN, GPTQ fp16/bf16 forward, GPTQ calibration-path device handling, and a
repaired AWQ calibration path (see Quantization). v5.4 is a robustness fix
release: packed-sequence cross-document attention isolation, strict MTP
sampling verification, engine edge-case cleanup (max_new_tokens=0, failed
requests), and input validation that fails loudly instead of silently
misbehaving (see CHANGELOG).

## P0: Core Architecture Improvements

### MLA (Multi-Head Latent Attention)
- Low-rank KV compression: hidden -> `kv_latent_dim` latent -> per-head K/V
- Decoupled RoPE: one k_rope shared across heads; V is never rotated
- `head_dim` decoupled from `hidden_size / num_heads` (no truncation)
- **Weight absorption (v5.2, default ON via
  `config.attention.use_absorption`)**: the cache stores only the shared
  latent `c_kv [B,1,L,kv_latent_dim]` and the shared `k_rope
  [B,1,L,rope_head_dim]`; attention scores are computed as
  `(q·W_UK)·c_kv^T + q_rope·k_rope^T` and the output goes through `W_UV`.
  Truthful accounting via `get_kv_cache_size()`: **72 values/token on lite
  (-71.9% vs its 256-value MHA baseline)** and **576 values/token on the
  full config (-97.7% vs 24,576 for 64-head MHA)** — the DeepSeek-style
  saving is now real, not projected.
- With `use_absorption=False` the cache falls back to the expanded per-head
  K_nope + V + shared k_rope layout (decode is then O(1) in projection
  compute). Both layouts are numerically equivalent on the same weights
  (verified, max diff < 1e-4 over prefill + decode, after dim-2 cache
  truncation, and under left-padding — the v5.2 absorbed-mode left-padding
  NaN was fixed in v5.3).
- Config structural constraints are validated eagerly (`ValueError` on
  invalid configs) instead of failing downstream.
- **Packed sequences (v5.4)**: at prefill, custom `position_ids` with
  per-document position restarts (e.g. `[0,1,2,0,1,2]` packing two
  documents into one row) are segmented at the reset points, so attention
  never crosses a packed document boundary (verified: perturbing one
  document leaves the other's logits unchanged). During cached decode only
  the default contiguous layout is accepted; non-default `position_ids`
  raise `ValueError` instead of silently mis-masking.

### MTP (Multi-Token Prediction)
- Predicts multiple future tokens with per-depth modules chained on hidden
  states (module k predicts position t+k+2)
- Real speculative decoding: greedy/sampling verification against the main
  model, first-rejection truncation, correction/bonus tokens, KV-cache
  rollback. Verified: `generate(use_mtp=True)` output is identical to plain
  greedy decoding.
- **Strict sampling verification (v5.4)**: in the sampling path the same
  top-k filter is applied to the main model's distributions (x0,
  per-position verification, correction, bonus) AND to the draft
  distributions, so both sides of the accept ratio refer to the same
  filtered target; the acceptance draw uses torch's RNG (reproducible via
  `torch.manual_seed`), and train/eval modes are restored even when a
  generation raises.
- **Batched speculative decoding (v5.2)**: batch > 1 is supported —
  sequences are grouped by cache length for batched drafting/verification,
  acceptance length is tracked per sequence, rollback truncates every
  cache tensor along the sequence dim (dim 2) instead of recomputing the
  prefix, EOS is handled per row, and early-finished rows are right-padded
  with `pad_token_id`. Verified: each batch row equals its per-row plain
  greedy reference.
- Acceptance rate depends on trained MTP weights (untrained weights accept
  ~nothing); no acceptance/throughput numbers are claimed.
- MTP modules share the main model's embedding and LM head (weight tying);
  `QuantizationManager.quantize_model` re-binds them after quantization so
  the tying survives.

### Sigmoid MoE with Device-Limited Routing
- Sigmoid routing (independent per-expert scores, no softmax competition)
- Aux-free load balancing: a selection bias affects **only** top-k expert
  selection, never the gating weights, and is updated by a fixed-step
  load-error heuristic (`update_bias`), not by gradient descent
- Device-limited routing across `device_group_size` devices under
  `torch.distributed`; single-process path routes over all experts
- Vectorized dispatch: one sort by expert id, no per-expert Python loop
- Shared experts always active

### PagedAttention
- Block-based KV cache (configurable page size) with block tables and
  non-contiguous physical storage
- Allocation-on-write: block-boundary crossings are handled inside the
  write path
- Reference-counted blocks with true copy-on-write `fork()` for shared
  prefixes (writes to shared blocks copy first; verified isolation)
- Cache dtype follows the input dtype (or a constructor-specified dtype)

## P1: Training Improvements

### FP8 Mixed Precision
- E4M3 for forward activations/weights, E5M2 gradient quantization via
  backward hooks (DeepSeek-V3 recipe)
- Real quantization: native `torch.float8_*` cast when available, otherwise
  explicit round-to-nearest FP8-grid simulation; straight-through estimator
  keeps gradients alive
- AdamW over fp32 master weights with grad clipping; gradients are zeroed
  every step (no cross-step accumulation)

### GRPO (Group Relative Policy Optimization)
- No value model needed; group mean as baseline with group-normalized
  advantages
- Importance ratio against the **old policy** (the frozen reference model
  only enters the KL term); DeepSeekMath's non-negative k3 KL estimator
- Real sampling through the model's own `generate`; tokenizer-pluggable
  (byte-level fallback)
- Honest deviation: the ratio/clip objective is sequence-level (one ratio
  per response), not DeepSeekMath's per-token objective — documented in the
  module docstring.

### DualPipe + Expert Parallelism
- Single-process **simulation** of the DualPipe schedule (warmup /
  interleaved 1F1B / cooldown ordering, inspectable via `trace`); there is
  no distributed p2p communication and no true cross-device overlap
- Gradient-correct: backward recomputes stage forwards from stored leaf
  inputs; gradients verified equal (atol 1e-5) to a direct
  forward+backward with micro-batch-averaged loss
- Expert-to-device mapping for expert parallelism; `all_to_all` requires an
  initialized `torch.distributed` process group and raises
  `NotImplementedError` otherwise

## P2: Multimodal

Multimodal encoders are built only when `config.multimodal.enabled=True`
(the `size="lite"` test config disables them to stay CPU-friendly).

### NaViT (Native Vision Transformer)
- Arbitrary resolutions/aspect ratios without resize distortion, up to
  `vision_max_grid` patches per side (default 32; larger grids raise
  `ValueError` — position-embedding interpolation is not implemented)
- Factorized 2D learned position embeddings (separate row/col tables, no
  hardcoded grid-stride collisions)
- `forward_packed` batches mixed image sizes via zero-padding + key padding
  mask (padding-based batching, not nested-tensor sequence packing)
- Degenerate inputs fail loudly (v5.4): images smaller than
  `vision_patch_size` and empty `forward_packed([])` raise a clear
  `ValueError` instead of a cryptic Conv2d shape error / IndexError

### Streaming Audio Encoder
- Chunk-based processing with causal convolutions (per-layer input-tail
  state) and a causal transformer with per-layer memory prefixes
- Chunked streaming output is numerically identical to a one-shot causal
  forward (verified, max diff ~1e-6)
- **Sliding-window memory (v5.2)**: each layer's memory prefix is capped at
  `multimodal.audio_max_memory_frames` frames (default 1000); on overflow
  the oldest frames are dropped. While the stream fits in the window,
  output is identical to the unbounded forward; beyond it, history older
  than the window is forgotten (sliding-window approximation).
- `process_stream` derives frames-per-chunk from sample_rate/hop_length
  (500 ms = 50 frames at 16 kHz / hop 160); input is a precomputed mel
  spectrogram (waveform -> mel is out of scope)

## P3: Quantization

- AWQ: real 4-bit per-group asymmetric RTN packing (2 values per uint8);
  optional activation-aware scaling when calibration data is passed;
  biases preserved. Measured reconstruction error ~6-8% relative
  (near the 4-bit RTN floor).
  - **Calibration path (repaired in v5.3)**: the per-input-channel scale
    exponent is grid-searched over {0, 0.25, 0.5} (alpha=0 is exactly
    RTN), scales are clamped to [1, 2] (salient channels are only scaled
    up, moderately), and each group runs a clip grid-search over
    [0.5, 1.0] x (min, max) minimizing the activation-weighted output
    error. Because the grids contain the plain-RTN configuration, the
    calibrated path is never significantly worse than RTN (<= ~1.17x on
    peaky/outlier activations, at parity elsewhere) — the v5.2
    full-magnitude (alpha=1) scaling degraded to ~11x RTN's error under
    peaky activations. This is still a *conservative approximation* of
    full AWQ, not a reimplementation of the paper.
- GPTQ: **true GPTQ algorithm (v5.2)** — with calibration activations the
  Hessian H = 2/N·X^T X is damped (`percdamp`), inverted via Cholesky, and
  columns are quantized one by one with OBS error propagation into the
  not-yet-quantized columns; without calibration data it falls back to
  per-group RTN. Measured output reconstruction error (RTN -> GPTQ):
  **10.1% -> ~7.1% on real lite `lm_head` activations** (calibrated on
  real forward-pass activations) and **10.0% -> 3.3% on synthetic
  low-rank + noise activations** (`test_gptq_calibration`) — the two
  numbers come from different data sources and are not interchangeable.
  GPTQ minimizes the activation-weighted *output* error; its raw
  weight-space error is typically comparable to or slightly above RTN —
  that trade-off is inherent to the algorithm, not a bug.
- Quantized layers expose a `.weight` property (slow-path dequantize) so
  weight-absorbing code (MLA's W_UK/W_UV folding) keeps working after
  quantization; `quantize_model` re-binds MTP's shared `lm_head` /
  `embed_tokens` at the end.
- Input validation (v5.4): calibration activations whose last dim does not
  equal `in_features` raise `ValueError` on the AWQ path too (it previously
  reshape-reinterpreted them silently, unlike GPTQ); `bits != 4` raises
  `ValueError` (no bare `assert`); a bare `nn.Linear` passed directly as
  `model` cannot be replaced in place and now triggers a warning instead of
  a silent skip. Odd `in_features`/`out_features` pack/unpack exactly.
- FP8: `float8_e4m3fn` weight-only storage with per-tensor scale when the
  torch build supports it; `NotImplementedError` otherwise.
- `QuantizationManager.quantize_model(model, method=...,
  calibration_data=...)`; unknown methods raise `ValueError`.

## P4: Deployment

- vLLM-style inference engine: paged KV block accounting with CoW,
  continuous batching over a `Request` protocol (`add_request`/`step`/
  `run`), and incremental decoding via the model's `past_key_values`.
  Continuous batching is fully implemented (v5.2): active requests are
  aligned to a common watermark with pad-token cache prefixes and decode
  is stepped as ONE batched `[B,1]` forward per step. Verified to match
  per-request greedy `generate` exactly (no block leaks).
- GPU monitoring: in-memory metric collection (utilization, memory);
  Prometheus gauge export only when the optional `prometheus_client`
  package is installed
- HPA replica recommendation derived from current GPU utilization / memory
  ratio (unit-consistent 0..1 load), not from metric-history length

## Quick Start

From the repository root (the directory containing `helioslm_v5/`):

```bash
python -m helioslm_v5.tests.test_v5
```

Runs 23 tests against the real `size="lite"` model on CPU (a couple of
minutes), prints a per-test PASS/FAIL summary, and exits non-zero if any
test fails.

## Architecture Comparison

| Feature | v4.1 | v5.4 (DeepSeek-Style) |
|---------|------|----------------------|
| Attention | GQA | **MLA (weight absorption; latent cache -97.7% values vs MHA full / -71.9% lite; packed-sequence doc isolation)** |
| Token Prediction | Single | **MTP (multi-depth modules, batched speculative verify + dim-2 cache rollback)** |
| MoE Routing | Softmax | **Sigmoid + aux-free bias balancing + device-limited** |
| KV-Cache | Contiguous | **Paged (block-based, refcounted CoW)** |
| Training Precision | bfloat16 | **FP8 mixed (E4M3 fwd / E5M2 grads, real grid quantization)** |
| RL Training | None | **GRPO (group baseline, k3 KL vs frozen ref)** |
| Pipeline | Simple | **DualPipe-style schedule (single-process simulation)** |
| Vision | Fixed 224x224 | **NaViT (any resolution up to max_grid)** |
| Audio | Non-streaming | **Streaming causal (chunked == one-shot; sliding-window memory cap)** |
| Quantization | Custom INT4/8 | **AWQ/GPTQ/FP8 (real 4-bit packing; true GPTQ Hessian compensation with calibration data)** |
| Inference Engine | Custom | **vLLM-style engine (paged KV, continuous batching with batched [B,1] decode steps)** |
| GPU Monitoring | CPU/内存 HPA | **GPU-utilization HPA (in-memory metrics, optional Prometheus export)** |
