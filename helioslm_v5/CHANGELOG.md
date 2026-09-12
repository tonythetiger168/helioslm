# HeliosLM v5 Changelog

## v5.4 (2026-09-12) - Packed Sequences & Robustness Fix Release

Fix release on top of v5.3: one attention correctness fix (packed-sequence
cross-document leakage), engine/MTP/training edge-case hardening, and a
batch of LOW-severity input-validation fixes that replace silent
misbehavior with loud errors. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 23/23 on CPU (19 v5.3 tests + 4
new: `test_packed_positions`, `test_navit_input_validation`,
`test_quant_input_validation`, `test_gptq_odd_dims`).

### Attention (`attention/mla.py`)
- **B1**: packed-sequence cross-document leakage fixed. At prefill, custom
  `position_ids` with per-document position restarts (e.g. `[0,1,2,0,1,2]`)
  are now segmented at the reset points (`pos[i] <= pos[i-1]` starts a new
  document) and the causal mask additionally requires both tokens to be in
  the same document — previously a purely positional `k_pos <= q_pos`
  comparison let a later packed document attend back into an earlier one.
  Applies to both cache modes (absorbed and expanded share
  `_build_attn_mask`). Decode with a KV cache still requires the default
  contiguous layout; non-default `position_ids` raise `ValueError` instead
  of silently mis-masking. Regression test: `test_packed_positions`
  (perturbing document A leaves document B's logits unchanged in both
  cache modes; packed doc == solo doc forward; default-position leak
  sanity check proves the test is not vacuous).

### Inference (`inference/vllm_engine.py`, `inference/mtp.py`)
- Engine: requests admitted already-complete (e.g. `max_new_tokens=0`)
  are retired before prefill instead of emitting one token anyway; a
  request that fails mid-step (forward exception) is now retired and its
  blocks released instead of leaking (O2). Engine block parameters default
  to `config.paged_attention`; the removed `config.batch_size` is no
  longer referenced (O5).
- MTP: `MTPModule.generate` / `MTPDecoder.generate` restore the original
  train/eval modes on exit — including when an exception is raised (O1).
  `num_rounds` now counts actual speculative rounds, matching the
  `MTPGenerateResult` field doc (O10). The sampling-path acceptance draw
  uses torch's RNG (reproducible via `torch.manual_seed`) instead of
  Python's `random` (O11), and the same top-k filter is applied to both
  the main-model and draft distributions so the accept ratio refers to the
  same filtered target (strict speculative sampling).

### Training (`training/fp8_trainer.py`, `training/grpo.py`)
- FP8: after FP8 conversion replaced `model.lm_head`, MTP modules kept
  pointing at the OLD `nn.Linear`, silently breaking weight tying — the
  trainer now re-binds every MTP module's `lm_head` / `embed_tokens` to
  the main model's current modules (same fix class as
  `QuantizationManager.quantize_model` in v5.3).
- GRPO: an empty prompts list, an empty/blank prompt, or a
  questions/answers length mismatch now raise a clear `ValueError` up
  front instead of producing empty-tensor crashes downstream.

### Multimodal (`audio/streaming_encoder.py`, `vision/navit.py`)
- Audio: `process_stream` calls `reset_state()` first, so leftover
  per-layer memory prefixes from a previous stream can no longer leak into
  a new one.
- NaViT: images smaller than `vision_patch_size` (zero patches) and
  `forward_packed([])` raise a clear `ValueError` instead of a cryptic
  Conv2d kernel-size error / `IndexError`.

### Quantization (`quantization/standard_quant.py`)
- AWQ calibration activations whose last dim != `in_features` now raise
  `ValueError` (aligned with the GPTQ path) instead of silently
  reshape-reinterpreting the calibration data.
- `quantize_model` on a bare `nn.Linear` (a module cannot replace itself
  in place) now emits a warning that the layer is left unquantized instead
  of skipping silently.
- `bits != 4` raises `ValueError` on both `AWQLinear.from_linear` and
  `GPTQLinear.from_linear` (bare `assert` removed — asserts vanish under
  `python -O`).
- GPTQ packing verified exact for odd `in_features`/`out_features`
  (pad-to-8 of qweight rows / qzeros cols); covered by
  `test_gptq_odd_dims`.

### Misc
- Version strings unified to v5.4 across `model_v5.py`,
  `configs/config_v5.py` (`model_name="HeliosLM-v5.4"`), the test suite,
  README, and this CHANGELOG.

## v5.3 (2026-09-12) - v5.2 Review Findings Fix Release

Fix release addressing the v5.2 code review
(`helioslm_v5.2_code_review.md`: 1 CRITICAL + 10 MAJOR + ~20 MINOR).
Finding IDs below refer to that report. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 19/19 on CPU (15 v5.2 tests + 4
new: `test_mla_absorbed_left_padding`, `test_awq_calibration`,
`test_quant_weight_property`, `test_mtp_rebind_after_quantization`).

### CRITICAL
- **C1** (`attention/mla.py`): absorbed-mode left-padding no longer
  produces NaN — fully-masked pad query rows are zeroed via
  `nan_to_num` after the softmax, stopping the cross-layer NaN pollution
  of real tokens. Regression test: `test_mla_absorbed_left_padding`
  (absorbed + left-padding finite, and equal to expanded mode on real
  tokens).

### MAJOR — quantization / docs (`quantization/standard_quant.py`, README/CHANGELOG)
- **M-Q1**: `GPTQLinear.forward` casts the bias to the input dtype
  (fp16/bf16 inputs no longer crash; AWQ/FP8 already did this).
- **M-Q2**: all allocations in the GPTQ calibration path (`err_blk`,
  `q`, `scales`, `zeros`, index tensors, packing shifts) now live on
  the weight's device, so calibration works when the layer is on CUDA.
- **M-Q3**: AWQ calibration path repaired. The v5.2 alpha=1
  full-magnitude scaling degraded to ~10x RTN's output error under
  peaky/outlier activations (85% vs 7.5%). The calibration path now
  grid-searches the scale exponent over {0, 0.25, 0.5} (alpha=0 ==
  plain RTN), clamps scales to [1, 2], and clip-searches every group
  over [0.5, 1.0] x (min, max) against the activation-weighted output
  error; since the grids contain the plain-RTN configuration the result
  is never significantly worse than RTN (measured: gauss 1.00x,
  lognormal sigma=2 1.04x, peaky 1.17x RTN output error, vs 11.3x
  before the fix). Covered by `test_awq_calibration`.
- **M-Q4**: README/CHANGELOG GPTQ numbers corrected — the "10.1% ->
  3.4%" figure was mislabeled as real `lm_head` activations. Corrected
  and labeled separately: 10.1% -> ~7.1% on real lite `lm_head`
  activations; 10.0% -> 3.3% on synthetic low-rank + noise data
  (`test_gptq_calibration`).

### MAJOR — inference
- **M-I1** (`inference/mtp.py`): MTP sampling correction token no longer
  applies temperature twice to an already-normalized probability vector.
- **M-I2** (`inference/paged_attention.py`): `_apply_rope` broadcasting
  for the documented `[S, D]` position shape fixed.
- **M-I3** (`inference/vllm_engine.py`): BlockManager accounting now
  includes the prompt pad prefix, keeping the watermark-based OOM
  guard consistent with real KV memory.

### MAJOR — training
- **M-T1** (`training/grpo.py`): the policy model is put back in train
  mode at the start of `train_step` (rollout `generate` switches to
  eval and previously never restored it).
- **M-T2** (`training/fp8_trainer.py`): FP8 amax history guarded against
  NaN/Inf so one bad batch can no longer permanently poison the scales
  and master weights.
- **M-T3** (`training/grpo.py`): reward matching extracts the final
  answer and compares for equality instead of substring matching
  ("25" no longer matches answer "2").

### MAJOR — model core
- **M-C1** (`attention/mla.py`): `_build_attn_mask` now compares
  positions in a consistent space (safe under non-monotonic
  `position_ids` / packed sequences).
- **M-C2** (`moe/sigmoid_moe.py`): `expert_load` only accumulates in
  training mode (RL rollouts no longer pollute the aux-free bias
  statistics).
- **M-C3** (`moe/sigmoid_moe.py`): distributed `update_bias` statistics
  all-reduced across ranks; `route_bias` stays consistent under DDP.
- **M-C4** (`attention/mla.py`): RoPE cos/sin buffers are
  non-persistent, so lazy growth no longer breaks strict checkpoint
  reloads.

### MINOR (selected)
- Tests: the always-true EOS assertion in `test_v5_model` is now a real
  per-element check (every position after the first EOS must equal
  `pad_token_id`); the no-calibration GPTQ path in `test_quantization`
  is labeled as the documented RTN fallback; new coverage for the
  quantized `.weight` property (identical to the forward effective
  weight), AWQ calibration path, MTP re-binding after quantization, and
  fp16/bf16 GPTQ forward.
- GRPO: sequence-level ratio clamped; k3 KL uses `expm1` to avoid
  catastrophic cancellation; generate continuation heuristic tightened.
- Engine/model: `_pad_caches` memo bounded; MTP path no longer silently
  drops `top_p`; cos/sin cast to the input dtype; version strings
  bumped to v5.3; NaViT boundary inputs raise clean `ValueError`.
- DualPipe: dead `num_micro_batches` parameter removed; `run_dual([])`
  raises a clear error; recompute saves/restores RNG state.

## v5.2 (2026-09-12) - Absorption, True GPTQ, Batched Inference

Feature release on top of the post-review v5.1 baseline. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 15/15 on CPU (13 v5.1 tests
unchanged + 2 new: `test_gptq_calibration`, `test_audio_sliding_window`);
the integration suite (`integration_test_v51.py`) passes 9/9.

### MLA weight absorption (`attention/mla.py`, `configs/config_v5.py`)
- `config.attention.use_absorption=True` (default ON): the KV cache stores
  only the shared latent `c_kv [B,1,L,kv_latent_dim]` and shared
  `k_rope [B,1,L,rope_head_dim]`; scores are computed as
  `(q·W_UK)·c_kv^T + q_rope·k_rope^T` with the output through `W_UV`.
- Real cache savings: 72 values/token on lite (-71.9% vs its 256-value MHA
  baseline) and 576 values/token on the full config (-97.66% vs 24,576 for
  64-head MHA). Absorbed vs non-absorbed outputs on identical weights
  differ by < 1e-4 (prefill + decode + post-truncation, verified in
  `test_mla_absorption`).

### True GPTQ Hessian error compensation (`quantization/standard_quant.py`)
- `GPTQLinear.from_linear(linear, group_size=128, bits=4,
  calibration_data=None, percdamp=0.01)`: with calibration activations it
  runs the real GPTQ algorithm — damped Hessian H = 2/N·X^T X, Cholesky
  inverse, column-by-column quantization in OBS order with quantization
  error propagated into not-yet-quantized columns (blocked lazy update);
  without calibration data it falls back to per-group RTN.
- Measured output reconstruction error (RTN -> GPTQ): 10.1% -> ~7.1% on
  real lite `lm_head` activations; 10.0% -> 3.3% on synthetic low-rank +
  noise data (`test_gptq_calibration`). (This entry originally mislabeled
  the 3.3% synthetic-data figure as "real lm_head activations"; corrected
  in v5.3 per review finding M-Q4.) Raw weight-space error can be slightly
  higher than RTN — inherent to minimizing the activation-weighted output
  error, documented in the module docstring.
- `QuantizationManager.quantize_model(..., calibration_data=...)` passes
  calibration data through (single tensor or per-layer dict).
- Quantized layers (`AWQLinear`/`GPTQLinear`/`FP8Linear`) now expose a
  `.weight` property (slow-path dequantize) so MLA weight absorption and
  other dense-weight consumers keep working on quantized models;
  `quantize_model` re-binds MTP's shared `lm_head`/`embed_tokens` after
  quantization.

### MTP speculative decoding: batch > 1 (`inference/mtp.py`)
- Batched drafting/verification with sequences grouped by cache length and
  per-sequence acceptance lengths.
- Rollback now truncates every cache tensor along dim 2 (no prefix
  recompute); EOS handled per row; early-finished rows right-padded with
  `pad_token_id`. Each batch row verified equal to per-row plain greedy.

### Inference engine: batched decode stepping (`inference/vllm_engine.py`)
- Continuous batching fully implemented: active requests are aligned to a
  common watermark with pad-token cache prefixes, and each decode step is
  one batched `[B,1]` forward. Engine output verified equal to
  per-request greedy `generate`; blocks remain leak-free.

### Streaming audio: sliding-window memory (`audio/streaming_encoder.py`)
- New `audio_max_memory_frames` (read via `getattr(config.multimodal, ...,
  1000)`, default 1000): each transformer layer's memory prefix is capped
  at this many frames; on overflow the OLDEST frames are dropped. Within
  the window, streaming output is numerically identical to the unbounded
  one-shot forward; `reset_state()` restarts cleanly.

### Known limitations (carried over / updated)
- DualPipe remains a single-process schedule simulation, not distributed.
- GRPO uses a sequence-level ratio/clip objective (not per-token).
- NaViT `forward_packed` is padding-based batching, not nested-tensor
  packing; no position-embedding interpolation beyond `vision_max_grid`.
- Resolved in v5.2: MLA weight absorption (now implemented, default ON),
  GPTQ Hessian compensation (implemented with calibration data), MTP
  batch-size-1 restriction (batched speculative decode supported).

## v5.1 (2026-09-12) - Post-Review Rewrite

Complete rewrite/fix pass following the full v5.0 code review
(`helioslm_v5_code_review.md`, 14 CRITICAL / 27+ MAJOR findings). Every
module was repaired and verified; the cross-module integration suite
(`integration_test_v51.py`) passes 9/9 and the rewritten unit suite
(`helioslm_v5/tests/test_v5.py`) passes 12/12 on CPU. Finding IDs below
refer to the review report.

### Model core (`model_v5.py`, `attention/mla.py`, `moe/sigmoid_moe.py`, `configs/config_v5.py`)
- **C1**: Fixed cached-decode attention mask — bottom-right-aligned causal
  mask merged with padding masks; `is_causal` fast path only when safe.
- **C2**: RoPE now uses absolute positions during cached decoding
  (`position_ids` default to `past_len..past_len+seq-1`).
- **C3**: V is a separate projection of the KV latent and is never
  RoPE-rotated (no more V=K copy).
- **M1/M2**: KV cache stores expanded per-head K_nope + V + shared k_rope,
  making decode O(1) per step; no double `kv_b_proj`. Weight absorption is
  documented as NOT implemented; `get_kv_cache_size()` reports truthful
  numbers.
- **M3**: RoPE cos/sin tables precomputed for a bounded budget and grown on
  demand (no more ~168 MB/layer 1M-position tables).
- **M4**: MLA head dims decoupled from `hidden_size / num_heads`;
  `hidden_size=4096` config uses 64 heads; divisibility validated.
- **M5**: Pre-norm architecture fixed — each sublayer input normalized
  exactly once (removed double-norming of the MoE input).
- **M6**: Aux-free balancing implemented as specified — `route_bias` only
  affects top-k selection (gating weights are bias-free), has
  `requires_grad=False`, and is nudged by fixed-step `update_bias()`.
- **M7/M8**: MoE dispatch vectorized (single sort by expert id, one sync);
  distributed device-limited path no longer crashes.
- **M9**: `generate()` is batch-safe (per-row EOS freezing with pad fill).
- **M10/M25**: paged block management moved to the inference engine layer
  (model keeps per-layer `past_key_values`; engine owns block accounting).
- **M11**: Padding `attention_mask` support throughout model and MLA.

### Training (`training/dualpipe.py`, `training/fp8_trainer.py`, `training/grpo.py`)
- **C4/M14**: DualPipe rewritten as an honest single-process simulation of
  the schedule (warmup / 1F1B / cooldown, recorded in `trace`) with
  gradient-correct backward via activation recomputation; dead
  `backward_buffers` removed; micro-batch-mean loss scaling.
- **C5/C6**: GRPO loss now carries gradients (grad-carrying logprob forward
  through the model) and rewards are aligned to answers per question group.
- **M12/M13/M18**: GRPO ratio is `exp(new − old_policy)`; KL uses
  DeepSeekMath's non-negative k3 estimator; the reference model is real,
  frozen, and eval-mode.
- **C7**: FP8 trainer zeroes gradients every step (no accumulation).
- **M15/M16/M17**: FP8 quantization is real (native `float8` cast or
  explicit round-to-nearest grid with straight-through estimator), E5M2
  gradient quantization via backward hooks, delayed dynamic scaling, AdamW
  over fp32 master weights with clipping.

### Inference (`inference/mtp.py`, `inference/paged_attention.py`, `inference/vllm_engine.py`)
- **C8/C9**: MTPDecoder matches the model's real
  `(logits, hidden, past)` contract; decode shapes fixed.
- **C10**: Real speculative decoding implemented — greedy/sampling
  verification against the main model, first-rejection truncation with
  correction token, bonus token on full acceptance, KV-cache rollback;
  `MTPGenerateResult` carries acceptance statistics. Verified:
  `generate(use_mtp=True)` output identical to plain greedy.
- **M22/M23/M24/m1-m3**: MTP transformer is causal-masked (no future-token
  leakage), modules are depth-indexed with chained hidden states, and
  embedding/LM head are shared with the main model.
- **C11**: Paged KV cache dtype follows the input dtype (or constructor
  override); reads cast to query dtype.
- **M19/M20**: Write path appends at `context_len` and allocates blocks
  itself (automatic across block boundaries); no more last-token clobber.
- **M21**: True reference-counted copy-on-write `fork()` — writes to shared
  blocks copy first; freeing a shared block no longer corrupts siblings.
- **M25/M26**: VLLMEngine rewritten around a real `Request` dataclass with
  `add_request`/`schedule`/`step`/`run`; continuous batching over unequal
  prompt lengths; incremental decoding via `past_key_values` (no O(L^2)
  recompute); blocks freed on finish (leak-free).
- **M27**: HPA recommendation uses unit-consistent load (utilization /
  memory ratio) and `current_replicas * load / target`.

### Multimodal & quantization (`vision/navit.py`, `audio/streaming_encoder.py`, `quantization/standard_quant.py`)
- **C12/C13/M28/M30**: Streaming audio encoder repaired (missing import,
  buffer shapes) with correct causal-conv input-tail state and per-layer
  causal memory prefixes — chunked streaming is numerically identical to a
  one-shot causal forward.
- **M29**: `process_stream` chunk size derived from sample_rate/hop_length
  (500 ms = 50 frames at 16 kHz / hop 160).
- **M31/M33**: NaViT uses factorized row/col position embeddings (no
  grid-stride collisions), raises `ValueError` beyond `vision_max_grid`,
  and `forward_packed` allocates on the input device.
- **C14/M32**: AWQ/GPTQ are real 4-bit per-group RTN quantization with
  packed storage and deterministic dequantize (AWQ supports odd
  `in_features`); biases preserved; FP8 path uses native `float8_e4m3fn`
  when available; unknown methods raise `ValueError`.

### Tests & docs
- **M34/M35/M36**: `tests/test_v5.py` rewritten — no swallowed exceptions,
  exit code 1 on any failure, lite-config CPU suite with numerical
  assertions (cache-vs-full-forward equivalence, bias/gating separation,
  CoW isolation, engine-vs-greedy equality, DualPipe gradient equality,
  quantization error bounds, streaming/causality, packed-mask correctness).
- **README claims audit**: removed or rewrote all unsupported numbers
  (93.3% KV reduction, 85-90% MTP acceptance, +80% throughput, +40%
  batching, -50% FP8 cost, 95% GPU utilization, "对标 o1", <200 ms latency);
  "vLLM integration" is now described as a vLLM-style engine; GPU
  monitoring described as in-memory metrics with optional Prometheus
  export.

### Known limitations (documented, not hidden)
- MLA weight absorption not implemented (expanded per-head cache).
- DualPipe is a single-process schedule simulation, not distributed.
- GPTQ has no Hessian error compensation (RTN only).
- GRPO uses a sequence-level ratio/clip objective (not per-token).
- NaViT `forward_packed` is padding-based batching, not nested-tensor
  packing; no position-embedding interpolation beyond `vision_max_grid`.
- MTP speculative decoding supports batch size 1 only.
