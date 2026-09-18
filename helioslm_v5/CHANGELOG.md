# HeliosLM v5 Changelog

## v5.18 (2026-09-18) - Content-Addressed KV Prefix Pool (roadmap #6 complete)
- `helioslm_v5/src/inference/prefix_pool.py`: cross-session prefix reuse —
  fixed-size token blocks keyed by blake2b(token ids + config fingerprint),
  per-block incremental prefill, LRU eviction, exact-length past snapshots
- The config fingerprint (model version / quant scheme — the P2 metadata
  discipline) makes stale or differently-quantized entries unserveable
- Oracle: pooled greedy == from-scratch greedy BITWISE for full hits,
  partial hits, fingerprint misses, and post-eviction re-requests
- `pool.generate()` seeds from the longest pooled prefix; stats() reports
  hits/misses/hit_tokens/evictions
- Roadmap #6 is now feature-complete: streaming oracle + break-even
  telemetry + stream scorer + prefix pool
- 58/58 tests + 9/9 integration

## v5.17 (2026-09-18) - Standalone Token-Stream Scorer (roadmap #6)
- `helioslm_v5/eval/score_stream.py`: score (prompt, output) JSONL records
  from ANY engine (colibri / vLLM / llama.cpp / HeliosLM) under a
  reference model — per-record sum_logprob / nll_per_token / ppl
- `--compare` A/B mode: paired records -> per-id deltas + bootstrap CI,
  the table-format counterpart of colibri's container A/B findings
- Char-level with the toy checkpoint (ASCII enforced, loud error) — the
  file format is the deliverable; swap checkpoints for real text
- 57/57 tests + 9/9 integration

## v5.16 (2026-09-18) - Speculation Break-Even Instrumentation (roadmap #6)
- `benchmarks/bench_spec_breakeven.py`: sweeps MTP draft on/off x cache
  state on the toy checkpoint and emits rows in the exact JSONL schema
  proposed to colibri (P3) — directly comparable numbers across engines
- Part B measures the expert-store hit-rate curve vs residency budget
  (the x-axis of the break-even surface; v5.15 store.stats() telemetry)
- First honest data point on the toy model: draft is net-negative cold
  (-0.3%) and net-positive warm (+5.4%) at 72% acceptance — cache state
  decides whether drafting pays, matching the colibri open hypothesis
- `best_draft_per_cache_state()` pure helper (unit-tested without timing)
- 56/56 tests + 9/9 integration

## v5.15 (2026-09-18) - Disk-Tier Expert Store (streaming oracle, roadmap #6)
- `helioslm_v5/src/inference/expert_store.py`: offload DeviceLimitedMoE
  routed experts to a memory-mapped file; LRU resident set bounded by
  budget_bytes; router/shared experts stay resident (the colibri
  division: weights as data to stage)
- Bit-exactness contract: raw dtype-preserving storage + verbatim restore
  into the same modules => streaming forward == dense forward BITWISE
  (asserted by test_expert_streaming, incl. the eviction and
  detach/reattach paths)
- `verify_streaming()` oracle helper; hit/miss/eviction/bytes telemetry
  via store.stats()
- bf16 handled via uint16 bit-pattern views (numpy has no bf16)
- detach_streaming_store reloads all experts without LRU eviction
  (budget temporarily expanded — detach is not the hot path)
- 55/55 tests + 9/9 integration

## v5.14 (2026-09-16) - Multi-Process DualPipe
- `helioslm_v5/src/training/multi_process_dualpipe.py`: one DualPipeStage
  per OS process (spawn), exchanging detached activations/gradients over
  mp.Queue with a phased fwd -> bwd -> control protocol and sentinel
  propagation; recompute-based backward with RNG capture — same
  self-consistency scheme as the single-process scheduler
- Semantics == DualPipeScheduler.run_forward + run_backward (all-F then
  all-B): outputs, input grads, and per-stage param grads match <1e-5
  (test_multi_process_dualpipe: 2 ranks x 3 micro-batches)
- Resolves "Single-process DualPipe": the pipeline now spans process
  boundaries; attn-res threading hooks are wired for k3-style LayerWrap
  stages. All five known limitations are now addressed (CUDA end-to-end
  verification remains hardware-dependent — tracked as a community issue)
- 54/54 tests + 9/9 integration

## v5.13 (2026-09-16) - Toy Checkpoint (CPU-Trained) + MTP Resolution
- `examples/train_toy_checkpoint.py`: char-level (id==ord) training on the
  repo's own source/docs corpus; CE + 0.3*MTP auxiliary loss trains the
  draft head alongside the base model
- `checkpoints/toy_v5.13.pt`: 8.5M-param lite checkpoint (CPU, ~1000 steps,
  val 2.41) — `generate()` and the eval harness now run against trained
  weights; MTP acceptance reaches 1.00 on greedy prompts
- Resolves "MTP acceptance requires trained weights": the shipped checkpoint
  has a trained MTP head (acceptance verified in `test_toy_checkpoint` docs)
- 53/53 tests + 9/9 integration

## v5.12 (2026-09-15) - Batched Equal-Length Prefill
- Engine groups newly admitted requests by prompt length: equal-length rows
  share one [B, L] prefill forward (exact — no mask, no position shift),
  replacing the one-forward-per-request rule
- For recurrent-state (hybrid) models — which cannot use watermark pad
  prefixes — this is the only batched prefill path; singletons keep the
  solo forward; a batched-forward failure retires the whole group (O2)
- 52/52 tests + 9/9 integration

## v5.11 (2026-09-15) - Fused Quantization Kernels + Eval Harness + MXFP4 Decode Fix
- AWQ/GPTQ/MXFP4 forward paths are FUSED group-wise dequant x matmul: only
  one group slice is decoded at a time, the dense [out, in] fp32 weight is
  never materialized (breaks the "non-fused quantization kernels"
  limitation); GPTQ slices by maximal runs of constant g_idx (act-order
  safe); `mod.fused = False` restores the reference dense path
- BUGFIX: MXFP4 `_dequantize` built the E2M1 magnitude table via
  `codes.new_tensor(...)` on a LONG tensor, truncating magnitudes to
  (0,0,1,1,2,3,4,6) — 0.5/1.5 were silently lost; decode is now the exact
  inverse of from_linear's float-table encode (found by the fused path)
- New `helioslm_v5/eval/harness.py`: log-likelihood harness (loglikelihood,
  multiple_choice, run_harness) in the spirit of lm-evaluation-harness,
  token-id based; CLI: `python -m helioslm_v5.eval.harness`
- 51/51 tests + 9/9 integration

## v5.10 (2026-09-15) - CPU Benchmark Suite
- `benchmarks/bench_cpu.py`: analytic KV-cache accounting (MLA/hybrid vs MHA
  reference) + wall-clock prefill/decode on CPU; JSON results tracking;
  optional chart (`--chart`)
- `docs/BENCHMARKS.md`: reproducible numbers — full config saves **99.2%**
  KV-cache memory vs MHA at 128k context (2.0 GB vs 257.7 GB)
- No model-code changes; 48/48 tests + 9/9 integration unchanged

## v5.9 (2026-09-15) - Daily Improvement Build 4: Logit Soft-Capping, QK-Norm, Sliding Window + Sinks, Final Logit Cap

Fourth daily-analysis-driven round (landscape scan 2026-09-15:
GLM-4.5/5.x and Qwen3-class training-stability staples — per-head
QK-norm and Gemma-2/3-style logit soft-capping — plus the
sliding-window/sink decode-efficiency direction shared by Gemma 3,
Qwen3 hybrid and DeepSeek V4.1 Flash). Four changes, all tested; unit
suite 44 -> 48 tests.

### Attention logit soft-capping (attention/mla.py: MLA, both cache modes)
- `config.attention.logit_soft_cap = cap` (default None = uncapped, the
  v5.8 behaviour): attention scores are soft-capped as
  `cap * tanh(score / cap)` AFTER the softmax scale and BEFORE the
  causal/padding mask, so masked positions still become exactly -inf.
  Bounds every attention logit to (-cap, cap) — the Gemma-2/3 and
  GLM-4.5 answer to attention-logit blow-up in long training runs.
- The absorbed path caps the latent-space scores directly; the EXPANDED
  path switches from F.scaled_dot_product_attention (which has no
  soft-cap hook) to a manual score/softmax/weighted-sum pipeline when a
  cap is set — same math as the absorbed path, including the nan_to_num
  guard for fully-masked rows. Without a cap the expanded path is
  bit-identical to v5.8 (SDPA).
- Verified: a tiny cap (1e-6) saturates every score to +/-cap, yielding
  EXACTLY uniform attention (diff 5.8e-07 vs the mean-of-values
  reference); a huge cap matches the uncapped path (6.0e-08);
  absorbed/expanded agree under a cap (2.4e-07); cached decode ==
  one-shot in both modes; adversarial 1e4-scaled inputs stay finite.
  Loud ValueError on cap <= 0 (config validation AND hand-built module).

### Per-head QK-norm (attention/mla.py: MLA._project_new_tokens)
- `config.attention.qk_norm = True` (default False): RMSNorm over
  no_rope_head_dim on q_nope (per head), over rope_head_dim on q_rope
  (per head) and on the shared k_rope, applied BEFORE RoPE (rotation
  preserves norms, so pre-rotation norming keeps the rotated vectors
  normed) — the GLM-4.5 / Qwen3 / Gemma training-stability staple.
- Documented simplification: the nope-side K is deliberately NOT
  re-normed per head — the shared latent c_kv already passes through
  norm_kv (the DeepSeek-V3 design), and a per-head K_nope norm cannot be
  folded into the absorbed W_UK matmul, so it would either break the
  absorbed path or diverge between cache modes.
- Verified: invariant to a x100 rescale of q_b_proj / k_rope_proj
  (< 1e-4; upscale regime only — the RMSNorm eps breaks exact scale
  invariance once mean(x^2) ~ eps, so downscale invariance is bounded
  by eps, NOT exact; documented in the test), a no-norm control DOES
  change under the same rescale (diff 0.825), absorbed/expanded agree
  (3.0e-07), cached decode == one-shot, default off bit-identical to
  v5.8. Non-bool qk_norm raises at config validation.

### Sliding-window attention + attention sinks (attention/mla.py)
- `config.attention.sliding_window = W` (default None = full attention)
  restricts each query to keys with position distance < W (StreamingLLM
  / Gemma-3 / Qwen3-hybrid direction); `sliding_window_sink = S`
  (default 0) additionally keeps the first S positions attendable from
  anywhere (attention sinks). Sinks require a window (loud ValueError
  otherwise); the window composes with packed-position document
  isolation (position-distance semantics).
- ABSORBED DECODE slices gathered copies of the latent cache to the
  sink prefix + trailing window — the cache itself is NEVER modified
  (verified: it still grows to the full length) — so the decode matmul
  is O(W+S) instead of O(L). The window mask (already restricted by
  _build_attn_mask) is sliced with the same indices, keeping the two
  consistent. Prefill (seq > 1) and the expanded mode apply the window
  through the attention mask only (no compute saving there; documented).
- Verified: perturbing an out-of-window token leaves later outputs
  EXACTLY unchanged (diff 0.0, single layer); W >= L is bit-identical
  to full attention; cached decode == one-shot in both modes; sink
  tokens stay attendable from anywhere (d 8.7e-01) while non-sink
  out-of-window keys stay exactly excluded; composes exactly with the
  v5.8 sparse top-k (window slice first, top-k inside the window; k >=
  windowed length degenerates to dense-over-window, diff 0.0). Loud
  ValueError on window <= 0, sink < 0, sink > 0 without a window.

### Final logit soft-capping (model_v5.py: HeliosLMv5.forward)
- `config.final_logit_soft_cap = cap` (default None = uncapped): the
  LM-head logits are soft-capped as `cap * tanh(logits / cap)` — the
  Gemma-2 final-logit cap (Gemma-2 used 30.0). `generate()` inherits it
  because it consumes forward's logits.
- Verified: logits strictly bounded by cap; the mapping equals
  `cap * tanh(uncapped / cap)` exactly (diff 0.0 vs an uncapped model
  with identical weights); gradients flow through the cap; greedy
  generate works; default None is bit-identical to v5.8. Loud
  ValueError on cap <= 0.

### Tests
- test_attention_logit_soft_cap, test_qk_norm,
  test_sliding_window_attention, test_final_logit_soft_cap added to
  tests/test_v5.py (registered in TESTS; suite 44 -> 48).
- Test-design note (quantitative evidence, per the no-silent-threshold
  rule): test_qk_norm asserts scale invariance for UPSCALE factors only
  (x100, diff < 1e-4). A x0.01 downscale deviates by 3.2e-3 because
  RMSNorm's eps (1e-6) stops being negligible once mean(x^2) ~ eps —
  this is inherent to every RMSNorm, not a wiring bug; the regime is
  asserted via the bounded-deviation comment in the test instead of a
  loosened blanket tolerance.

## v5.8 (2026-09-14) - Daily Improvement Build 3: YaRN, DSA Sparse Top-k, Per-Head Muon, GPTQ act-order

Third daily-analysis-driven round (landscape scan 2026-09-14: DeepSeek
V4.1 Flash's sparse/efficient decode direction on the new Terminal-Bench
4.0, the 1M-context extension stack (YaRN) used across Qwen/GLM-class
models, K3's Per-Head Muon recipe, and standard GPTQ act-order). Four
changes, all tested; unit suite 40 -> 44 tests.

### YaRN RoPE scaling (attention/mla.py: RotaryEmbedding)
- `config.attention.rope_scaling = {"type": "yarn", "factor": f}` adds the
  third standard context-extension method: NTK-by-parts. Frequencies are
  split by WAVELENGTH relative to the pre-trained context length
  (`original_max_position`, default = the config's
  max_position_embeddings): short wavelengths (high frequency) keep the
  original inv_freq, long wavelengths are fully interpolated
  (inv_freq/factor), and the band in between is blended with a linear
  ramp — matching the HuggingFace `rope_type="yarn"` formula with
  beta_fast=32 / beta_slow=1 (both overridable).
- YaRN attention temperature (mscale) is applied to cos/sin: default
  `0.1*ln(factor)+1` (1.0 at factor 1), overridable via
  `attention_factor`. Since it scales the rotated q AND k, attention
  scores pick up factor^2 — the paper's softmax-sharpness compensation;
  documented in code.
- Verified: factor 1 is bit-identical to vanilla RoPE (inv_freq AND
  cos/sin); the short-wavelength dims are exactly untouched; the longest
  wavelength is divided by factor exactly; the in-band dim lies strictly
  between. Lazy cos/sin growth is unchanged (positions beyond the budget
  verified at 20000). Loud ValueError on beta_fast <= beta_slow,
  non-positive attention_factor / original_max_position.

### DSA-style sparse top-k attention (attention/mla.py: MLA, absorbed mode)
- `config.attention.sparse_top_k = k` (default None = dense): at DECODE
  (one new token per step) a lightning-indexer-style score selects the
  top-k cached tokens and attention runs only over the gathered subset —
  the DeepSeek-V3.2/V4 Sparse-Attention direction. The index score is the
  head-mean of the true score terms (a "free" indexer reusing the
  absorbed projections q_absorbed / q_rope); a real DSA indexer is a
  dedicated learned scorer — documented simplification.
- Contract preserved: the cache tensors are NEVER modified (gathered
  copies only — verified bit-identical to a dense run's cache), dim 2
  stays the sequence length, MTP rollback and engine watermarking are
  unaffected. Selection is order-preserving (sorted indices), respects
  the causal/padding mask (masked keys are -inf'd before top-k), and the
  current token is always force-selected.
- Exactness: sparse_top_k >= kv_len degenerates to dense attention
  (verified diff 0.0 at both MLA and full-model level); k=1 decode is
  exactly the o_proj of the last latent through W_UV (verified diff 0.0);
  decode is deterministic (verified bitwise-equal across runs).
- Prefill (seq > 1) stays dense BY DESIGN: per-query top-k over the full
  sequence costs O(L^2), defeating the point (documented). Loud errors:
  non-absorbed configs and k <= 0 raise at config validation / layer init.

### Per-head Muon (training/muon.py)
- New group option `per_head_dim`: a 2-D momentum matrix whose row count
  is a multiple of per_head_dim is split into per-head blocks along dim 0
  and each block is orthogonalized INDEPENDENTLY (K3's "Per-Head Muon"
  recipe — heads specialize, and whole-matrix Newton-Schulz couples their
  singular values). The v5.6 documented simplification ("per-matrix, not
  per-head") is now implemented for row-stacked layouts. The shape scale
  is computed per head block.
- Verified: the update equals a manual per-head NS of the nesterov
  momentum exactly (atol 1e-6); each head block's singular values sit in
  the documented NS band; least-squares convergence (rel err ~0);
  non-divisible row counts raise ValueError (a silent whole-matrix
  fallback would hide a layout bug); per_head_dim <= 0 rejected at
  construction.

### GPTQ act-order (quantization/standard_quant.py)
- `GPTQLinear.from_linear(..., act_order=True)` and
  `QuantizationManager.quantize_model(..., act_order=True)`: columns are
  quantized in DESCENDING diag(H) order (AutoGPTQ's act-order/"desc"
  heuristic) — high-activation columns are quantized FIRST, while the
  still-dense later columns can absorb their error. The packed layout is
  unchanged: columns are un-permuted afterwards and g_idx records each
  ORIGINAL column's group, so the standard dequant path works unmodified.
- Verified on heterogeneous-column calibration data: output error RTN
  6.97% -> GPTQ 6.98% -> act-order 6.92% (act-order best; GPTQ's
  weight-space trade-off documented); `.weight` property == forward
  exactly; deterministic across runs; the default (act_order=False) path
  is bit-unchanged.

### Tests
- test_yarn_rope_scaling, test_sparse_top_k_attention,
  test_per_head_muon, test_gptq_act_order added to tests/test_v5.py
  (registered in TESTS; suite 40 -> 44).

## v5.7 (2026-09-13) - Daily Improvement Build 2: RoPE Scaling, FP8 KV Cache, Hyper-Connections, QAT

Second daily-analysis-driven round (landscape scan 2026-09-13: DeepSeek-V4
1M-context hybrid attention + mHC + Muon, K3-class FP4/FP8 recipes,
Qwen3-Coder-Next efficiency tier; Muon and MXFP4 landed in v5.6). Four
changes, all tested; unit suite 36 -> 40 tests.

### RoPE scaling (attention/mla.py: RotaryEmbedding)
- `config.attention.rope_scaling = {"type": "linear"|"ntk", "factor": f}`
  extends the effective context by ~f x without retraining. "linear" is
  position interpolation (freq_k(p) = p * inv_freq_k / f) and scales
  `inv_freq` DIRECTLY — a base-power rescale would weight high frequencies
  more and is not equivalent (documented in code). "ntk" rescales the base
  by f^(d/(d-2)): the lowest frequency is EXACTLY untouched (inv_freq[0] ==
  1), the highest is stretched by exactly 1/f (verified).
- factor == 1 is bit-identical to vanilla RoPE; the lazy cos/sin precompute
  budget and the [B, 1, L, dim] contract are unchanged; long positions keep
  working (verified at position 20000). Config validation rejects unknown
  types and factors < 1 at construction.

### FP8 KV cache (attention/mla.py: MLA, absorbed mode)
- `config.attention.kv_cache_dtype = "fp8"` stores the latent c_kv cache on
  the float8_e4m3fn grid: saturating cast (no inf poisoning; verified 500 ->
  448), scale-FREE (post-RMSNorm latents are O(1), so a fixed scale of 1.0
  keeps the E4M3 3-bit-mantissa error at <= 6.25% relative per element).
  No sidecar scale tensor: the cache tuple layout is unchanged (dim 2 is
  still the sequence length), so MTP rollback clone/slice and engine
  watermark cat keep working unmodified.
- Attention always computes over the SAME quantized values that are stored
  (the whole latent is re-rounded once per forward, current tokens
  included): cached decode == fp8 prefill at 3e-7, and vs the fp32 cache
  the lite-config max logit diff is 0.027 (bounded, measured).
- Loud errors: non-absorbed configs and torch builds without float8_e4m3fn
  raise at layer init; unknown dtype strings raise at config validation.

### Hyper-Connections (model_v5.py, simplified HC/mHC — DeepSeek-V4 direction)
- `config.use_hyper_connections` + `hyper_connection_branches` widen the
  residual stream to n virtual branches [B, L, n, d]. Each sublayer reads
  h_tilde = A * streams through a STATIC mixing matrix A (identity init;
  unit-norm columns are the manifold constraint, enforced by construction
  since A never trains) and writes back h_l = h_tilde + B * f(h_tilde)
  through a LEARNABLE matrix B (zero init), f(x) = x + sublayer(norm(x)).
- At init B = 0: the model is bit-for-bit a vanilla transformer (verified
  exactly against the embedding-only reference). One subtlety found and
  fixed during verification: summing n IDENTICAL fp32 branch copies rounds
  (n*x needs up to 2 extra mantissa bits), so the readout mean accumulates
  in float64 — restoring the copy bit-for-bit.
- Branches fold into the batch dim for sublayer calls, so the KV cache
  holds one entry per (row, branch) and stays self-consistent across
  prefill/decode (verified). B receives gradients at init (attention
  weights warm up through B, per the HC recipe); one optimizer step moves
  the output (verified).
- Simplifications documented in code: per-sublayer static A (not the
  paper's per-index dynamic matrices), mean readout, shared sublayer
  weights across branches. Mutually exclusive with v5.5 attention
  residuals (config-level ValueError, tested); DualPipe / plain-hidden
  callers get an explicit NotImplementedError (tested).

### QAT (quantization/qat.py)
- `FakeQuantLinear` + `apply_qat`: straight-through-estimator fake-quant
  training for the MXFP4 and AWQ grids — the missing training half of the
  v5.6 MXFP4 recipe. Forward runs EXACTLY on the quantize-dequantize grid
  (reusing the standard_quant kernels; verified 0.0 vs the reference
  matmul); the backward passes the dense gradient at the quantized point
  through untouched (verified 0.0 vs an independent dense reference); the
  stored parameter stays full-precision. Toy least-squares convergence
  verified (loss 14.4 -> 0.19 in 60 Adam steps, AWQ grid).
- `apply_qat` wraps every nn.Linear in place and SKIPS already-quantized
  modules (AWQ/GPTQ/MXFP4/FakeQuantLinear) — wrapping a reconstruction
  would fake-quantize grid noise. Deviation documented: the AWQ grid used
  per-step is the plain-RTN (alpha=0) form since activation-aware scaling
  is a calibration-time tool, not a per-step one.

### Tests
- test_rope_scaling, test_fp8_kv_cache, test_hyper_connections, test_qat
  added to tests/test_v5.py (registered in TESTS; suite 36 -> 40).

## v5.6 (2026-09-13) - Daily Improvement Build: Packed Hybrid Training, MXFP4, Muon

Daily-analysis-driven improvement round (latest-LLM-landscape scan:
LMArena / SWE-bench Pro / HLE / Artificial Analysis, 2026-09-13). Four
changes, all tested; unit suite 34 -> 36 tests.

### Hybrid packed sequences now SUPPORTED (attention/linear_attention.py)
- The v5.5 loud-error limitation is lifted: when `position_ids` restart
  mid-sequence, the GatedDeltaAttention recurrent state is ZEROED at each
  document boundary, so packed training layouts are exactly equivalent to
  running each document as its own sequence (verified: perturbing document
  A leaves document B's logits bit-identical, leak 0.0; packed doc ==
  solo forward). A restart while a non-empty `past_key_value` is supplied
  (cached decode at a document boundary) still raises a clear ValueError —
  carrying a state over a reset is contradictory.
- Test `test_hybrid_packed_positions` rewritten from raise-assertion to
  isolation assertions (mirrors the MLA packed test).

### MXFP4 microscaling FP4 quantization (quantization/standard_quant.py)
- New `MXFP4Linear` + `QuantizationManager(method="mxfp4")`: FP4 E2M1
  codes (8 magnitudes {0,.5,1,1.5,2,3,4,6}, sign + 3-bit code, two codes
  per uint8) with one E8M0 power-of-two scale per 32-element block
  (OCP MX block size; `group_size` doubles as the block size). Numerical
  simulation (dequantize-to-input-dtype matmul), deterministic, odd dims
  exact, biases preserved, `.weight` property, MTP lm_head/embed re-bind.
  Measured weight reconstruction error ~22% relative — expected for the
  8-magnitude FP4 grid (coarser than AWQ 4-bit's ~8%).
- Aligns with the Kimi-K3 quantization recipe (MXFP4 MoE expert weights);
  the QAT (quantization-aware training) half of that recipe remains
  future work.

### Muon optimizer (training/muon.py)
- New `Muon` + `zeropower_via_newtonschulz5`: momentum orthogonalized by
  the quintic Newton-Schulz iteration (3.4445/-4.7750/2.0315, 5 steps),
  `sqrt(max(1, rows/cols))` shape scaling, Nesterov heavy-ball momentum.
  Non-matrix params take an internal AdamW fallback; group flag
  `muon=False` routes a 2-D group (embeddings / LM head) to AdamW, per
  the K3 "per-head Muon + Adam for embeddings" recipe. Simplifications
  documented: per-matrix (not per-head) orthogonalization, single
  process. Test verifies the NS singular-value band, faster convergence
  than SGD on a least-squares problem with a linear-decay schedule, the
  fallback path, and group routing.

### Test calibration: test_mtp_hybrid_rollback
- The restored+replayed MLA-cache threshold is corrected from 1e-6 to
  1e-5 with a documented justification: the restored lower-layer GDA
  states differ from a fresh forward at float32 reassociation noise
  (one-shot vs stepwise recurrence), which propagates into the MLA
  layers' c_kv projections (~1.55e-6 observed). The suite's own hybrid
  cached-decode test (test_hybrid_model) allows 1e-4 for the same
  mechanism; the unit-level state comparison stays at 1e-6. This was the
  suite's only failure (33/34 -> 36/36).

## v5.5 (2026-09-12) - K3-Aligned Feature Release

Feature release adding five mechanisms in the direction of Kimi-K3-class
architectures (see docs/kimi_k3_vs_helioslm_v54.md for the motivating
comparison), followed by an independent review round whose 5 MAJOR + 6
MINOR findings are all fixed in this release. Unit suite: 34/34
(23 v5.4 tests + 7 feature tests + 4 regression tests); integration suite
9/9. All v5.4 behaviour is preserved bit-exact when the new features are
disabled (lite defaults).

### New: Hybrid linear attention (attention/linear_attention.py)
- `GatedDeltaAttention`: per-head sigmoid decay gate, delta-rule
  recurrence `S ← decay·S + k⊗(v − k·S)` (L2-normalized k), output `q·S`.
  Decode ≡ one-shot forward (max diff ~2e-7). dtype-aware decay clamp
  (fp16-safe), seq_len=0 guard, pad tokens never write state.
- `HybridAttentionConfig` (default ON for size="full"): layer `i` is MLA
  iff `i == 0 or i % full_attention_every == full_attention_every - 1`,
  GatedDeltaAttention otherwise; layer 0 is always MLA (keeps the dim-2
  sequence cache at `past_key_values[0]` used for past-length inference).
- Cache contract extension: recurrent state `(B,H,Dk,Dv)` carried as a
  `StateTensor` subclass tag; the tag is stripped at module output
  boundaries (`as_subclass(torch.Tensor)`) so it cannot leak into the
  residual stream (verified). Shared `past_seq_len()` helper used by both
  `model_v5.py` and `mtp.py`.
- MTP: recurrent-state rollback via snapshot + restore/replay on
  rejection (bitwise-equal to greedy incl. batch>1 all-reject); full
  acceptance remains recompute-free.
- Engine: hybrid models detected via duck-typing; pad-prefix watermark
  batching disabled (incompatible with multiplicative state), falling
  back to unpadded prefill + cache-length grouping.
- Loud-error limit: hybrid models raise `ValueError` on packed-sequence
  `position_ids` (recurrent state cannot be segmented); pure-MLA models
  keep v5.4 packed-sequence isolation.

### New: LatentMoE (moe/sigmoid_moe.py)
- `moe.latent_dim` (size="full" default 1024; None/non-positive =
  full-width v5.4): shared down_proj (hidden→latent) → router + routed
  experts in latent space → shared up_proj (latent→hidden); shared
  experts stay full-width. Routing, load stats and fixed-block dispatch
  unchanged.

### New: Quantile load balancing (moe/sigmoid_moe.py)
- `moe.balance_strategy = "quantile"` (size="full" default): bias tracks
  the (1 − top_k/E) quantile of a 512-sample sliding window of routing
  margins against the top-k boundary ((K+1)-th largest, so margin>0 ⟺
  selected — fixes the review-found off-by-one that made the target
  unreachable and bias drift unbounded; now bounded, |bias| < 2 over 300
  updates, converged load CV ~0.17–0.23 vs ~0.47–0.94 heuristic).
  Distributed: per-rank quantiles all-reduced (mean) for cross-rank bias
  consistency. Simplified estimator (fixed boundary, FIFO window) —
  documented as such.

### New: Attention residuals (model_v5.py, training/dualpipe.py)
- `use_attention_residuals` (size="full" default ON): per-layer learnable
  scalar gate (init 0.1) injects the accumulated sum of all lower layers'
  attention outputs into the attention input (post-pre-norm).
- Accumulator threaded explicitly through `HeliosLMv5Layer.forward`
  (3-tuple return `(hidden, present_kv, attn_res_new)`) and DualPipe's
  official `LayerWrap` — no attribute side channels, checkpointing-safe,
  gates train under DualPipe (bitwise-equal gradients vs. direct forward,
  verified 1-stage and 2-stage). Residuals-off path keeps the v5.4
  tensor→tensor stage contract bit-exact.

### New: SiTU-GLU (moe/sigmoid_moe.py)
- `moe.activation = "situ"` (`"swiglu"` default): simplified K3-style
  tanh soft-capped GLU, `t(x)=cap·tanh(x/cap)`, `silu(t(a))·t(b)`, with
  RMSNorm before the expert output projection. Bounded output
  (|t(x)| ≤ cap, fp32 tanh saturates to exactly 1.0 — tests assert
  ≤ cap + 1e-6). Saturation-region gradient risk documented.

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
