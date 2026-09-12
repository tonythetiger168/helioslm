# helioslm v5.0 — Code Review Report

**Date:** 2026-09-12
**Scope:** `/mnt/agents/output/helioslm_v5/` — 14 Python modules, ~1,850 LOC
**Method:** 4 parallel specialist reviews (model core / training / inference / multimodal-quant-tests-README), with empirical runtime verification of key findings on torch 2.8 + independent orchestrator spot-checks (all confirmed).

---

## Executive Summary

**Overall verdict: NOT FUNCTIONAL / NOT TRAINING-READY.** The v5.0 codebase is best understood as an architectural skeleton: module files exist and APIs are sketched, but nearly every headline feature is either broken at runtime, mathematically wrong, or a stub behind a real-sounding name.

| Severity | Count | Theme |
|---|---|---|
| **CRITICAL** | 14 | Decode path broken (MLA cache masking + RoPE positions), MTP crashes + no verification, DualPipe computes no gradients, GRPO loss has no grad path + misaligned rewards, FP8 grads never zeroed, audio encoder crashes on every call, GPTQ/AWQ are random-weight stubs, V=K copy with RoPE leakage |
| **MAJOR** | 27 | Claimed techniques absent (weight absorption, aux-free balancing, DualPipe schedule, FP8 recipe, CoW fork, continuous batching), MoE distributed path broken, test suite cannot fail, README materially overclaims |
| **MINOR** | ~25 | Dead code, vacuous assertions, device/dtype hazards, docstring drift |

Key systemic patterns:
1. **Test suite is not a gate** — `tests/test_v5.py` swallows all exceptions and always exits 0; reproduced: 3 failures + 2 OOM-kills out of 8 tests, all hidden.
2. **README overclaims pervasively** — every quantitative performance claim (93.3% KV reduction, 85–90% MTP acceptance, +40% paged attention, <200ms audio latency, FP8 −50% cost, DualPipe 95% GPU util) is unsupported by any code or benchmark; several are directly contradicted by the code.
3. **Reference-design drift** — modules claim DeepSeek-V3 MLA / DualPipe / MTP / GRPO / FP8 recipes, but each deviates in ways that break the defining property of the technique.

---

## 1. Model Core (`model_v5.py`, `attention/mla.py`, `moe/sigmoid_moe.py`, `configs/config_v5.py`)

### CRITICAL

**C1. Cached decoding attends only to the first cached key — `is_causal` misalignment** — `src/attention/mla.py:168-172`
With `past_key_value` present, `q_len=1` vs `kv_len=past+1`, but SDPA `is_causal=True` builds a top-left-aligned mask → each decode query attends **only to key 0**. Empirically verified (output equals `v[...,0,:]`). Every generated token ignores the KV cache except token 0.
→ Fix: explicit bottom-right-aligned attn_mask whenever past exists.

**C2. RoPE positions restart at 0 during cached decoding** — `mla.py:136-141`, `model_v5.py:113-141`
`cos/sin` sliced at positions `0..seq-1` for new tokens only; `position_ids` arg exists but `model_v5.forward` never computes or passes it. Decode steps rotate q/k as if at position 0.
→ Fix: thread `position_ids`/past_len through layer → MLA.

**C3. Value tensor is a copy of K including RoPE-rotated dims — not MLA** — `mla.py:147-149, 164-165`
`full_v = cat([full_kv_nope, full_k_rope])` makes V identical to K, including position-dependent rotations. DeepSeek MLA uses a separate V up-projection with no RoPE. Attention output becomes a weighted average of keys; no learned K/V distinction; relative-position invariance broken.
→ Fix: split `kv_b_proj` into K_nope and V heads; V never rotated.

### MAJOR

- **M1. "Weight absorption" claimed, absent; O(L) re-compression per step** — `mla.py:6,74-76` vs `mla.py:160`: cache stores `c_kv` and re-applies `kv_b_proj` to the entire cached latent every decode step — the exact cost absorption exists to remove.
- **M2. Dead compute every forward** — `mla.py:128-129, 144-149`: K/V built and discarded; `kv_b_proj` applied twice to the same latents.
- **M3. ~8 GB of RoPE buffers** — `mla.py:107` + `config_v5.py:65`: cos/sin precomputed for 1M positions × 48 layers (~168 MB/layer) at construction.
- **M4. `hidden_size=4096 % num_heads=96 ≠ 0`** — `mla.py:83` + `config_v5.py:9,66`: silent head_dim truncation (42 vs 42.67); 64 hidden channels silently dropped from attention.
- **M5. Missing pre-attention norm; MoE input normed twice** — `model_v5.py:42-54` + `sigmoid_moe.py:84`: MLA runs on raw hidden states; `post_attn_norm` + MoE internal `input_norm` double-norm the MoE input.
- **M6. Sigmoid-gate bias corrupts gate weights; bias is gradient-trained** — `sigmoid_moe.py:41-42, 87-88`: DeepSeek-V3 aux-free design adds bias **only for top-k selection**, gates with bias-free score, and updates bias via load-error heuristic with `requires_grad_(False)`. None of that exists despite docstring claims.
- **M7. Distributed MoE path crashes** — `sigmoid_moe.py:92-102, 106`: CPU-tensor device mismatch on GPU; `IndexError` when experts-per-device < top_k.
- **M8. "Vectorized" dispatch is a 2,048-iteration loop with GPU syncs** — `sigmoid_moe.py:104-116`: `top_k × num_experts` Python loop with `mask.any()` syncs; dominates training time.
- **M9. `generate()` crashes for batch > 1** — `model_v5.py:176`: `next_token.item()` on `[B,1]`; no per-sequence EOS handling.
- **M10. PagedAttention block manager constructed but never used** — `model_v5.py:94-101, 128-134`: advertised paged-KV integration doesn't exist in the model path.
- **M11. No attention/padding mask support anywhere** — `model_v5.py:113`, `mla.py:113`: padded batches attend to PAD tokens.

### MINOR

- "93.3% KV-cache reduction" claim false for this config — real figure ≈ 20.8% (repo's own `test_mla` prints this).
- `apply_rotary` zero-pads cos/sin silently (`mla.py:62-64`); unused `layer_idx` (`mla.py:113`).
- `model_v5.py:131`: `len(past_key_values)` TypeError if past passed with `use_cache=False`.
- `model_v5.py:161`: `temperature=0` → inf/nan, no greedy guard.
- `model_v5.py:118-126`: vision/audio prefix tokens get no modality/position bookkeeping; LM logits computed over them.
- Dead config fields: `intermediate_size`, `attention.type`, `moe.type`, `moe.device_group_size` (hardcoded 8 at `sigmoid_moe.py:68`); `task_type` arg unused (`sigmoid_moe.py:74`).

---

## 2. Training (`training/fp8_trainer.py`, `training/dualpipe.py`, `training/grpo.py`)

### CRITICAL

**C4. DualPipe backward never computes parameter gradients** — `dualpipe.py:54, 73-74`
`run_forward` stores `out.detach().requires_grad_(True)`, severing the graph; `run_backward` calls `.backward(grad)` on the detached leaf → verified: `stage.weight.grad is None` after backward. **Training runs without error but no parameter ever learns.**
→ Fix: store live outputs (or recompute stage forward in backward) and `torch.autograd.backward` on the connected graph.

**C5. GRPO loss has no gradient path; `loss.backward()` raises** — `grpo.py:147-151, 115, 136`
`_sample_response` returns a hardcoded string and `torch.tensor(0.0)`; the model is never called → `loss.requires_grad=False` → `RuntimeError` on `.backward()` (verified). No new-policy log-prob recomputation exists anywhere.
→ Fix: sample under `no_grad` (store old log-probs), then a grad-carrying forward for the ratio.

**C6. GRPO rewards scored against wrong answers** — `grpo.py:93-106`
`answers * group_size` interleaves `[a1,a2,...]` instead of `[a1,a1,...,a2,a2,...]` → for batch>1 every response except the first per group is scored against another question's answer (verified). Entire update is garbage.
→ Fix: `[a for a in answers for _ in range(group_size)]`.

**C7. FP8 trainer never zeroes gradients** — `fp8_trainer.py:119-135`
No `zero_grad()` anywhere → gradients accumulate across all steps → runaway effective step size.

### MAJOR

- **M12. GRPO ratio uses ref model instead of old policy** — `grpo.py:123`: `ratio = exp(logprobs − ref_logprobs)`; must be `exp(new − old_policy)`. Ref model belongs only in the KL term.
- **M13. GRPO KL is the wrong estimator and can go negative** — `grpo.py:129`: mean of `log π − log π_ref` can be negative → minimizing loss actively rewards divergence. Use DeepSeekMath's non-negative k3 estimator.
- **M14. DualPipe implements none of the DualPipe schedule** — `dualpipe.py:81-98`: purely sequential forward-then-backward; no F/B interleaving, no p2p, no distributed code at all. "5% bubble" claim unsupported. `backward_buffers` is dead code.
- **M15. `_quantize_to_fp8` performs no quantization** — `fp8_trainer.py:35-49`: `x/scale` then `*scale` = identity + clamp; no FP8 grid rounding; matmul in full precision. Numerically a no-op.
- **M16. FP8 recipe deviations** — `fp8_trainer.py:17,29,67-68`: no gradient quantization (no backward hook), `weight_scale` never updated, `fp8_format` param dead, no per-tile scaling.
- **M17. FP8 "trainer" has no optimizer** — `fp8_trainer.py:127-133`: manual SGD; no AdamW/momentum/wd/schedule/clipping; crashes on `param.grad is None`; master-weight aliasing after step 1 provides no extra precision.
- **M18. GRPO ref model is a stub, never frozen** — `grpo.py:25, 153-157`: `_get_ref_logprobs` returns zeros; `ref_model` never called/eval'd/frozen.

### MINOR

- `fp8_trainer.py:64`: `max(scale, 1e-8)` → Python float assigned to buffer → `TypeError` on all-zero input batch (verified).
- `fp8_trainer.py:53`: `.item()` forces GPU sync each forward; scaling state not checkpointed (plain attr).
- `dualpipe.py:100`: `sum(losses)` over microbatches inflates gradient ×N vs conventional mean.
- `dualpipe.py:61`: "activation checkpointing" comment — nothing recomputed (this is why C4 can't be fixed without recompute).
- `dualpipe.py:109-123`: `ExpertParallelism` — non-divisible expert counts yield invalid device ids; `.item()` per element; `all_to_all` is a pass-through stub.
- `grpo.py:114-126`: sequence-level ratio/clip vs DeepSeekMath per-token ratios with length normalization — objective deviation.
- `grpo.py:66,115`: CPU tensors → device mismatch with a real GPU model.

---

## 3. Inference (`inference/mtp.py`, `inference/paged_attention.py`, `inference/vllm_engine.py`)

### CRITICAL

**C8. MTPDecoder unpacks a 4-tuple the model never returns** — `mtp.py:130`
`main_logits, main_hidden, _, _ = self.main_model(...)` but `HeliosLMv5.forward` returns a 2-tuple and never returns hidden states → `ValueError` on first iteration (verified). The entire `use_mtp=True` generate path is dead code.

**C9. MTP decode shapes collapse to an empty sequence** — `mtp.py:63-70, 141-143`
Decode path feeds `[B,L,H]` hidden + `[B,1]` token; training-time shift slices `next_emb[:,1:]` to length 0; `min_len` reconciliation empties both → `IndexError` at `mtp_logits[:,-1,:]` (verified).

**C10. No verification or rollback — speculative decoding absent** — `mtp.py:147-151`
Draft tokens appended unconditionally; no accept/reject sampling, no distribution comparison, no rollback. Output distribution ≠ main model's — the defining guarantee of speculative decoding is violated; "85–90% acceptance" never measured.

**C11. PagedAttention hardcoded bf16 cache crashes in fp32/fp16** — `paged_attention.py:52-58` vs `111-114,163`
Read path matmuls model-dtype queries against bf16 cache → `RuntimeError: expected m1 and m2 to have the same dtype`. The repo's own `test_paged_attention` fails with exactly this on CPU/fp32.

### MAJOR

- **M19. Off-by-one: new K/V written at `context_len−1`, clobbering the last token** — `paged_attention.py:135-136`; nothing calls `append_tokens` before forward, so each step overwrites and the sequence never grows.
- **M20. No block allocation on the write path** — `paged_attention.py:130-140`: `append_tokens` is dead code; block-boundary crossings → `IndexError` or silent stagnation.
- **M21. "Copy-on-write" fork has no CoW and double-frees** — `paged_attention.py:85-99`: no refcounts; `free()` returns shared blocks while siblings still reference them; partial-block appends silently mutate siblings.
- **M22. MTP transformer is bidirectional — future-token leakage** — `mtp.py:29-35, 76`: no causal mask; position j attends to its own prediction target; module learns to copy. Dropout also active in generation (no `eval()`).
- **M23. MTP chaining/depth ≠ DeepSeek MTP** — `mtp.py:138-145, 63-64`: hidden state not chained module-to-module; `module_index` unused so all modules predict the same depth (t+2, not t+2/t+3); docstring contradicts the shift.
- **M24. MTP embedding/head not shared; no MTP training loss exists** — `mtp.py:26,38`: fresh random embedding/head ("shared" comment is false); zero MTP loss in `src/training/` → MTP weights stay at random init forever.
- **M25. VLLMEngine never uses BlockManager/PagedAttention; full-prefix recompute** — `vllm_engine.py:28-32, 54-58`: no allocate/append/free; concatenates full token ids every step, no cache → O(L²) recompute, crashes on unequal lengths; "continuous batching / prefix caching" claims false.
- **M26. `Request` protocol undefined anywhere in the repo** — `vllm_engine.py:42-64`: `is_done()/get_input_ids()/append_token()` have no implementing class; no `add_request` API. Engine unusable as-is.
- **M27. HPA recommendation unit mismatch** — `vllm_engine.py:92,100-108`: GB values compared against fractions 0.85/0.4 → scale-up always fires; replica counts derived from metrics-sample history length, not load.

### MINOR

- `mtp.py:128`: `max_new_tokens // (num_mtp+1)` drops remainder tokens; small budgets silently generate nothing.
- `mtp.py:130`: returned past-KV discarded; full sequence recomputed per outer iteration; unverified drafts fed back.
- `mtp.py:57-59`: `next_token_ids=None` branch uses zeros embedding → meaningless logits (this broken path is what `MTPModule.generate` relies on).
- `paged_attention.py:144-167`: per-token Python gather + per-sequence loop; no vectorization; no RoPE applied to q/k; single global cache (not per-layer) — cannot reproduce a causal LM's numerics even when it runs.
- `paged_attention.py:51-59, 47`: cache shape fixed by first allocation; `free_blocks.pop(0)` is O(n).
- `vllm_engine.py:69-71`: hardcoded temperature 0.7; sampling params ignored.
- `vllm_engine.py:74-93`: "Prometheus" docstring but no prometheus_client; declared metrics never collected.

---

## 4. Multimodal, Quantization, Tests, README

### CRITICAL

**C12. `StreamingAudioEncoder.forward` crashes on every call** — `streaming_encoder.py:43,65,70`
`conv_state` registered as `[2, n_mels, 2]` → `conv_state[0]` is 2-D, concatenated with 3-D input → `RuntimeError` (verified). The module has never successfully run; failure hidden by the test runner.

**C13. `F.gelu` used but `F` never imported** — `streaming_encoder.py:6-7,66,71` → `NameError` (verified after patching C12; currently masked by it).

**C14. `GPTQLinear.forward` is a stub returning random weights** — `standard_quant.py:63-66`
`F.linear(x, torch.randn(...) * 0.01)` — never touches `qweight/qzeros/scales`, non-deterministic across calls (verified), CPU tensor on GPU → device mismatch. No GPTQ algorithm exists anywhere. Additionally `QuantizationManager.quantize_model` swaps real `nn.Linear`s for random-weight `AWQLinear`s **without ever reading the original weights** (bias silently dropped) — quantizing a model destroys it (verified). FP8 path silently no-ops (`METHODS["fp8"]=None`).

### MAJOR

- **M28. Conv-state carry semantics wrong even after C12/C13** — `streaming_encoder.py:67,72`: stores conv *output* tail as next chunk's conv *input* state — wrong semantics and wrong channel count.
- **M29. `process_stream` chunk math off by 160×** — `streaming_encoder.py:94`: `chunk_ms * 16` → a "500 ms chunk" is 80 s of audio; contradicts "<200 ms latency" claim.
- **M30. No cross-chunk state; non-causal transformer** — `streaming_encoder.py:44,49,78`: `prev_chunk` registered but never used; bidirectional attention within chunks. "Stateful streaming" claim false.
- **M31. NaViT position mapping breaks "arbitrary resolution"** — `navit.py:33,67,97`: `pos_id = i*32+j` hardcodes a 32-column grid → `IndexError` for 512×512 (verified), silent position-id collisions for width > 448 px (615/1168 patches collide, verified).
- **M32. `AWQLinear._unpack` crashes for odd `in_features`** — `standard_quant.py:22,32` (verified).
- **M33. `forward_packed` allocates on CPU regardless of input device** — `navit.py:106-107` → device mismatch on CUDA; this is the only heterogeneous-size path.
- **M34. Test runner cannot fail** — `tests/test_v5.py:102-109`: all exceptions swallowed, always exits 0. Reproduced: `test_streaming_audio` FAIL, `test_mtp` FAIL (`embed_dim must be divisible by num_heads`), `test_sigmoid_moe` FAIL (no NVIDIA driver), `test_paged_attention` + `test_v5_model` OOM-killed — CI would report success. Also `size="lite"` is accepted but ignored → full suite OOMs.
- **M35. Quantization tests vacuous; GPTQ/FP8/manager zero coverage** — `tests/test_v5.py:77-83`: shape-only assertion on random weights passes precisely because weights are garbage; no quantization-error, determinism, or fp8-path tests.
- **M36. NaViT packing untested and isn't packing** — `tests/test_v5.py:57-64` + `navit.py:104-111`: only uniform 224×224 tested; `forward_packed` is zero-padding, not sequence packing.

### MINOR

- `navit.py:48-49` vs `forward` signature: single tensor requires identical sizes despite "variable H, W" docstring; `forward_packed` skips final norm and returns a different type (`navit.py:114-115` vs `75`).
- `streaming_encoder.py:24`: unused `chunk_size=160` with contradictory comment; hardcoded `nhead/num_layers` ignore config (`:34,38`); docstring claims online mel computation — none exists (`:16`); `reset_state` shape must track C12/C13 fixes (`:49`).
- `standard_quant.py:31,66`: CPU allocations; `METHODS` dict unused, unknown methods silently no-op (`:72-90`); O(n²) parent-map rebuild (`:96`).
- `tests/test_v5.py:3`: hardcoded `sys.path.insert("/mnt/agents/output")`; all assertions shape-only (no numerics, determinism, or chunk-continuity checks).

---

## 5. README Claims Audit

| Claim (README line) | Status | Evidence |
|---|---|---|
| MLA "93.3% KV-Cache reduction" (8, 79) | **UNSUPPORTED (contradicted)** | Repo’s own `test_mla` reports 20.8% via `get_kv_cache_size` |
| MTP "85–90% acceptance / +80% throughput" (15–16) | **UNSUPPORTED** | No verification logic exists (C10); decode path crashes (C8/C9); no benchmarks anywhere |
| PagedAttention "+40%" (27) | **UNSUPPORTED** | Block manager unused by model/engine (M10/M25); cache dtype crashes (C11) |
| FP8 "−50% cost" (34) | **UNSUPPORTED** | `_quantize_to_fp8` is identity (M15); no benchmarks |
| GRPO "对标 o1" (39) | **UNSUPPORTED** | GRPO cannot run a single backward (C5) |
| DualPipe "95%+ GPU util / ~5% bubble" (44) | **UNSUPPORTED** | No schedule/distributed code exists (M14); backward broken (C4) |
| NaViT "arbitrary resolution / no resize distortion" (49–50) | **UNSUPPORTED** | >448px width collides or crashes (M31) |
| NaViT "sequence packing for batching" (51) | **PARTIAL** | Padding, not packing (M36); CPU-only buffers (M33) |
| Streaming audio "chunk-based causal processing" (53–54) | **UNSUPPORTED** | Forward always crashes (C12/C13); no cross-chunk state (M30) |
| "Real-time latency <200ms" (55) | **UNSUPPORTED** | Chunk math makes a "500ms chunk" 80s (M29); no latency code |
| AWQ "4-bit activation-aware" (59) | **UNSUPPORTED** | Random weights; original weights never read; no calibration (C14) |
| GPTQ "accurate post-training quantization" (60) | **UNSUPPORTED** | Stub returning fresh random weights per call (C14) |
| FP8 quantization (61) | **UNSUPPORTED** | Silently no-ops (`METHODS["fp8"]=None`) |
| "vLLM engine integration" (65) | **PARTIAL** | Custom vLLM-*style* class; no `import vllm` |
| "GPU monitoring with Prometheus" (66) | **UNSUPPORTED** | Python lists only; no prometheus_client; `src/monitoring/` empty |
| "HPA autoscaling" (67) | **PARTIAL** | Heuristic only; unit-mismatched (M27); no K8s manifests; `src/deployment/` empty |
| Quick Start `python -m helioslm_v5.tests.test_v5` (72) | **PARTIAL** | Runs but OOMs at `test_v5_model` and always exits 0 (M34) |
| Architecture existence (MLA/MTP/MoE/GRPO/FP8/DualPipe files) | **PARTIAL** | Files exist; core math deviates or is broken as detailed above |

---

## 6. Prioritized Fix Roadmap

**P0 — Make it run end-to-end (blocks everything):**
1. Fix MTP unpack contract + decode-path shapes (C8, C9) or disable `use_mtp` by default.
2. Fix MLA cached-decode masking + RoPE position threading (C1, C2) — generation is currently wrong token-by-token.
3. Fix audio encoder state rank + missing `F` import (C12, C13).
4. Make the test runner fail loudly (`sys.exit(1)`), use a genuinely tiny config for tests, honor `size="lite"` (M34).
5. Remove or quarantine GPTQ/AWQ stubs so `quantize_model` can't silently destroy a model (C14).

**P1 — Make training learn:**
6. DualPipe: restore autograd connectivity (C4) via recompute or live-graph backward; zero grads in FP8 trainer (C7); real AdamW over master weights (M17).
7. GRPO: real sampler + new-policy log-prob forward (C5), answer alignment (C6), old-policy ratio (M12), k3 KL estimator (M13).

**P2 — Match claimed techniques:**
8. MLA: separate V projection without RoPE (C3), actual weight absorption or revised claims (M1), delete dead compute (M2), lazy RoPE cache (M3), head_dim validation (M4).
9. MoE: selection-only bias + aux-free update rule (M6), fix distributed path (M7), sorted-token dispatch (M8).
10. PagedAttention/engine: write-at-`context_len` convention + allocation on write (M19/M20), refcounts + true CoW (M21), wire BlockManager into model/engine (M10/M25), define a `Request` class (M26).
11. MTP: causal mask (M22), hidden-state chaining + per-module depth (M23), shared embedding/head + MTP training loss (M24), real verification/rollback (C10).
12. NaViT: row/col position embeddings sized to true max grid (M31); audio: correct chunk math + conv-state semantics + cross-chunk attention (M28–M30).

**P3 — Honesty & hardening:**
13. README: downgrade or evidence every quantitative claim; add a benchmarks harness before publishing numbers.
14. Batch>1 generation (M9), padding masks (M11), temperature guard, pre-attention norm (M5), device/dtype hygiene across quant/vision paths.

---

## Appendix — Verification Log

Independent spot-checks by orchestrator (torch 2.8, all confirmed reviewers' claims):
- SDPA `is_causal` with `q_len=1, kv_len=5` → output equals `v[...,0,:]` exactly (C1) ✅
- Unpacking model's 2-tuple return into 4 names → `ValueError: not enough values to unpack` (C8) ✅
- GRPO `answers * group_size` → interleaved ordering, confirmed misaligned (C6) ✅

Reviewer-executed reproductions: audio forward `RuntimeError` (C12), `NameError: F` (C13), DualPipe `weight.grad is None` (C4), GRPO `loss.backward()` `RuntimeError` (C5), GPTQ non-determinism (C14), AWQ odd-dim crash (M32), NaViT 512×512 `IndexError` + 224×1024 id collisions (M31), fp32 paged-attention dtype crash (C11), MoE autograd-clearing check (cleared), full test-suite run (3 FAIL + 2 OOM-kill, exit 0) (M34).
