# Launch posts — r/LocalLLaMA and r/MachineLearning variants

Updated for v5.14 (2026-09-16): GGUF export, benchmark suite, fused quant
kernels, batched prefill, CPU-trained toy checkpoint, multi-process
DualPipe. 54 unit + 9 integration tests, all five known limitations
resolved.

## Variant A — r/LocalLLaMA

**Title options:**
- `[Project] HeliosLM — DeepSeek-V3-style LLM stack in pure PyTorch, GGUF export included, runs on CPU`
- `I rebuilt the DeepSeek-V3 stack in pure PyTorch on my laptop — MLA, hybrid linear attention, MTP (acceptance 1.00), GGUF export, 99.2% KV-cache savings`

**Body:**

Hey r/LocalLLaMA — I've been rebuilding the DeepSeek-V3/K3-class stack from
scratch in pure PyTorch, designed so every component is inspectable,
verifiable on CPU, and hackable. Two days ago it still had five known
limitations; as of v5.14, all five are resolved. Highlights:

**Inference / serving**
- **MLA with weight absorption** — latent-only KV cache; the full config
  holds **99.2% less KV-cache memory than MHA at 128k context** (2.0 GB vs
  257.7 GB, analytically verified, chart in the repo)
- **Hybrid linear attention** — Gated Delta Rule layers interleaved with
  MLA, fixed ~1 MB recurrent state per layer regardless of context; decode
  bit-equivalent to one-shot (<2e-7)
- **vLLM-style engine** — paged KV, copy-on-write forks, watermark batching,
  and as of v5.12 **batched equal-length prefill** (exact, mask-free — the
  only batched prefill path available to recurrent-state models)
- **MTP speculative decoding** with strict (p−q)₊ verification — and a
  shipped CPU-trained toy checkpoint where greedy decoding hits **100%
  MTP acceptance** (11/11)

**Training**
- FP8 trainer (native float8 + STE), GRPO with k3 KL, Muon (per-head
  Newton–Schulz), QAT
- **DualPipe** — gradient-exact schedule simulation, and as of v5.14 it runs
  **one stage per OS process** (phased queue protocol, gradients match the
  single-process scheduler <1e-5)

**Quantization & export**
- True GPTQ (Hessian OBS + error compensation + act-order), AWQ, MXFP4 —
  all with **fused group-wise dequant×matmul kernels** since v5.11 (dense
  weight never materialized; the fused rewrite also exposed and fixed a
  latent MXFP4 decode bug)
- **GGUF v3 export/import** — spec-faithful writer + reader, 83/83 bit-exact
  round-trip, cross-checked against the official gguf reader

**Verification culture**
54 unit tests + 9 integration tests; key paths (MLA absorption, hybrid
decode, DualPipe grads, GGUF round-trip) are checked bitwise or at tight
tolerance, not "loss went down". A CPU benchmark suite and a
log-likelihood eval harness (`python -m helioslm_v5.eval.harness`) are
included.

Honest positioning: correctness-focused reference implementation, not a
throughput competitor to llama.cpp/vLLM. The toy checkpoint is char-level
and clearly a toy (trained on the repo's own source, ~10 CPU-minutes) —
it's there to make `generate()`, MTP, and the harness runnable out of the
box, not to be useful.

Repo: https://github.com/tonythetiger168/helioslm

Questions for the community:
1. For GGUF: would you rather see a llama.cpp-compatible MLA layout
  mapping, or keep the spec-faithful container and let converters handle
  layout?
2. The hybrid linear-attention state is bf16 [heads, 128, 128] per layer —
  is there appetite for FP8 recurrent states in this community?
3. What's the most useful next checkpoint: bigger toy, or a real tokenizer
  + small BPE vocab?

---

## Variant B — r/MachineLearning

**Title:** [P] HeliosLM — from-scratch, CPU-testable DeepSeek-V3/K3-style stack; v5.14 adds multi-process DualPipe + fused quant kernels + CPU-trained checkpoint

**Body:**

Hi r/MachineLearning — I maintain a pure-PyTorch reference implementation
of the modern LLM stack where the emphasis is *verified correctness you can
read*. Recent two-day push (v5.10 → v5.14) closed all five known
limitations:

- **Multi-process DualPipe** (v5.14): one `DualPipeStage` per spawn'd
  process; phased fwd/bwd queue protocol with sentinel propagation;
  recompute-based backward with RNG capture. Outputs, input grads, and
  per-stage parameter grads match the single-process scheduler <1e-5
  (2 ranks × 3 micro-batches test).
- **Fused group-wise dequant×matmul** (v5.11) for AWQ/GPTQ/MXFP4 — the
  dense [out, in] weight is never materialized; GPTQ slices by maximal
  runs of constant `g_idx` (act-order safe). The fused rewrite exposed a
  latent reference bug: the MXFP1 magnitude table was built on a long
  tensor, silently truncating (0.5, 1.5) — decode is now the exact inverse
  of encode.
- **Batched equal-length prefill** (v5.12) in the vLLM-style engine:
  exact (no mask, no position shift), and the only batched prefill path
  for recurrent-state (Gated Delta Rule) models.
- **CPU-trained toy checkpoint** (v5.13): char-level, CE + 0.3·MTP aux
  loss; greedy MTP acceptance 1.00 — makes `generate()` and the
  log-likelihood harness runnable against trained weights.
- **Benchmark suite** (v5.10): analytic KV-cache accounting — full config
  2.0 GB vs 257.7 GB MHA at 128k — plus wall-clock CPU generation numbers,
  reproducible via `python benchmarks/bench_cpu.py`.

Older core (all tolerance/bitwise-verified): MLA weight absorption,
hybrid Gated-Delta attention with doc-boundary packed-sequence training,
auxiliary-loss-free MoE balancing, strict MTP verification, FP8 training
with E5M2 gradient hooks, DualPipe gradient-exactness vs naive schedule,
NaViT + streaming audio encoders.

Repo: https://github.com/tonythetiger168/helioslm

Would especially appreciate critical eyes on:
1. The phased-queue multiprocess protocol: the sentinel-propagation
   design trades a third control phase for determinism — better patterns?
2. GPTQ fused slicing by g_idx runs vs. per-column gather: any accuracy
   or performance concerns I'm missing?

---

## Variant C — Hacker News (Show HN)

> **Show HN: HeliosLM – DeepSeek-V3-style LLM stack in pure PyTorch, all 5 known limitations closed**
>
> Pure-PyTorch reference implementation of the modern LLM stack, built to be
> read and verified on a CPU: MLA with weight absorption (99.2% KV-cache
> savings vs MHA at 128k), hybrid Gated-Delta linear attention, strict MTP
> speculative decoding, FP8 training, GRPO, DualPipe (now multi-process,
> one stage per OS process), fused AWQ/GPTQ/MXFP4 kernels, GGUF v3
> export/import (83/83 bit-exact round-trip), vLLM-style engine with
> batched prefill, a CPU-trained toy checkpoint (greedy MTP acceptance
> 1.00), benchmark suite, and a log-likelihood eval harness.
>
> 54 unit + 9 integration tests; key paths checked bitwise or at tight
> tolerance.
>
> Feedback wanted:
> 1. For a correctness-first repo, what's the most valuable next
>    component: CUDA end-to-end verification, or a real tokenizer?
> 2. Any interest in a writeup of the multiprocess pipeline protocol?
>
> GitHub: https://github.com/tonythetiger168/helioslm

---

### Posting tips
- Post Tuesday–Thursday, 8–10am US Eastern; reply to every comment in the
  first 3–4 hours
- Lead with the demo GIF in the README; the 99.2% KV-cache chart is the
  strongest single image for ML audiences
- The GGUF export + fused kernels are the strongest hooks for
  r/LocalLLaMA; the multiprocess DualPipe protocol + MXFP4 bug story are
  the strongest hooks for HN
- Don't ask for stars; the community questions above invite technical
  discussion, which is what these communities reward
