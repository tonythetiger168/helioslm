# Launch posts — r/LocalLLaMA and r/MachineLearning variants

## Variant A — r/LocalLLaMA

**Title options:**
- `[Project] HeliosLM — DeepSeek-V3-style LLM stack in pure PyTorch, fully runnable on CPU`
- `I rebuilt the DeepSeek-V3 stack in pure PyTorch so it runs on a laptop — MLA, MoE, hybrid linear attention, MTP decoding, FP8 training, vLLM-style engine`

**Body:**

Hey r/LocalLLaMA — I've spent the last months rebuilding the DeepSeek-V3/K3-class LLM stack from scratch in pure PyTorch, designed so every component is inspectable and testable on a CPU before you ever touch a GPU.

What's inside:

- **MLA with weight absorption** — latent-only KV cache, −97.7% memory vs MHA on the full config, verified mathematically equivalent (<1e-4) to the expanded path
- **Hybrid linear attention** — Gated Delta Rule layers interleaved with MLA, fixed-size recurrent state cache instead of growing KV. Decode is bit-equivalent to one-shot (<2e-7), and it supports packed sequences via doc-boundary state reset
- **Sigmoid-gated MoE** with auxiliary-loss-free load balancing (heuristic or quantile bias updates), plus a latent-space MoE variant
- **DeepSeek-style MTP speculative decoding** with *strict* verification — residual (p−q)₊ resampling, not the approximate shortcut
- **vLLM-style serving engine** — paged KV accounting, copy-on-write forks, watermark-aligned continuous batching
- **Training stack** — FP8 trainer (native float8 + STE), DualPipe schedule simulation (gradient-exact), GRPO, Muon optimizer
- **Quantization** — true GPTQ (Hessian OBS + error compensation, act-order), AWQ, FP8, MXFP4
- **Multimodal** — NaViT-style vision encoder + streaming causal audio encoder

Everything is covered by 44 unit tests + 9 integration tests, and the repo has been through four rounds of adversarial code review. Key paths (MLA absorption, hybrid decode, DualPipe gradients) are checked with bitwise-equivalence tests, not just "loss went down."

Repo: https://github.com/tonythetiger168/helioslm

Honest limitations: no pre-trained weights yet (you train from scratch or plug in your own), CPU-verified (CUDA paths static-checked), and it's a correctness-focused reference implementation — not a throughput competitor to llama.cpp/vLLM.

Questions for the community:
1. For hybrid linear-attention models, what matters more to you — the fixed-size state cache at long context, or throughput on short prompts?
2. Would a tiny pre-trained checkpoint (few hours of training, clearly a toy) be useful for kicking the tires, or is train-from-scratch fine?
3. Any interest in a GGUF/llama.cpp-compatible export path for the MLA variant?

Happy to answer technical questions — the code is meant to be read.

---

## Variant B — r/MachineLearning

**Title:** [P] HeliosLM — a from-scratch, CPU-testable reference implementation of a DeepSeek-V3/K3-style stack (MLA, hybrid linear attention, MoE, DualPipe, MTP decoding)

**Body:**

Hi r/MachineLearning — I wanted a codebase where I could actually *read and verify* the mechanisms from recent frontier architectures instead of trusting black-box implementations, so I rebuilt the stack in pure PyTorch. Everything below has correctness checks (bitwise-equivalence or tight tolerances), and the whole thing runs on CPU.

Technical highlights:

- **MLA weight absorption** with an exactness check against the expanded attention path; latent-only KV cache at −97.7% memory vs MHA
- **Hybrid attention**: Gated Delta Rule linear layers interleaved with MLA; recurrent-state decode verified ≡ one-shot (<2e-7); packed-sequence support via doc-boundary state reset
- **Auxiliary-loss-free MoE balancing** via selection-only bias updates (heuristic and quantile variants); sigmoid gating; latent-space routed experts
- **DualPipe schedule simulation**, recompute-based and gradient-exact (bitwise-verified against the naive schedule), single-process
- **Strict MTP speculative decoding**: residual (p−q)₊ resampling with O(1) rollback, batched, including hybrid-state restore+replay
- **Training**: FP8 (E5M2 grad hooks + STE + master weights), GRPO with k3 KL, Muon (Newton–Schulz, optional per-head blocks), QAT straight-through fake-quant
- **Quantization**: Hessian-OBS GPTQ with error compensation and act-order, AWQ grid search, MXFP4

44 unit tests + 9 integration tests; four rounds of adversarial review are documented in `docs/`.

Repo: https://github.com/tonythetiger168/helioslm

Would especially appreciate critical eyes on:
1. The hybrid linear-attention doc-boundary state reset — is the packing semantics what you'd expect for training on variable-length documents?
2. The quantile variant of the loss-free balancing bias — convergence behavior in your experience vs. the heuristic update?

---

## Variant C — Hacker News (Show HN)

> **Show HN: HeliosLM – DeepSeek-V3-style LLM stack in pure PyTorch, runs on CPU**
>
> I built HeliosLM as a reference implementation of the modern LLM stack after finding that most open-source repos assume you already have 8 GPUs. Everything — MLA with weight absorption (−97.7% KV cache memory vs MHA, verified equivalent), sigmoid-gated MoE with auxiliary-loss-free balancing, hybrid Gated-Delta linear attention, DeepSeek-style MTP speculative decoding, DualPipe scheduling, FP8 training, GPTQ/AWQ/MXFP4 quantization, and a vLLM-style paged serving engine — is plain PyTorch and unit-testable on a laptop.
>
> The code has been through four rounds of adversarial review; key paths are checked with bitwise-equivalence tests rather than "looks right."
>
> Would love feedback on two things:
> 1. Which part is most interesting to dig into — the hybrid linear attention or the DualPipe simulation?
> 2. What's the biggest gap for you as a potential user/contributor?
>
> GitHub: https://github.com/tonythetiger168/helioslm

---

### Posting tips
- Post Tuesday–Thursday, 8–10am US Eastern for HN; similar window for Reddit
- Reply to every comment in the first 3–4 hours — early engagement decides the thread's fate
- Don't ask for stars; the questions above invite technical discussion, which is what these communities reward
- Record the demo GIF *before* posting — you can edit it into the README the same day
