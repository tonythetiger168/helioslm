# HeliosLM Local Deployment Memory Baselines

Status: **planning targets, not released checkpoints or measured compatibility**.

Six device-memory tiers: **16, 32, 64, 128, 256, 512 GB**. Each tier refers to **usable accelerator VRAM or unified memory**, not system RAM. For multi-GPU systems, effective usable capacity is lower than the sum of advertised VRAM and depends on sharding, interconnect and runtime.

| Tier | Candidate model parameters | Deployment positioning |
| --- | --- | --- |
| 16 GB | 7–8B | Personal assistant, lightweight inference |
| 32 GB | 14–16B | Coding assistant, advanced local inference |
| 64 GB | 30–32B | Coding and agent workflows |
| 128 GB | 70–72B | Complex reasoning, multi-agent |
| 256 GB | 120–150B | Research and enterprise inference |
| 512 GB | 250–300B | Flagship experimental inference |

These are **aspirational size classes**, generally assuming weight-only 4-bit quantization, low batch size, limited context, and sufficient headroom. They are **not guaranteed fits**: quantization metadata, embeddings, runtime buffers, activations, KV cache, fragmentation and longer context can exceed available memory. Full-precision weights or training require much more memory. MoE models must report *total* and *active* parameter counts separately; total parameters largely determine weight residency.

## Acceptance gates for each tier

1. Publish exact model/config, checkpoint provenance, license, parameter counts (total and active), tokenizer and quantization format.
2. Test on named hardware with available memory, accelerator driver, PyTorch/runtime versions, batch size and context length.
3. Record model-load peak memory, prefill peak memory, decode peak memory, OOM behavior, time-to-first-token, prefill throughput and decode tokens/sec.
4. Validate accuracy and safety against a reference precision, including quantization regressions and decision-head calibration.
5. Test cold load, sustained generation, concurrent requests, cache growth, and deterministic failure/recovery.
6. Publish a reproducible JSON benchmark artifact and command line for each verified device/config pair. Label untested combinations **unverified**.

## Capacity planning

Approximate weight bytes = parameter count × effective bits per weight / 8, plus scale/zero-point metadata and unquantized tensors. Reserve additional memory for KV cache, kernels, framework and fragmentation; do not claim fit based on weights alone. Provide configurable context/batch budgets and enforce admission control on constrained tiers.

## Release policy

Do not advertise a tier as supported until the full acceptance gates pass on real hardware. This roadmap does not assert that 7B–300B HeliosLM weights have been trained or published. CUDA Issue #1 is owned by Claude and is out of scope for this documentation change.
