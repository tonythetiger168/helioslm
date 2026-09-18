# Launch posts — r/LocalLLaMA / r/MachineLearning / HN variants

Updated for **v5.18** (2026-09-18): roadmap #6 ("a verifiable colibri") is
feature-complete — disk-tier expert streaming, speculation break-even
telemetry, engine-agnostic stream scorer, content-addressed prefix pool.
58 unit + 9 integration tests, all five known limitations resolved.

## Variant A — r/LocalLLaMA

**Title options:**
- `[Project] HeliosLM — the audit layer for consumer-scale LLM serving: bit-exact expert streaming, prefix pool, MTP break-even data`
- `I built the verifiable complement to colibri: DeepSeek-V3-style stack in pure PyTorch where every serving trick has a bit-exactness oracle`

**Body:**

Hey r/LocalLLaMA — colibri made frontier MoE serving on consumer hardware
credible by streaming experts from disk. What's been missing on the audit
side: when you tier weights, prune experts, or toggle speculative decoding,
**how do you know the answer didn't change?** HeliosLM v5.18 is my answer:
a pure-PyTorch DeepSeek-V3/K3-style stack where every serving optimization
ships with a verification oracle.

**The audit stack (all landed today, roadmap #6 complete):**
- **Disk-tier expert store** (v5.15): routed experts offloaded to a
  memory-mapped file, LRU residency with hit/miss/eviction telemetry.
  Streaming forward is **bitwise-identical** to the dense forward — even
  under eviction churn (oracle-asserted, incl. detach/reattach cycles)
- **Speculation break-even instrumentation** (v5.16): sweep emits the exact
  JSONL schema proposed to colibri. First honest data point on the toy
  checkpoint: MTP drafting is **−0.3% net cold, +5.4% net warm** at 72%
  acceptance — cache state decides whether drafting pays
- **Engine-agnostic stream scorer** (v5.17): score (prompt, output) JSONL
  from *any* engine — colibri, vLLM, llama.cpp — under a reference model;
  A/B compare with bootstrap CI. Built for exactly the gs64-vs-per-row
  container question: quality deltas belong in tables, not anecdotes
- **Content-addressed prefix pool** (v5.18): cross-session prefix reuse,
  keyed by blake2b(token block + config fingerprint — quant scheme included,
  so a stale/different container can never be served). Pooled greedy ==
  from-scratch greedy, bitwise, for hits/misses/evictions alike

**The stack underneath** (all CPU-verified, 58 tests):
MLA with weight absorption (−99.2% KV-cache vs MHA at 128k, analytic),
hybrid Gated-Delta linear attention (bit-equivalent decode), sigmoid MoE
with auxiliary-loss-free balancing, strict MTP verification, vLLM-style
paged engine (batched equal-length prefill — exact, mask-free), FP8
training + GRPO + Muon, **multi-process DualPipe** (one stage per process,
gradients match single-process <1e-5), fused AWQ/GPTQ/MXFP4 kernels
(exposed and fixed a latent MXFP4 decode bug), GGUF v3 export/import
(83/83 bit-exact round-trip), CPU benchmark suite, loglikelihood harness,
and a char-level toy checkpoint with a trained MTP head (greedy acceptance
1.00).

Honest positioning: correctness-first reference implementation, not a
throughput competitor. The toy checkpoint is a toy — it's there so
generate()/MTP/harness run against trained weights out of the box.

Repo: https://github.com/tonythetiger168/helioslm
Demo (after HF upload): https://huggingface.co/spaces/tonythetiger168/helioslm-demo

Questions for the community:
1. Would you wire the stream scorer into your eval pipeline? What record
  format does your engine already emit (prompt/output JSONL)?
2. For the prefix pool: is block-level (16–64 tokens) the right reuse
  granularity for agentic multi-turn workloads, or do you want
  session-scoped pooling?
3. What's the most useful next oracle: FP8 expert streaming, or act-order
  GPTQ under eviction?

---

## Variant B — r/MachineLearning

**Title:** [P] HeliosLM v5.18 — "verifiable serving": every inference optimization ships with a bit-exactness oracle (streaming experts, prefix pooling, MTP break-even)

**Body:**

HeliosLM is a pure-PyTorch DeepSeek-V3/K3-style reference stack with a
verification culture (bitwise/tolerance oracles on MLA, hybrid decode,
DualPipe gradients, GGUF round-trips; 58 unit + 9 integration tests).
Today's v5.15–v5.18 push completes roadmap #6 — the audit layer for
weight-tiered serving:

- **Disk-tier expert store**: mmap + LRU residency; streaming forward is
  bitwise-equal to dense under eviction. Two implementation bugs found by
  the oracle itself: LRU eviction during detach leaving empty params, and
  full-prefix past snapshots double-counting future positions (fixed via
  per-block incremental prefill).
- **Break-even telemetry**: first measured point — MTP drafting −0.3% cold
  / +5.4% warm at 72% acceptance on an 8.5M toy model; emitted in the
  colibri-P3 JSONL schema for cross-engine comparability.
- **Stream scorer + A/B bootstrap CI**: quality-gates any engine's token
  stream (greedy NLL 0.38 vs random 7.26 on the toy model — the
  discrimination direction is correct).
- **Prefix pool**: content-hash keyed KV snapshots with config fingerprint
  guarding; four oracle scenarios (full/partial hit, fingerprint miss,
  post-eviction) all bitwise-equal to from-scratch.

Repo: https://github.com/tonythetiger168/helioslm

Critical questions:
1. Per-block incremental prefill snapshots cost O(L²/B) copies — is there
  a smarter immutable-past structure for prefix pools?
2. For break-even tables: is a toy-model data point useful methodology
  evidence, or only noise until measured at frontier scale?

---

## Variant C — Hacker News (Show HN)

> **Show HN: HeliosLM v5.18 – every LLM serving optimization ships with a bit-exactness oracle**
>
> Pure-PyTorch DeepSeek-V3-style stack. When you stream experts from disk,
> pool KV prefixes across sessions, or toggle speculative decoding, how do
> you know the answer didn't change? Here each optimization has an oracle:
> streaming forward == dense forward bitwise (even under LRU eviction),
> pooled greedy == from-scratch greedy bitwise, MTP draft net-gain measured
> per cache state (−0.3% cold / +5.4% warm), and a standalone scorer
> quality-gates any engine's output with A/B bootstrap CIs.
>
> 58 unit + 9 integration tests; multi-process DualPipe; fused AWQ/GPTQ/
> MXFP4 kernels (found a latent MXFP4 decode bug); GGUF v3 round-trip
> bit-exact; CPU benchmark suite; CPU-trained toy checkpoint with a trained
> MTP head.
>
> Feedback wanted:
> 1. Which serving optimization most needs an external audit layer today?
> 2. Writeup interest: the three bugs the oracles caught (eviction-during-
>    detach, future-position past snapshots, long-dtype magnitude table)?
>
> GitHub: https://github.com/tonythetiger168/helioslm

---

### Posting tips
- Best window: Tue–Thu 08:00–10:00 US Eastern (20:00–22:00 HKT); Friday
  nights bury threads — hold for next week if missed
- Reply to every comment in the first 3–4 hours; the three questions above
  are chosen to surface technical discussion, not star-begging
- Lead images: the 99.2% KV-cache chart (ML crowd), the demo GIF (general)
- The "three bugs the oracles caught" hook is the strongest HN angle —
  engineers love post-mortems more than features
