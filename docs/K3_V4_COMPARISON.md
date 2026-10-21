# Kimi K3 vs DeepSeek-V4 in HeliosLM — what is implemented and what it measured

- **Date:** 2026-10-21
- **Roadmap:** Phase B completion criterion ("publish a K3 / V4 comparison
  document"), [`ROADMAP_2026Q4.md`](ROADMAP_2026Q4.md)
- **Verification tier:** the numbers below are copied from committed,
  hash-checked artifacts (tier `measured`, see
  `benchmarks/verification_tiers.json`); every row names its artifact and
  test. All results are toy-scale CPU probes with a single seed. They say
  which mechanism does what on a controlled task, not how K3 or V4
  perform.
- **Sources:** Kimi K3 Technical Report (arXiv 2607.24653), DeepSeek-V4
  Report (arXiv 2606.19348), torchtitan's K3 Quantile Balancing PR
  (#4412 / #4577). The reports were not reachable from the development
  containers; details were checked against search excerpts and the
  torchtitan code, and every deviation is listed below.

## 1. The two designs in one table

| | Kimi K3 | DeepSeek-V4 |
|---|---|---|
| Long-context idea | Mostly linear attention: 3 KDA layers per 1 Gated MLA layer | Keep softmax attention, shrink what it reads: compress the KV cache, then attend sparsely |
| Linear-attention layer | KDA: gated delta rule with per-channel decay | — |
| Full-attention layer | Gated MLA (sigmoid output gate) | MLA with Compressed Sparse Attention (CSA) and heavily compressed HCA layers |
| Residual stream | Attention Residuals | standard |
| MoE | Stable LatentMoE: routing in a latent space, Quantile Balancing, RMSNorm, SiTU-GLU | DeepSeekMoE (V3 lineage: aux-loss-free bias) |
| HeliosLM preset | `k3_block_config()` (v5.57) | `CompressedSparseAttention`, `v4_attention_stack()` (v5.58) |

## 2. What HeliosLM implements

### Kimi K3 (`configs.config_v5.k3_block_config`)

| K3 component | HeliosLM | Since | Test |
|---|---|---|---|
| KDA per-channel decay | `GatedDeltaAttention` per-channel decay gate | v5.21 | `case_hybrid` |
| 3:1 KDA:MLA in K3 order (KDA, KDA, KDA, MLA) | `hybrid_attention.first_layer_full=False` | v5.57 | T91 |
| Gated MLA | `attention.output_gate` (`MLA.o_gate_proj`), both expanded and absorbed paths | v5.57 | T91 |
| Attention Residuals | attention-residual accumulator | v5.5 | `case_model_mtp` |
| LatentMoE | `moe.latent_dim` (router and routed experts in latent space; shared experts full width) | v5.5 | `case_moe` |
| Quantile Balancing | `moe.balance_strategy="quantile_balancing"` | v5.77 | T116 |
| SiTU-GLU | `moe.activation="situ"` — **simplified**: tanh soft-cap GLU plus RMSNorm | v5.5 | `case_moe` |

Not reproduced: K3's exact SiTU-GLU and RMSNorm placement inside Stable
LatentMoE, FP4 training, and scale itself (per-head Muon exists, but no K3-scale
training run).

### DeepSeek-V4 (`attention.kv_compress`)

| V4 component | HeliosLM | Since | Test |
|---|---|---|---|
| CSA: compress, then select | `CompressedSparseAttention`: one entry per completed chunk of `ratio` tokens, lightning-style indexer picks top-`topk` past chunks, exact attention inside the query's own chunk; equals dense at `ratio=1` | v5.58 | T92 |
| HCA (heavily compressed layers) | HCA 128× compression | B1-2 | `test_hca_*` |
| IndexShare | `shared_indexer` across a group of CSA layers | B1-1 / v5.58 | T92 |
| Learned compressor | `compressor="learned"`: per-head softmax pooling, `MLP(x_j)` + per-position bias, zero init = mean pooling | v5.59 | T93–T95, T117 |

Recorded deviations: an MLP softmax pooler of our own design (not V4's
compressor), an fp32 linear indexer (V4: FP4), head-shared selection, and
a standalone module rather than V4's MLA-latent integration.

## 3. What the probes measured

### 3.1 V4's CSA: the compressor is the bottleneck, not the selection

Passkey probe (`examples/csa_ab_passkey.py`, `csa_learned_compressor.py`;
evaluation stream, L=512 / L=1024; dense attention 0.910 / 0.875):

| CSA compressor | 4× | 16× | Artifact |
|---|---|---|---|
| mean pooling | 0.130 / 0.105 | 0.150 / 0.125 | `csa_ab_2026-10-15` |
| learned, round 1 | 0.665 / 0.650 | 0.150 / 0.125 (chance) | `csa_learned_2026-10-15` |
| learned, round 2 (hidden 64) | 0.725 / 0.645 | 0.375 / 0.390 | `csa_learned_h64_2026-10-15` |
| learned, round 3 (9600 steps) | **0.825 / 0.800** | — | `csa_learned_r3_2026-10-15` |
| 4× → 16× curriculum | — | **0.790 / 0.740** | `csa_learned_16x_2026-10-21` |
| oracle ceiling (perfect pooling) | 0.900 / 0.870 | 0.860 / 0.845 | same artifacts |

- With mean pooling the indexer still picks the right chunk in 86–100% of
  trials; pooling divides the planted token's score lead by the ratio.
- Selecting every chunk gives the same accuracy as top-8, so selection is
  not what limits recall.
- A learned compressor closes most of the gap: about 92% of the ceiling
  at 4× and 92% / 88% at 16× (the 16× compressor needed a curriculum from
  the 4× weights; trained from scratch it plateaus at 0.39).

### 3.2 Repository-scale look-up (AC4): hybrid vs CSA vs dense

`benchmarks/repo_longctx_2026-10-19.json` (T113). A definition must be
found among 8 / 16 / 32 files; trained at 8.

| arm | 8 files | 16 | 32 |
|---|---|---|---|
| dense | 0.29 | 0.20 | 0.12 |
| dense, 2× steps | 0.91 | 0.62 | 0.38 |
| all sliding window (negative control) | 0.27 | 0.17 | 0.11 |
| sliding / global hybrid | **1.00** | **1.00** | **1.00** |
| CSA, mean compressor | 0.54 | 0.38 | 0.28 |
| CSA, learned compressor | 0.20 | 0.16 | 0.08 |

The hybrid (local layers plus periodic global ones, the family K3's 3:1
interleave and MiMo's 5:1 belong to) is the only arm that generalises to
4× the training size. CSA's learned compressor, trained end to end with
the model here rather than on the readout as in 3.1, does worse than mean
pooling: the compressor is hard to learn jointly at this budget.

### 3.3 K3's Stable LatentMoE: Quantile Balancing

`benchmarks/quantile_balancing_ab_2026-10-21.json` (T116). One LatentMoE
layer, skewed token stream; MaxVio = largest excess expert load.

| rule | first 20 updates | steady state | after a drift | bias drift |
|---|---|---|---|---|
| sign (DeepSeek-V3 aux-loss-free) | 0.412 | 0.298 | 0.439 | 0.014 |
| SMEBU (Arcee Trinity) | 0.451 | **0.261** | 0.496 | 0.001 |
| Quantile Balancing (K3) | **0.370** | 0.313 | **0.375** | **0.000** |

K3's rule adapts fastest and never drifts; a momentum rule is quieter in
steady state.

### 3.4 Structural checks on the K3 block

T91: defaults bit-identical when the gate is off; the gate is bitwise
equal to its ungated twin in both MLA paths; expanded == absorbed with the
gate on (1.5e-7); the preset's cached decode equals the full forward
(1.6e-6), with gradients reaching the output and residual gates.

## 4. Takeaways

1. **Two answers to the same KV-cost problem.** K3 replaces most attention
   layers with a fixed-size recurrent state; V4 keeps attention and reads
   less of it. On these probes the interleaved design is the robust one:
   exact long-range recall survives in the global layers, and nothing has
   to be learned to compress.
2. **V4's quality hinges on the compressor.** Selection is cheap and
   accurate; pooling is where information is lost. A learned compressor
   gets within 8–12% of perfect pooling, but only with care (a curriculum
   at 16×), and joint training at small scale did not get there (3.2).
3. **K3's MoE stability pieces are independently useful.** Quantile
   Balancing is a drop-in bias rule that removes a step-size
   hyperparameter and recovers fastest from distribution shifts.
4. **What these probes cannot say:** anything about scale, wall-clock
   speed on GPUs (CPU only), or quality on natural text. Phase C (~1B GPU
   run) is where that would be tested.
