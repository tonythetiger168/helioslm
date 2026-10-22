# MiniMax Sparse Attention (MSA): reading notes for HeliosLM

OS8 from the 2026-10-22 open-model review. Source: *MiniMax Sparse
Attention*, arXiv:2606.13392v2 (read 2026-10-22), the attention behind
MiniMax-M3. The kernel is at github.com/MiniMax-AI/MSA. Numbers below are the
paper's own; none were reproduced here.

## What MSA is

- **Backbone:** plain GQA. Blocks are selected over the real, uncompressed
  KV cache. DSA (DeepSeek V3.2) instead sits on MLA in MQA mode.
- **Index branch:** one index query head per GQA group and one index key
  head shared by all groups. Token scores are max-pooled within each KV
  block, and blocks that are not causally visible get -inf.
- **Selection:** top-k blocks **per GQA group**, shared by that group's
  query heads. M3 uses block size 128 and k = 16, so each query sees 2,048
  tokens. The query's own (incomplete) block is always selected. Forced
  sink and wide local window were tried and dropped (Table 5: no quality
  difference), as was a learnable sink logit (B.5).
- **Main branch:** exact attention over the selected blocks only.
- **Kernel:**
  - top-k runs on raw scores ("exp-free"; softmax preserves order) with a
    per-thread register heap, 3.7-5.1x faster than TileLang or torch at
    128K;
  - attention runs "KV outer, gather Q": blocks in the outer loop, the
    queries that picked each block gathered inside it.
- **M3 layout:**
  - 60 layers, 64 query heads, 4 KV heads, head dim 128;
  - dense attention in layers 1-3 and MSA from layer 4 on;
  - 4 index heads of dim 128.

## Training recipe

- **From scratch (MSA-PT):** 40B tokens of indexer warm-up under full
  attention, then sparse for the rest of 3T tokens.
- **Conversion (MSA-CPT):** a 2.6T-token GQA checkpoint, then 40B tokens of
  warm-up and 400B tokens of sparse continued pre-training.
- **Indexer loss:** KL from the stop-gradient, head-averaged main-branch
  distribution over the selected tokens into the index distribution, added
  as LM loss + lambda * sum over layers.
- **Gradient detached at the indexer input (B.3):** letting the KL reach the
  backbone caused gradient spikes and short-context regression.
- **Loss ablation (B.2, C.3):** LM loss alone hurt long-context retrieval,
  KL alone hurt short-context ability, and both together worked best.

## Results

| | Full | MSA-PT | MSA-CPT |
|---|---|---|---|
| RULER 8K / 32K (Table 2) | 79.8 / 75.0 | 84.2 / 77.5 | 77.2 / 75.7 |
| RULER 128K (Table 3) | 72.00 | - | 72.12 |
| HELMET 128K (Table 3) | 46.53 | - | 45.93 |

- **Block size ablation (Table 4):** 32, 64 and 128 at a fixed token budget
  leave perplexity unchanged. RULER-8K goes 72.5, 72.8, 73.8 and RULER-32K
  goes 66.1, 65.3, 64.6.
- **Efficiency on the 109B testbed at 1M context:** per-token attention
  compute is 28.4x lower, and wall-clock on H800 is 14.2x faster for
  prefill and 7.6x faster for decode. Wall-clock trails the FLOP cut
  because of index, top-k, reverse-index, gather and load-balancing
  overhead.
- **Regressions after conversion:** GSM8K 73.7 vs 76.2, HumanEval 57.9 vs
  61.0, EgoSchema 25.8 vs 29.6. The outlook names a residual long-context
  retrieval gap. No NIAH or LongBench results are given, and decode KV
  memory is not analysed.

## Against HeliosLM today

HeliosLM's sparse path follows DSA. It does token-level top-k
(`attention.sparse_top_k`) with the head-mean or learned lightning indexer
(`attention/lightning_indexer.py`), keys come from the MLA latent `c_kv`,
IndexShare is available, and the indexer is distilled from the head-mean
teacher through `MLA.indexer_distill_loss`.

| Point | MSA | HeliosLM | Take |
|---|---|---|---|
| Granularity | 128-token blocks, max-pooled | single tokens | Blocks fit tensor cores and paged KV. Worth a flag when a GPU kernel exists; no CPU gain. |
| Who shares a selection | each GQA group | all heads (MQA-mode MLA, like DSA) | No direct analogue: absorbed MLA has one latent KV head. |
| Forced blocks | own block only | current token forced | Matches MSA's finding that sinks and windows add little. |
| Indexer gradient | detached at input | reaches the latent projections | **Adopted (v5.101):** `indexer_distill_loss(..., detach_input=True)`. Default unchanged. |
| Top-k on raw scores | yes | yes (no softmax in the indexer) | Already the same. |
| Dense early layers | first 3 | hybrid: MLA layers are sparse when enabled | Worth trying with OS7: keep the first MLA layer dense. |
| Warm-up | 40B tokens dense before sparse | none specified | Record in the training plan: warm the indexer under full attention first. |
| Loss | LM + KL | KL (distill) | Same KL idea; HeliosLM's teacher is the head-mean score, not the selected-token distribution. |

## Follow-ups

1. `detach_input=True` is the recommended setting for indexer distillation
   in any joint run. It is opt-in so existing runs reproduce.
2. A block-granular selection mode (`sparse_block_size`) is only worth
   building with a GPU kernel. Track it with OS7 (long-context probe), where
   RULER at 8K/32K/128K would show whether token-level selection loses
   anything against blocks at HeliosLM's scale.
3. In the training plan, warm the indexer up under full attention before
   switching to sparse, and keep one dense MLA layer at the bottom.
