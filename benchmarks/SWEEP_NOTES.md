# Inference sweep notes (helioslm-toy-8.5M)

Two local sweep artifacts, recorded 2026-09-28 from runs dated 09-18 and
09-23. Both are inference-side benchmarks on the lite model: MTP
draft-token speculative decoding and sigmoid-MoE expert streaming.

## MTP draft-depth sweep (depth 1 vs autoregressive baseline)

| Run | Cache | Acceptance | Gain vs draft-0 | tok/s |
|---|---|---|---|---|
| 09-18 | cold | 0.720 | **-0.003** | 146.5 |
| 09-18 | warm | 0.720 | +0.054 | 154.9 |
| 09-23 | cold | 0.788 | **+0.352** | 207.6 |
| 09-23 | warm | 0.788 | +0.146 | 202.5 |

Findings:
1. Acceptance rose 0.720 -> 0.788 between the runs (draft quality
   improved; the artifact pair preserves both regimes).
2. The 09-18 run shows speculative decoding at ~79% acceptance still
   NOT paying off on cold cache (-0.3%) -- overhead eats the wins. By
   09-23 the same acceptance profile yields +35% cold. Whatever changed
   between the runs (draft overhead reduction) crossed the payoff line.
   Recorded as the regime boundary for MTP economics on CPU.

## Expert streaming cache sweep (resident experts vs hit rate)

| Resident | Budget | Hit rate | Evictions | RAM saving vs dense |
|---|---|---|---|---|
| 1 | 24 KB | 0.000 | 61 | 0.875 |
| 2 | 48 KB | 0.000 | 60 | 0.75 |
| 4 | 96 KB | 0.000 | 58 | 0.5 |
| 8 | 192 KB | **0.871** | 0 | 0.0 |

Finding: a hard CLIFF, not a curve. The workload touches 8 distinct
experts (62 loads); below 8 resident the cache thrashes at 0% hits with
constant eviction churn, at 8 it fits the working set (87% hits, zero
evictions). No intermediate regime exists in this data -- streaming
expert RAM saving is effectively all-or-nothing relative to the working
set, and "87% of dense RAM" is the honest cliff-edge number for this
workload, not the 0.875 headline.

## Relation to mid

These are lite-era inference benchmarks. Mid (360M) shifts the
economics: per-token cost is ~40x higher, so speculative decoding
acceptance and expert-cache hit rates matter correspondingly more.
Re-running these sweeps against the mid checkpoint is v5.33 follow-up
work.
