# Phase B v4_csa: learned compressor — passkey A/B follow-up

Date: 2026-10-15 · Script: `examples/csa_learned_compressor.py` · Data:
`csa_learned_2026-10-15.json` (includes the trained compressor weights)
· Baseline: `csa_ab_2026-10-15.md` (mean-pooled CSA). Same probe, same
seed-0 evaluation stream, same identity wiring.

## Result

| arm | L=512 | L=1024 | indexer picks the right chunk (L=512 / 1024) |
|---|---|---|---|
| dense (reference, from csa_ab) | 0.910 | 0.875 | — |
| CSA 4×, mean-pooled (csa_ab) | 0.130 | 0.105 | 1.000 / 0.995–1.000 |
| **CSA 4×, learned compressor** | **0.665** | **0.650** | 0.720–0.775 / 0.745–0.775 |
| CSA 16×, mean-pooled (csa_ab) | 0.150 | 0.125 | 0.925–0.980 / 0.855–0.965 |
| CSA 16×, learned compressor | 0.150 | 0.125 | 0.930–0.980 / 0.845–0.965 |

Top-8 and top-32 give the same accuracy in every row; hit-rate ranges
span top-8 to top-32. Chance: 0.125. 200 trials per cell.

**Selection check (4×, L=512, same stored weights):** selecting EVERY
chunk (top-1000) also gives 0.665, identical to top-8 and top-32. With
top-8, accuracy is 125/144 when the planted chunk is selected and 8/56
when it is not. T93 pins the top-8 == all-chunks equality.

## What was trained, and how it was chosen

- Only the compressor is trained: an MLP softmax-pooler (hidden 16,
  zero-initialised so it starts as exact mean pooling). q/k/v/out and
  the indexer stay identity-wired and frozen.
- Loss: end-to-end cross-entropy of the passkey readout through the full
  CSA forward; the compressor is never told which token is planted.
- Training data: a separate seed stream (1000) at L=128. Evaluation: the
  A/B seed-0 stream at L=512/1024 — unseen and 4–8× longer.
- Settings (logits ÷ amp, lr 1e-2, 2400 steps) were chosen on training
  loss and a separate validation stream (seed 2000, L=128), never on the
  evaluation stream. Validation at 4×: 0.487 / 0.533 / 0.567 for
  lr 3e-3·1200 / 1e-2·1200 / 1e-2·2400 steps.

## Reading

1. **A learned compressor restores most of the exact recall at 4×**
   (0.13 → 0.67, about 73% of dense), and it holds at sequence lengths
   4–8× longer than it was trained on.
2. **Selection is NOT the bottleneck.** The hit rate drops to
   72.0–77.5%, but selecting every chunk leaves accuracy unchanged at
   0.665: when the planted chunk is missed, its pooled entry also scores
   too low to win in attention. With identity wiring the indexer's
   scores ARE the attention scores, so the usual fix — distilling the
   indexer toward attention — has nothing to learn here. The remaining
   gap (0.665 vs dense 0.910) is in the compressed entry itself: the
   compressor does not always make the planted token dominate its
   chunk's entry.
3. **16× does not learn** under the same budget (training loss stays at
   the 8-way chance level of ln 8 ≈ 2.08): one softmax-pooled entry per
   16 tokens could not be shaped into an exact-token carrier here.
   Recorded as a limit of this compressor and budget, not of CSA.

## Caveats

Constructive probe with identity wiring, n = 200 per cell, one training
seed. The compressor is our own design, not DeepSeek-V4's.
