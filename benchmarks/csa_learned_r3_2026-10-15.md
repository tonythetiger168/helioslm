# Phase B v4_csa: learned compressor, round 3 (4×, longer training)

Date: 2026-10-15 · Script: `examples/csa_learned_compressor.py --ratios 4 --hidden 64 --steps 9600 --oracle-ceiling`
· Data: `csa_learned_r3_2026-10-15.json` (includes the trained weights)
· Previous rounds: `csa_learned_2026-10-15.md` (v5.59), `csa_learned_h64_2026-10-15.md` (v5.60).
Same probe, same seed-0 evaluation stream, same identity wiring.

## Result (4×)

| arm | L=512 | L=1024 |
|---|---|---|
| dense (from csa_ab) | 0.910 | 0.875 |
| oracle pooling ceiling | 0.900 | 0.870 |
| learned, hidden 16, 2400 steps (v5.59) | 0.665 | 0.650 |
| learned, hidden 64, 4800 steps (v5.60) | 0.725 | 0.645 |
| **learned, hidden 64, 9600 steps (this round)** | **0.825** | **0.800** |

Top-8 and top-32 give the same accuracy. The indexer now picks the right
chunk 92.5–93.5% of the time. Chance: 0.125; 200 trials per cell.

16× is **not** re-run here: on validation, 9600 steps did not beat 4800
(below), so 16× stays at its v5.60 result (0.375 / 0.390).

## How the settings were chosen (validation stream only)

Separate validation stream: seed 2000, 150 trials, at L=128 (training
length) and L=512. The evaluation stream was never used for a choice.

| candidate | 4×, L=128 | 4×, L=512 | 16×, L=128 | 16×, L=512 |
|---|---|---|---|---|
| GELU, hidden 64, 4800 steps (v5.60 weights) | 0.667 | 0.693 | 0.387 | 0.380 |
| squared ReLU, hidden 64, 4800 steps | 0.633 | 0.533 | 0.407 | 0.367 |
| **GELU, hidden 64, 9600 steps** | **0.747** | **0.807** | 0.313 | 0.387 |

## Reading

1. **4× reaches about 92% of the oracle ceiling** at both lengths
   (0.825/0.900, 0.800/0.870), and the L=1024 shortfall of v5.60 is
   gone (0.645 → 0.800). The compressor was under-trained, not
   under-sized.
2. **Negative result — squared ReLU:** the idea that a quadratic
   activation would detect the high-norm token more easily did not hold;
   it is clearly worse at 4× on validation. `compressor_act="relu2"`
   stays available but is not used.
3. **Negative result — 16× with more steps:** doubling training did not
   help at 16× (0.313/0.387 vs 0.387/0.380 on validation). 16× needs a
   different idea, not more of the same.

## Caveats

Constructive probe with identity wiring, n = 200 per cell, one training
seed. The oracle ceiling uses knowledge of how the probe plants tokens.
