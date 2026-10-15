# Phase B v4_csa: learned compressor, round 2 (hidden 64) + oracle ceiling

Date: 2026-10-15 · Script: `examples/csa_learned_compressor.py --hidden 64 --steps 4800 --oracle-ceiling`
· Data: `csa_learned_h64_2026-10-15.json` (includes the trained weights)
· Previous round: `csa_learned_2026-10-15.md` (hidden 16, 2400 steps).
Same probe, same seed-0 evaluation stream, same identity wiring.

## Result

| arm | 4×, L=512 | 4×, L=1024 | 16×, L=512 | 16×, L=1024 |
|---|---|---|---|---|
| dense (from csa_ab) | 0.910 | 0.875 | 0.910 | 0.875 |
| **oracle pooling ceiling** | 0.900 | 0.870 | 0.860 | 0.845 |
| mean pooling (csa_ab) | 0.130 | 0.105 | 0.150 | 0.125 |
| learned, hidden 16 (v5.59) | 0.665 | 0.650 | 0.150 | 0.125 |
| **learned, hidden 64 (this round)** | **0.725** | **0.645** | **0.375** | **0.390** |

Top-8 and top-32 give the same accuracy in every learned row. Chance:
0.125; 200 trials per cell.

## What changed and how it was chosen

- Only the compressor's size and training length: hidden 16 → 64,
  2400 → 4800 steps; everything else as in v5.59.
- Chosen on the same separate validation stream (seed 2000, L=128, 150
  trials), never on the evaluation stream. Ratio 4: hidden16/2400 0.567,
  hidden64/2400 0.620, hidden64/4800 0.667, hidden256/4800 0.580.
  Ratio 16: hidden64/4800 0.387, hidden256/4800 0.340.

## Reading

1. **The design can match dense.** With a perfect compressor (oracle:
   all weight on each chunk's largest-norm token — a measuring stick
   only, it uses the probe's construction) CSA reaches 0.900/0.870 at 4×
   and 0.860/0.845 at 16×, within 0.01–0.05 of dense. The entire gap is
   the compressor's ability to single out the important token.
2. **Diagnosis behind the change:** in v5.59's failing trials the
   compressor put only 0.21 weight on the planted token (uniform is
   0.25) vs 0.88 in the successful ones — it failed to detect the token,
   not to carry it.
3. **16× now learns:** 0.150/0.125 → 0.375/0.390, about 2.5–3× chance.
4. **4× is mixed:** better at L=512 (0.665 → 0.725) but flat at L=1024
   (0.650 → 0.645). The bigger compressor helps on lengths closer to
   training; it does not yet generalize further.
5. **Still far from the ceiling** (0.645–0.725 vs 0.87–0.90 at 4×;
   0.38–0.39 vs 0.85–0.86 at 16×). A larger compressor alone is not the
   answer here (hidden 256 validated worse); what is missing is a
   compressor that reliably detects salient tokens at any length.

## Caveats

Constructive probe with identity wiring, n = 200 per cell, one training
seed per ratio. The oracle uses knowledge of how the probe plants
tokens and is not a deployable compressor.
