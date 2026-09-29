# Copy curriculum verdict: NO effect at 360M (2026-09-29)

Controlled A/B on the mid agentic suite (same checkpoint recipe, same
eval seed, same 12 tasks -- only the training data differs):

| run | training data | agentic v4 result |
|---|---|---|
| mid_sft_v5.33 (baseline) | chat SFT (3526 tool / 1200 text) | 0/12, confabulated args (`19 * -92` -> `12 * -9`, `mhyIdE9KQ` -> `wIIIIIIK9`), conf 0.9999 |
| mid_sft_v5.33 + 800 echo | chat SFT + copy curriculum | 0/12, BYTE-IDENTICAL confabulations on the same tasks, conf 0.9999 |

Grounding (v5.34) takes the same checkpoint to 12/12. The curriculum
does nothing for the copy disease.

## Diagnosis narrowed (three eliminations)

1. NOT the tokenizer: BPE at 360M already fixed mode-choice 0/20 -> 40/40.
2. NOT the protocol: agentic loop is fluent (correct calc->finish shape).
3. NOT data volume / curriculum: 800 echo copy-specific episodes,
   confabulation unchanged to the byte.

Remaining hypothesis: a CAPACITY/DYNAMICS wall at 360M -- the model
learns output-shape statistics but character-level copy grounding
(attention localization over long token spans) does not emerge from
SFT at this scale. Recorded honestly: this is a single-point
measurement at one model scale; it does not rule out that copy emerges
at larger scale, or with RL pressure, or with architectural changes.

## Consequence for the stack

GroundingGate is not a workaround -- it is the correct answer at this
scale. The retirement path for grounding shifts from "more copy data"
to: (a) larger-scale check, (b) RL pressure (GRPO with copy-shaped
rewards -- async skeleton ready, v5.31), (c) architectural (retrieval
heads / constrained decoding). The copy curriculum module
(agent/copy_curriculum.py, EchoEnv) stays -- echo remains the cleanest
copy probe, and the data is cheap to generate for any future run.

Artifacts: mid_agent_eval_v4(1).json (this run, uploaded by user) vs
benchmarks/mid_agent_eval_v4.json (baseline) -- per-task byte
comparison available.
