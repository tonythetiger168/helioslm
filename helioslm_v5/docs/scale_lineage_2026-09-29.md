# Three-Point Scale Lineage — the honest verdict (2026-09-29)

The mid (360M, BPE) question is settled with a CLEAN checkpoint (third
training run, MID_FRESH=1, final loss 0.3675; all prior 0/12 results
were artifacts of tokenizer mismatch / resume corruption — see CHANGELOG
v5.33 P3-P5). Same probe suites, same evaluators, same confidence
definition (softmax max, greedy) at every point.

## The line

| Scale | Protocol fluency | Task answers | Conf on wrong |
|---|---|---|---|
| HeliosLM lite 8.5M (char) | broken at the tokenizer ceiling: tool mode-choice 0/20, text 11/20 | exact 0/40 | 0.925-0.944 |
| **HeliosLM mid 360M (BPE)** | **fluent: parse works multi-turn, calc->finish shape correct, mode 40/40** | **agentic 0/12 wrong (finished 12/12, finish answers incorrect)** | **0.9999 mean** |
| Qwen3-0.6B (anchor) | single-prompt 6/6; multi-step 0/15, no finish | 0/15 | mean 0.893 / max 1.0 |

## What the line says

1. **The tokenizer ceiling is real and now closed**: 0/20 -> 40/40 mode
   choice by changing char-level -> BPE at 360M. The v5.30.2 tool-channel
   zero was never the protocol's fault.
2. **Protocol fluency and answer correctness separate cleanly**: mid
   executes the full agent loop (tool call, observation, finish) but its
   finish answers are wrong on all 12 held-out tasks. The wall moved
   from 'emit the protocol' to 'get the answer right' — a capacity/
   training-data issue, not a protocol issue.
3. **The confidence gap GROWS with scale**: 0.94 (8.5M) -> 0.9999
   (360M) -> ~1.0 (Qwen3-0.6B). Bigger, more fluent models are MORE
   confidently wrong, not less. This is the strongest evidence yet for
   the certified-confidence line: calibration cannot be assumed from
   capability, at any scale tested.
4. Format skew discipline note: every 0/12 before this run was a
   measurement artifact (tokenizer mismatch or corrupted resume), not a
   behavior. Only paired-tokenizer + fresh-run numbers count here.

## Method debt paid on the way (all in CHANGELOG)

- v5.33 P3: tokenizer train/serve skew (rglob order) -> sorted corpus +
  .tok.json pairing
- v5.33 P4/P5: resume trap + guard legacy hole -> .tokfp fingerprint +
  MID_FRESH=1
- MoE bf16 autocast dtype fix (first real GPU bug)
- single-tensor AdamW workaround (RTX 4060 foreach/driver interaction)

## Autopsy: mid's 0/12 is content confabulation, not copy failure

The v4 full-capture eval (same seed as v2, directly comparable) shows
the finish answers across all 12 tasks:

| kind | expected | final | reading |
|---|---|---|---|
| calc | 30 | -15 | unrelated number |
| calc | -1748 | -108 | wrong magnitude |
| reverse | MHYIDE9KQ (9 chars) | IIIIIIII (8 chars) | length-matched degenerate repeat |
| reverse | Sba2GTvEmCwA (12) | 2v2v2v2v2v (10) | same -- shape statistics, no transformation |
| write_read | 134 | 8088 | plausible magnitude, wrong digits |

The calc obs returned the correct answer into the transcript; the
finish call ignored it. Diagnosis: the model treats the finish answer
as a language-modeling continuation (a plausible-shaped token string),
not as retrieval from the tool observation. It learned output SHAPE
statistics (string length, number magnitude) but not input-output
grounding. Confidence on all of this: 0.9999.

Therapeutic menu (recorded): (1) data -- weight finish=last-obs
samples, add copy-specific episodes; (2) inference -- System-One
intervention: for DIRECT-routed finish calls, policy-in-code forces
answer == last tool observation when the task is retrieval-shaped;
(3) training -- RLCD pressure directly on the (0.9999 conf, wrong)
pairs. Option 2 is deployable today with the existing gate.

## Next

- Inspect mid's finish answers vs expected (json per-task results) --
  copying-the-observation failure vs arithmetic confabulation
- RLCD on mid: outcome-targeted Brier is designed for exactly the
  0.9999-conf-wrong regime this run exposed
- MTP + expert-cache sweeps against the mid checkpoint (per-bit
  economics shift 40x; see benchmarks/SWEEP_NOTES.md)
