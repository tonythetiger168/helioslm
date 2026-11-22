# CALIBRATION — confidence that answers to outcomes

Status: measured, 2026-11-23 (P2 outcome-recalibration row added the
same day). Every number below was produced by running the code in this
repo on this checkout; nothing is assumed or copied from a vendor claim.

## Why this document exists

HeliosLM routes real work on confidence: the trust gate DIRECTs or
ESCALATEs, the swarm quorums on it, the decision head emits it. A
confidence score that stays high when the model is wrong is worse than no
score at all — it converts errors into silent, auto-approved decisions.
This document pins down what we actually measure, the current numbers,
and the known gap.

## The core finding: calibration on wrong answers

Historical evidence already in the repo
(`checkpoints/mid_agent_eval.json`, a ChatSession agent run):

| Metric | Value |
| --- | --- |
| tasks correct | 0/12 |
| mean confidence on those wrong answers | **1.0000** (`conf_on_wrong_mean`) |

The agent was wrong on every task and reported ~maximal confidence on
every one of them. `mid_agent_eval_v2.json` (finished 12/12, still 0/12
correct) shows the same: `conf_on_wrong_mean = 0.9999`. Confidence was
uncorrelated with correctness — a coin flip would have been more
informative.

## Measured baseline: DecisionHead under distribution shift

Experiment (fully reproducible, see below): the non-autoregressive
`DecisionHead` (`agent/decision_head.py`) is trained on 400 synthetic
routing records (`route` = 3-way choice, `solvable` = noul), then
evaluated on 200 held-out in-distribution records and 200 records under
an inverted rule (the marker mapping is flipped, so a head that learned
the training rule is systematically wrong).

| Metric (route head) | Held-out ID | Shifted |
| --- | --- | --- |
| accuracy | 0.995 | 0.340 |
| 10-bin ECE | 0.265 | 0.390 |
| AUROC (confidence vs correctness) | 0.990 | **0.545** |
| Brier | 0.088 | 0.382 |
| noul accuracy (solvable) | 0.995 | 0.995 |

Two honest readings:

1. **In-distribution the head is under-confident, not calibrated.** ECE
   0.265 at 99.5% accuracy means reported confidence sits well below
   empirical correctness — the CE+Brier objective (Brier tied to outcome
   targets) pulls probabilities inward. Safe direction, but it is not
   calibration.
2. **Under shift, confidence stops ranking correctness.** AUROC 0.545 is
   a coin flip: on the shifted data the head cannot tell its right
   answers from its wrong ones, and ECE nearly doubles. This is the toy
   version of the `conf_on_wrong_mean = 1.0` failure above, now with a
   measured mechanism.

## P2: closing the shift gap with verified outcomes (measured)

The shift above is a CONCEPT shift — the inverted split reuses the exact
same state words, so p(x) is unchanged and only p(y|x) flipped. No
input-side mechanism can detect that: density/OOD scores, ensembles of
identically-trained heads, and split-conformal are all blind to it by
construction. The only signal that can separate right from wrong there
is an OUTCOME, which the verified-decision engine already defines: a
decision counts as verified when an oracle or replay check backs it
(`cost_per_verified_decision.md`).

`agent/outcome_calibration.py` spends that signal. Each verified outcome
is stored as the head's own encoder features plus the 0/1 verdict; a
small ridge-logistic probe learns `P(correct | state)` from exactly
those verified records, and reported confidence is rescaled by it.
Answers are never changed, so accuracy is untouched by construction and
every point of lift is confidence ranking. Cold start (no verified
outcomes yet) is the identity — an uninstrumented deployment can never
be silently degraded.

Measured (same probe, same seeds; the first 40 of the 200 shifted
decisions are verified, the remaining 160 are scored after warmup):

| Metric (route head, shifted) | Raw | After 40 verified outcomes |
| --- | --- | --- |
| accuracy (post-warmup subset) | 0.356 | 0.356 (unchanged) |
| 10-bin ECE | 0.390 (full split) | **0.155** |
| AUROC (confidence vs correctness) | 0.545 (full split) | **0.935** |
| Brier | 0.382 (full split) | **0.083** |

Honest boundaries of this result:

- The lift costs verified decisions — 40 of 200 here (20%). With zero
  outcome budget the shift row stands at AUROC 0.545 and nothing in
  this repo claims otherwise.
- The calibrator knows nothing about the task or the shift; it sees
  encoder features and verdicts only. It does not make the head
  shift-robust — accuracy stays 0.356 — it makes confidence answer to
  verified outcomes, which is what a gate should route on.
- The mechanism and its guards (cold-start identity, single-class
  buffer refusal, 0/1 verdict enforcement) are pinned in
  `tests/case_outcome_calibration.py`; the numbers reproduce with
  `python3 docs/calibration_probe.py`.

## What we measure (definitions as implemented)

- **ECE** — 10 equal-width bins over max-softmax confidence;
  `sum_b (n_b/N) * |conf_b - acc_b|` (`DecisionHead.evaluate`).
- **AUROC** — probability that a correct answer carries higher
  confidence than an incorrect one (`eval/calibration_board.auroc`).
- **Brier** — mean squared error of confidence vs the 0/1 correctness
  outcome (`eval/calibration_board.brier`).
- **noul** — continuous P(yes); reported as mean |P − target| and
  threshold accuracy. Uncertainty is a probability near 0.5, not a class.

## Reproduce

```bash
# DecisionHead shift experiment (CPU, ~1 min): trains on 400 records,
# evaluates on 200 in-distribution + 200 rule-inverted records, then
# streams the same shifted records through the P2 outcome calibrator
# (40 verified) — prints all three blocks above as JSON.
python3 docs/calibration_probe.py   # self-contained, seeds fixed
```

The probe asserts nothing about specific values — it prints measured
numbers. If a future training change improves calibration, update this
document with the new measured row, not the old one.

## Implications for the verified-decision engine

- Never gate on raw confidence from a model outside its training
  distribution; AUROC-on-shift is the metric that must clear a floor,
  not ID ECE. Wherever decisions are verified anyway, feed the verdicts
  to `OutcomeCalibrator` and gate on the recalibrated confidence — the
  measured row above is the difference between routing on a coin flip
  (0.545) and routing on a ranked signal (0.935).
- A decision is "verified" only when an oracle or replay check backs it
  (`cost_per_verified_decision.md` builds the cost model on exactly that
  definition: `correct AND final == expected`).
- The trust gate's quant-drift widening (`test_quant_drift_trust_gate`)
  is the right shape of mitigation: detect shift, widen escalation,
  never silently trust.
