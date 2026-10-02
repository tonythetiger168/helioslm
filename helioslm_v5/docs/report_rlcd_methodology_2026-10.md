# Calibrated Agency: an open-source RLCD stack, from toy to 360M (2026-10-02)

Status: working report. Every claim below has an artifact and a test in
github.com/tonythetiger168/helioslm; version tags cite CHANGELOG entries.

## 1. The question

LLMs decide tool calls, routes, and answers -- and state confidence.
Three reinforcement paradigms exist for training them: RLHF (optimize
human preference), RLVR (optimize verifiable answers), RLCD (optimize
CALIBRATION: probabilities answer to outcomes, not preferences).
The first two are mainstream; the third is the whole thesis of the
commercial "decision layer" category (Jev / TypeSafe, 2026-09) and,
as far as we can tell, has no open-source reference implementation with
scale evidence. This repo is that reference.

## 2. What we built (v5.32-v5.37)

A complete open-source RLCD stack, each layer independently tested:

| Layer | Module | Verdict artifact |
|---|---|---|
| Typed decision primitives | decision.py (Choice/Score/Noul) | T27 5/5 |
| Non-autoregressive decision head | decision_head.py (one forward, K questions) | T28 4/4 |
| Outcome records + reward shaping | decision_data.py (RecordingGate, rlcd_reward_adjustment) | T29 3/3 |
| Head-backed gate (one-pass pi-warden) | calibrated_router.py | T30 4/4 |
| Continuous Noul (P(yes) in [0,1], no separate confidence) | v5.36 | T37 3/3 |
| Deterministic grounding (policy in code) | grounding.py | T33 6/6; mid 0/12 -> 12/12 |
| Calibrated abstention | trust_gate.py | T38 4/4; mid acceptance pending |
| Grounded-replay audit | verify_grounded_replay | T39 2/2 |

Plus the infrastructure the paradigm needs: a task grammar shared by
producer and consumer (task_grammar.py, T34), a byte-level BPE
tokenizer trained on the repo's own corpus (bpe.py, T31), a 360M
"middle rung" model (mid, v5.33), async GRPO with bitwise sync parity
(v5.31, T26), and a long-horizon env family (file_env, T24).

## 3. The evidence chain (all measurements, no anecdotes)

1. **The disease is scale-invariant**: wrong answers at confidence
   0.94 (8.5M char toy), 0.9999 (360M BPE), ~1.0 (Qwen3-0.6B anchor,
   probed with our own harness). Confidence on wrong answers GROWS
   with scale (docs/scale_lineage_2026-09-29.md).
2. **The mid failure is copy-shaped, not knowledge-shaped**: the model
   executes the agent protocol perfectly but confabulates content at
   both ends (task->args dominates; obs->finish mostly works).
   Controlled A/B: +800 copy-curriculum episodes changed the
   confabulations BYTE-IDENTICALLY (COPY_CURRICULUM_VERDICT.md).
   Three eliminations: not tokenizer, not protocol, not data volume.
   Remaining hypothesis: capacity/dynamics wall at 360M.
3. **Grounding is the correct answer at this scale**: same checkpoint,
   same eval seed, GroundingGate takes the agentic suite 0/12 -> 12/12
   (benchmarks/mid_agent_eval_v4.json vs _v5.json). Policy-in-code is
   not a workaround; it is the therapy.
4. **The model already flags OOD**: an accidental tokenizer-mismatch
   run scattered confidences (0.13-0.97) where the clean run pins
   ~0.9999. The calibration gap is precise: overconfidence on
   IN-DISTRIBUTION errors only (v5.35f).
5. **Failure is partially state-predictable**: an RLCD-trained head
   separates mid's right/wrong states at 0.645/0.521 (n=24) where the
   model's own confidence carries zero signal (0.9999/0.9999,
   benchmarks/rlcd_head_summary.json).
6. **Label-targeted calibration is redundant; outcome-targeted bites**:
   Brier(conf, train label) duplicates CE (proper-scoring-rule
   equivalence, A/B-measured); Brier(conf, replay-verified outcome) is
   the load-bearing term (T28/T29).
7. **Binary confidence has a 1/K floor; continuous Noul retires it**:
   sub-50% uncertainty was inexpressible on the ternary form; P(yes)
   expresses it by construction (T37). Triple-evidenced: our floor
   finding + TypeSafe's official spec + RLCDAlignBench's readout.

## 4. The therapy pair

Grounding fixes actions the model TAKES (0/12 -> 12/12). TrustGate
withholds actions on states the head DISTRUSTS (low P(correct) ->
ESCALATE; loud without an oracle, delegated with one). Together they
are the deployed form of the TypeSafe behavior tiers -- and both halves
are independently verified, which no vendor publishes.

## 5. Boundaries (honest)

- 24 real outcome records is demonstration-grade, not production-grade;
  the value is the closed pipeline any larger run plugs into.
- Grounding coverage is the six known task grammars; unknown grammars
  pass through ungrounded (groundable() reports coverage).
- The 360M capacity finding is a single-point measurement; copy may
  emerge at larger scale or under RL pressure (async GRPO ready).
- Our probes are toy-grade by construction; the Qwen3-0.6B anchor is
  the smallest frontier-family model, not the frontier.

## 6. Positioning against the vendor line

RLCDAlignBench (ICLR 2027 under review) detects alignment failures in
OTHER models' outputs (7,193 instances, 0.886 median AUROC zero-shot).
We are the task-INTRA branch: the same RLCD readout lets an agent
police itself. Methodology conclusions agree: read probabilities not
argmax, thresholds scale with risk, calibration signal is cheap. Our
contribution is the open, replay-auditable, scale-measured version --
and the finding that where RLVR-style training stops (answers right),
RLCD starts (claims honest).
