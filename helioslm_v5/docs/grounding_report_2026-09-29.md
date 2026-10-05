# Grounding Report: from 0/12 confabulation to 12/12 with policy-in-code (2026-09-29)

The complete closed loop of the v5.33-v5.34 line: disease discovery at
scale, autopsy, therapy design, synthetic proof, and the real-model
before/after verdict.

## The before/after (same checkpoint, same eval seed 2026-0928)

| eval | gate | correct | behavior |
|---|---|---|---|
| mid_agent_eval_v4 | FixedGate(DIRECT) | **0/12** | protocol fluent, every finish answer confabulated (wrong exprs like "19 * -92" -> "12 * -9"; length-matched degenerate repeats), confidence 0.9999 |
| mid_agent_eval_v5 | GroundingGate(DIRECT) | **12/12** | model keeps the tool-sequence decision; policy in code fills both ends; finals match expected verbatim |

Checkpoints: mid_sft_v5.33.pt (360M, BPE, final loss 0.3675) with its
paired tokenizer (v5.33 P3-P5 pairing + fingerprint discipline).

## Why this matters (three claims, three pieces of evidence)

1. **Fluency != correctness**: mid executes the agent protocol perfectly
   while hallucinating content at 0.9999 confidence (v4 autopsy,
   benchmarks/mid_agent_eval_v4.json).
2. **The failure is copy-shaped, not knowledge-shaped**: the model never
   learned to ground args/answers in context; it learned output-shape
   statistics. Copy is a POLICY problem, not a capacity problem -- at
   least at this scale and grammar.
3. **Policy-in-code is sufficient therapy**: when the task grammar is
   known, deterministic extraction beats more training. T33 proved it
   synthetically (a policy that confabulates EVERY argument scores
   24/24 under grounding); v5 proves it on a real model.

## Architecture (v5.34)

GroundingGate wraps any gate. On Route.DIRECT only:
- task -> tool args: parsed from the task text by family grammar
  (calc/str/compose + three file-env families)
- tool chain -> finish answer: the deterministic replay of tools.calc /
  tools.str_op over the grounded args (tools are deterministic, so the
  prediction IS the observation); finish anchors to family SEMANTICS,
  not sequence shape (code-review fix: last-event anchoring could
  produce grounded-but-wrong on deviation)
- ESCALATE / oracle paths are never touched; unknown grammars pass
  through ungrounded (groundable() reports coverage)

The model's remaining job -- tool SEQUENCE selection -- is exactly the
decision type small models can learn (v5.30.2: chat mode-choice text
11/20 at 8.5M; v5.33: 40/40 at 360M).

## Boundaries (honest, from the code review)

- Coverage = the six known task grammars. New grammars need new rules;
  the TASK_GRAMMAR triplication (grounding/test/env generators) is a
  recorded future skew surface to unify.
- Grounded trajectories do not verify under verify_chat_replay (the
  recorded call is the grounded one; replay re-derives raw). Verified
  by determinism instead; grounding-aware replay is future work.
- write_transform file_reads redirect to the result file; _state is
  per-session and not thread-safe (documented in the module).
- This is a THERAPY, not a cure: the model itself still cannot copy.
  The retirement path for grounding is the copy curriculum (data) and
  RLCD (training pressure on conf-wrong pairs) -- both queued.

## Numbers for the record

8.5M char (lite): tool mode-choice 0/20, text 11/20, conf-on-wrong 0.94.
360M BPE (mid) ungrounded: 0/12 agentic, conf-on-wrong 0.9999.
360M BPE (mid) grounded: 12/12 agentic.
Qwen3-0.6B anchor: single-prompt 6/6, multi-step 0/15, conf ~1.0.
Confidence on wrong answers grows with scale at every point tested.
