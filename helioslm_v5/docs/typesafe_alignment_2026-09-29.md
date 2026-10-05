# TypeSafe/Jev alignment report (2026-09-29)

Sources: TypeSafe official docs (docs.typesafe.ai), the RLCDAlignBench
paper ("Just Ask Jev: RL for Calibrated Decisions as a Zero-Shot
Detector of AI Alignment Failures", under review at ICLR 2027), and
our v5.32-35 stack. Purpose: calibrate our open-source reference
against the canonical vendor implementation, record what we matched,
what differs, and what to adopt.

## Concept alignment: what we already match

| Our v5.32-35 design | TypeSafe / Jev |
|---|---|
| Three typed primitives Choice/Score/Noul | identical names and roles |
| ask_batch: K questions answered in ONE forward (T30 counts passes) | "questions evaluated in parallel; adding questions barely changes latency"; measured 13-question batch = 11.5x cheaper, 9.6x faster |
| Model decides sequence, code fills content (GroundingGate) | "atomic questions, composed in code -- change weights, not prompts" |
| Confidence tiers drive behavior (TrustGate design) | official three-tier behavior: high-confidence auto-execute / medium caution / **low do not act** |
| tau threshold scales with risk | "thresholds scale with risk: different thresholds for different operations, your code encodes risk tolerance" |
| confidence from the distribution (max softmax) | confidence is a statistic of the probability distribution (peakedness), user-replaceable |

## Three design differences (recorded, one is an adoption)

1. **Noul (adoption candidate -- now triple-evidenced)**. Ours: ternary
   yes/no/unknown + independent confidence; floor at 1/K=0.5 on binary
   Choice (v5.32.1 finding). Official: P(yes) in [0,1] continuous, NO
   independent confidence -- "near 0.5 = uncertain". AlignBench reads
   the same way. Triple evidence points the same direction: evolve our
   Noul to the continuous form; sub-floor uncertainty becomes a
   probability, not a separate confidence scalar.
2. **Score**. Ours: free [0,1] sigmoid. Official: ordered-tier rubric
   (user-defined tiers, model returns a distribution, score can fall
   between tiers) -- anchored semantics, more interpretable. They warn
   explicitly: Noul 0.5 != "medium degree"; use Score for degrees.
3. **Confidence definition**. Ours returns max-prob; official returns
   the full distribution plus a derived confidence, and states
   "confidence != probability, and you are not locked into our
   definition".

## RLCDAlignBench: where our repo sits in their plane

They are the **task-INTER** branch: detecting alignment failures in
OTHER LLMs' outputs (10 failure classes -- sycophancy, jailbreak,
deception, prompt injection, hallucination, privacy, bias, reward
hacking, concealing uncertainty, power seeking -- x 44 benchmarks x
7,193 instances, 5 target models 2-7B). We are the **task-INTRA**
branch: the same agent deciding whether to trust its own tool calls
(task text -> groundable -> outcome). Same RLCD methodology (probabilities
answer to outcomes, not preferences); same conclusions: read
probabilities not argmax, thresholds scale with risk, calibration
signal is cheap and strong.

Headline numbers: zero-shot single generic Noul question reaches 0.886
median AUROC on 31 applicable benchmarks (vs TF-IDF/length baselines);
~1/63 the cost of an API LLM-judge; reading probabilities (not argmax)
is load-bearing. Their "confident dissent" detail: Jev's low-confidence
readouts helped find LABEL DEFECTS in 3 existing benchmarks --
calibration signal strong enough to repair benchmark ground truth.

## Two implementation details worth adopting

- Question IDs are for YOUR code, not sent to the model -- the full
  question text lives in instructions. Our ask() currently uses the
  description as the question body; adopt the ID/content split.
- Their eval protocol: half the data selects the question, the other
  half scores it -- best-of over the same data inflates. Applies
  unchanged when TrustGate starts asking multiple questions.

## Standing comparison for the writeup

8.5M char: conf-on-wrong 0.94, tool mode-choice 0/20. 360M BPE
(mid): conf-on-wrong 0.9999, agentic 0/12 ungrounded -> 12/12
grounded. Qwen3-0.6B anchor: conf ~1.0, multi-step 0/15. AlignBench
(much larger scale): the same RLCD readout separates alignment
failures at 0.886 AUROC. Confidence on wrong answers grows with scale
at every point we can measure -- the through-line of the whole
certified-confidence thesis.
