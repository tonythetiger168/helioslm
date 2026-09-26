# HeliosLM vs Top-10 LLMs — benchmark position paper (2026-09-27)

Purpose: an honest, sourced benchmark positioning. Not a capability claim.
Method: verified public leaderboard data (2026-08-26/08-31/09-27 pulls) +
HeliosLM's own published artifacts. Everything unmeasurable is marked
"not comparable" rather than estimated.

## 0. Which leaderboard? (they disagree)

Three mainstream boards give three different top-10s in Sept 2026 —
task benchmarks (SWE-Bench Verified), human preference (Arena Elo), and
composite quality all rank differently, and Opus 5 is #1 on one board
and #7 on another. Scores at the top are compressed (quality index
97-100 for ranks 6-10). This paper uses the swfte composite (56 models,
updated 2026-09-27) as the primary table, cross-referenced with vals.ai
independent SWE-Bench runs (2026-08-26) and Arena Elo (2026-08-31).

## 1. The top 10 right now (composite quality, swfte 2026-09-27)

| # | Model | Quality | Arena Elo | $/M in/out | Notes |
|---|---|---|---|---|---|
| 1 | Claude Fable 5 | 100 | 1525 | $10/$50 | Frontier agentic coding |
| 2 | Claude Mythos 5 | 100 | 1531 | $10/$50 | Ceiling capability, limited access |
| 3 | Claude Opus 5 | 99 | 1522 | $5/$25 | #1 on vals.ai SWE-Bench (97.00%) |
| 4 | Claude Opus 4.8 | 98 | 1512 | $5/$25 | SWE-Bench 88.6% (vendor) |
| 5 | GPT-5.6 Sol | 98 | 1514 | $5/$30 | SWE-Bench 96.20% (vals.ai) |
| 6 | GPT-5.5 | 97 | 1506 | $5/$30 | |
| 7 | Kimi K3 OSS | 97 | 1500 | $3/$15 | **Best open weight on this board** |
| 8 | GPT-5.5 Pro | 96 | 1510 | $30/$180 | Reasoning-at-any-cost tier |
| 9 | Claude Opus 4.7 | 96 | 1505 | $5/$25 | |
| 10 | Gemini 3.1 Pro | 96 | 1505 | $2/$12 | Cheapest in top 10 |

Just outside: Qwen3.8 Max OSS (96 quality, #11, $2/$6, best value 24.0),
Qwen 3.7 Max (#13), Grok 4.5 (#14). Independent SWE-Bench (vals.ai)
order differs at the top: Opus 5 > DeepSeek V4 Pro 0813 (96.40%, best
open weight there) > GPT-5.6 Sol > Grok 4.6 > GPT-5.6 Terra = GLM-5.3.

Notable for this project: Moonshot's Kimi K3 OSS holds #7 composite and
the open-weight crown on one board; the open-weight task order is
DeepSeek V4 Pro > GLM-5.3 > Kimi K3. The open/closed gap on SWE-Bench is
under one point.

## 2. Where HeliosLM sits (same table, no rounding up)

| | HeliosLM v5.30.2 | Frontier top-10 |
|---|---|---|
| Parameters | 8,495,760 | ~10^11-10^12 (V4 Pro: 1.6T total / 49B active) |
| SWE-Bench V | n/a (not submitted — would score ~0) | 95.4-97.0% |
| Arena Elo | n/a | 1500-1531 |
| Quality index | n/a | 96-100 |
| Inference cost | $0 (2 CPU threads) | $0.10-1.29 per resolved task |
| Context | ~1,500 chars effective | 1M tokens standard |
| Tokenizer | char-level ord<1024 | BPE-family |

Gap: ~5 orders of magnitude. Anyone claiming a toy can benchmark against
frontier models on capability numbers is selling something. HeliosLM does
not compete on this axis and this document does not pretend otherwise.

## 3. The axes where comparison IS meaningful

Frontier benchmark numbers are increasingly distrusted (vendor GA figures
published as images only; third parties measuring 94% hallucination rates
on unreleased weights; SWE-Bench saturating with 7 models inside 5
points). HeliosLM's research line targets exactly this trust gap. On
these axes we publish numbers that no top-10 vendor publishes at all:

| Axis | HeliosLM artifact | Top-10 status |
|---|---|---|
| Confidence calibration on WRONG answers | max conf 0.925-0.944 while 12/12 wrong (three-modes artifact, T19 PASS) | not published by any vendor |
| Routing-gate monotonicity | strict-monotone tau curve 0.000->0.083->0.250->0.333->1.000 (regenerated artifact) | not published |
| Trajectory replay verification | verify_replay / verify_chat_replay / verify_think_replay — every recorded step re-derivable | not published |
| Eval-prompt data bias | caught own <700 filter dropping 100% of text targets (v5.30.2) | vendors' filters unaudited |
| Per-save checkpoint provenance | HF-synced, sha256-logged every 50 steps | n/a |

## 4. The smallest honest anchor (blocked, recorded)

A real head-to-head needs a model we can run locally. The smallest
credible open model from the current families (Qwen3.8 27B at 4-bit,
~17GB) exceeds this sandbox's CPU/RAM, and the `transformers` runtime is
not installable here (recorded 2026-09-27). Fallback options, in order of
cost: (a) user provides any OpenAI-compatible API key — we run the probe
suite in one afternoon; (b) rent a GPU host for a Qwen3-0.6B-class anchor
run; (c) wait for milestone-13 infra.

## 5. Probe suite (ready to run against any API)

`helioslm_v5` toy envs (12 tasks) + `file_env` long-horizon families +
chat protocol probes. Scored on: task accuracy, tool-parse rate,
mode-choice, and — the metric nobody else reports — confidence on wrong
answers. Scripted policy oracles already exist for every family
(T17/T24), so a frontier model's failures are classifiable, not anecdotal.

## 6. Verdict

On capability benchmarks HeliosLM is not in the top 10, the top 5,000,
or the top 50,000 — it is a research vehicle 5 orders of magnitude below
the frontier, and its value is that every number it publishes can be
re-derived by anyone with 2 CPU threads. The day frontier vendors publish
calibration-on-wrong-answers curves with replay-verifiable trajectories,
this project has won its argument and can retire.

Sources: swfte.com/ai/leaderboard (2026-09-27, 56 models); vals.ai
SWE-Bench Verified (2026-08-26 update, 86 models); LMArena/Arena text
Elo (2026-08-31); llm-stats.com (vendor-reported SWE-Bench, Sept 2026);
morphllm.com/swe-bench-pro (Scale SEAL public set, 2026-09-14);
llmbase.ai (Artificial Analysis Index, 2026-09-22).
