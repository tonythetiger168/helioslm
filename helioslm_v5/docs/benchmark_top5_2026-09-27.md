# HeliosLM vs Top-5 LLMs — benchmark position paper (2026-09-27)

Purpose: an honest, sourced benchmark positioning. Not a capability claim.
Method: verified public leaderboard data (2026-08-26/08-31 pulls) +
HeliosLM's own published artifacts. Everything unmeasurable is marked
"not comparable" rather than estimated.

## 1. The top 5 right now (independent task benchmarks)

Source: vals.ai SWE-Bench Verified (86 models, updated 2026-08-26) with
cost/latency from the same runs; prices ofox.ai catalog 2026-08-31.
Secondary: Artificial Analysis Intelligence Index (2026-08-31).

| Rank | Model | SWE-Bench V | AA Index | $/M in/out | Latency/task |
|---|---|---|---|---|---|
| 1 | Claude Opus 5 | 97.00% | 63.1 (max) | $5.00 / $25.00 | 577s |
| 2 | DeepSeek V4 Pro 0813 | 96.40% | — | $1.32 / $3.96 | 240s |
| 3 | GPT-5.6 Sol | 96.20% | 60.9 (max) | $2.50 / $15.00 | 182s |
| 4 | Grok 4.6 | 95.60% | 60.9 (high) | $2.00 / $6.00 | 604s |
| 5 | GPT-5.6 Terra | 95.40% | 56.6 (max) | $2.00 / $12.00 | 180s |
| (6 tie) | GLM-5.3 (open) | 95.40% | 59.5 (max) | $1.26 / $3.96 | 841s |

Human-preference ranking (Arena text Elo, 2026-08-31) orders differently:
Claude Fable 5 (1507) > Opus 4.6-thinking (1505) > Opus 4.7-thinking (1502)
> Muse Spark 1.2 (1498) > Opus 4.6 (1497). Task benchmarks and chat Elo
are different sports — Opus 5 is #1 on tasks and #7 on Elo.

Notable for this project: Kimi K3 (Moonshot) is #7-8 on the task board
(93.40% SWE-Bench) and leads the open weights on some boards; the
open-weight frontier is now DeepSeek V4 Pro > GLM-5.3 > Kimi K3.

## 2. Where HeliosLM sits (same table, no rounding up)

| | HeliosLM v5.30.2 | Frontier top-5 |
|---|---|---|
| Parameters | 8,495,760 | ~10^11-10^12 (V4 Pro: 1.6T total / 49B active) |
| SWE-Bench V | n/a (not submitted — would score ~0) | 95.4-97.0% |
| Arena Elo | n/a | 1497-1507 (top) |
| Inference cost | $0 (runs on 2 CPU threads) | $0.10-1.29 per resolved task |
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
these axes we publish numbers that no top-5 vendor publishes at all:

| Axis | HeliosLM artifact | Top-5 status |
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

On capability benchmarks HeliosLM is not in the top 5, the top 5,000,
or the top 50,000 — it is a research vehicle 5 orders of magnitude below
the frontier, and its value is that every number it publishes can be
re-derived by anyone with 2 CPU threads. The day frontier vendors publish
calibration-on-wrong-answers curves with replay-verifiable trajectories,
this project has won its argument and can retire.

Sources: vals.ai SWE-Bench Verified (2026-08-26 update, 86 models);
LMArena/Arena text Elo (2026-08-31); Artificial Analysis Intelligence
Index (2026-08-31); ofox.ai catalog (2026-08-31). Aggregated via
ofox.ai/blog/llm-leaderboard-best-ai-models-ranked-2026 and
swfte.com/ai/leaderboard (2026-09-27).
