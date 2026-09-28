# Qwen3-0.6B anchor benchmark on the HeliosLM probe suites (2026-09-28)

The first REAL cross-scale data point: a frontier-family model (Qwen3-0.6B
base, 70x HeliosLM's parameters) run through HeliosLM's own evaluators —
AgentLoop with gate DIRECT and replay verification, plus chat probes.
Runner: hand-rolled pure-torch inference (no transformers in sandbox):
manual safetensors load, byte-level BPE, RMSNorm/RoPE/GQA/SwiGLU/QK-norm,
KV cache. Deviations recorded: bf16 weights (RAM), greedy decode,
few-shot prefix (2 solved exemplars) for the base model.

## Headline

| Metric | Qwen3-0.6B | HeliosLM v5.27/30.2 (same suites) |
|---|---|---|
| Tool suite correct | **0/15** | 0/12 (three-modes dense, T19) |
| File-env (long-horizon) correct | **0/3** | n/a (suite added v5.31) |
| Chat mode-choice | **6/6** | text 11/20, tool 0/20 (v5.30.2) |
| Conf on wrong answers (tool) | mean 0.893, **max 1.000** | max 0.925-0.944 (three-modes) |
| Conf on wrong answers (file) | mean 0.969, max 0.996 | n/a |

## What the numbers say

1. **The overconfidence disease is scale-invariant.** Qwen3-0.6B answers
   wrong with mean confidence 0.893 and hits 1.000 repeatedly; HeliosLM's
   recorded finding (0.925-0.944 on 12/12 wrong) is the same pathology.
   Five orders of magnitude apart, same disease — and HeliosLM's line is
   the only one training against it (v5.32 outcome-targeted Brier, P2
   RLCD wiring pending).

2. **Mode selection is nearly free; multi-step agency is the wall.**
   Single-prompt chat probes: 6/6 correct mode choice, even correct tool
   blocks. Multi-step agentic loops: 0/15 tool, 0/3 file-env — full
   budget burned without a single finish, all while highly confident.
   The gap between "answer one question" and "act across steps" is where
   0.6B lives and where our long-horizon env + async RL roadmap points.

3. **Protocol honesty:** Qwen ran few-shot (base model) vs HeliosLM's
   fine-tuned zero-shot — an advantage to Qwen that still didn't close
   the agency gap. Confidence here is softmax max from greedy decode,
   same functional definition as HeliosLM's confidence_fn.

## Raw data

`qwen_bench_results2.jsonl` (24 records: 15 tool + 3 file + 6 chat),
append-only with per-task conf/replay flags. Replay: Qwen trajectories
pass verify_replay under DIRECT routing — the audit trail discipline
applied to a foreign model without modification.
