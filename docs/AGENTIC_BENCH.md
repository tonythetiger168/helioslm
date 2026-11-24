# AGENTIC_BENCH — the toy-scale, locally reproducible agentic evaluation

Status: measured inventory, 2026-11-23. Review-plan item B2-3. Every
number below is produced by `docs/agentic_bench_probe.py` on this
checkout; the only externally sourced figure is K3's Terminal-Bench
score, quoted for positioning and never mixed into our numbers.

## Why a toy-scale agentic bench

Frontier agentic scores (K3's 85.4 Terminal-Bench, MCP-Atlas rows) come
from cloud-scale models on harnesses nobody else can run. They answer
"how strong is the model" but not "can I trust the number". HeliosLM's
answer is the other cell of the table: benchmarks small enough to run
on a laptop CPU, with oracle-verified outcomes, provenance-hashed
artifacts, and every failure class enumerated. A score you cannot
reproduce is marketing; a score you can reproduce is engineering.

## The evaluation surface (all local, all deterministic)

| Track | Module | Surface |
| --- | --- | --- |
| Terminal-style tasks | `eval/term_bench.py` | **50 tasks, 3 tiers (18 easy / 19 medium / 13 hard)**, state-verified, verbalized `DONE confidence=` capture feeding the calibration board |
| Long-horizon file tasks | `eval/long_horizon_bench.py` | multi-step tasks through the real AgentLoop with injection and self-grading negative controls |
| SWE-style repair | `eval/swe_bench.py` | repair bench + RL env |
| Tool-call recovery | `agent/tool_recovery.py` | strict parser + a fixed, ordered pipeline of 7 named repairs |
| Aggregation | `eval/agentic_index.py` | fixed weights `{terminal .30, swe .30, long_horizon .20, tool_recovery .10, hdl .10}`, provenance-hashed components |

## Measured: the tool-call error taxonomy

Models in the wild emit near-miss tool calls. The strict parser
(`schema.parse_tool_call`) stays strict; the recovery layer applies a
FIXED list of named repairs, in order, and records every repair in the
trajectory. Measured battery (40 random samples per class, seed 1):

**Repairable — 7 classes, 40/40 recovered each** (strict parser
correctly rejects first, recovery restores the exact calls):

| Class | Repair applied |
| --- | --- |
| fence_and_prose (block in a code fence with prose around it) | `extract_block` |
| missing_close_marker | `close_marker` |
| inner_whitespace | `strip_inner_ws` |
| trailing_commas | `trailing_commas` |
| openai_arguments_string (`arguments` as a JSON string) | `args_json_string` |
| bare_call_list (no `calls` wrapper) | `wrap_calls` |
| fenced_json_no_markers | `extract_block` |

**Never repaired — 6 classes, all correctly refused** (`ToolCallError`,
no silent salvage): `unknown_tool`, `wrong_arg_type`,
`truncated_json`, `two_blocks` (ambiguous intent), `prose_only`,
`deep_nesting`. The refusal set is the safety half of the taxonomy: a
recovery layer that "fixes" an unknown tool name is a security hole,
not a feature.

## Measured: aggregation contract

On the committed schema fixtures (explicitly `is_model_score: false` —
a math/provenance check, not a model score): components 0.75/0.50/0.50/
0.75/0.50 across the five tracks aggregate to **0.625** under the fixed
weights, byte-identical under reversed input order and CLI regeneration
(`--check`).

## Candidate runs (pipeline smoke, 2026-11-23)

`examples/agentic_candidate_run.py` runs one checkpoint through all five
components and writes native artifacts, `model_run`/`measured`
descriptors and the aggregate index into a results directory. The first
run used the committed toy checkpoint `checkpoints/toy_v5.13.pt` as a
pipeline smoke test only; its outputs are kept out of this repository
(`results/agentic/` is git-ignored). Any results directory verifies
byte-for-byte:

```bash
python3 -m helioslm_v5.eval.agentic_index \
  --results results/agentic/<run> \
  --output results/agentic/<run>/agentic_index.json --check
```

Re-running a component regenerates its artifact byte-for-byte (greedy,
fixed seeds; verified for tool_recovery). One contract fix landed with
this run: the long-horizon adapter now accepts `evaluate_policy`'s real
int `violations` counts alongside the fixture's list form (both score
identically; the drift surfaced only when the first real artifact was
scored).

A second run (2026-11-23) scored the trained mid SFT checkpoint
(`--candidate mid`, `mid_sft_v5.33`, 360M params): Agentic Index
**0.000000** — the five harness protocols are out-of-distribution for a
chat-tool-SFT model, while the same checkpoint scores 9/12 + 3 safe
abstentions on grounded chat in-distribution (0/12 ungrounded). Full
report: `docs/AGENTIC_MID_CANDIDATE.md`.

## Reproduce

```bash
python3 docs/agentic_bench_probe.py   # CPU, seconds; prints measured JSON
```

## Positioning (external numbers, different harness — not comparable)

K3's 85.4 Terminal-Bench is a cloud-model score on an upstream harness;
nothing on this page is claimed comparable to it. What we claim instead:
every number here regenerates on your machine in seconds, every task
has an oracle that a no-op run fails, and the aggregation contract is
hash-pinned. When a measured HeliosLM five-component run lands, it will
be reported in this document's own units, clearly labeled, never
converted into a Terminal-Bench-equivalent score.
