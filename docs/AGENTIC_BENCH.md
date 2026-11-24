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
(equality of files, not just scores). A no-op candidate (empty reply
everywhere) scores 0 on all five — the oracle rejects silence; the
false-completion probe (claims DONE, no file writes) is caught by
state verification, not self-report. Aggregation is a pure function of
the five component files plus their task-set hashes, so any score
travels with its provenance.

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

## External reference row — Kimi K3.1 Pro (hosted API, 2026-11-24)

`examples/k31_reference_run.py` (Route C of `docs/K31_DISTILL_PLAN.md`)
drives the hosted Kimi K3.1 Pro chat-completions API through the *same*
five-track interfaces, with per-call `usage` cost accounting (rates are
vendor-reported assumptions: $2/1M input, $10/1M output, checked
2026-11-24), a hard `--max-cost-usd` abort (default $1.00), and the API
key read from the environment only (artifacts store a sha256
fingerprint, never the key). This row is **our measurement of an
external model**, not a vendor-reported number; it is not comparable to
Terminal-Bench and is never mixed into the Agentic Index of our own
candidates.

Full 50-episode run (`reasoning_effort=low`, 2026-11-24), **HeliosLM
same-harness measurement of an external model**, not vendor-reported:

| Track | Episodes | Passed | Pass rate |
| --- | ---: | ---: | ---: |
| Terminal | 16 | 15 | 0.9375 |
| SWE | 8 | 8 | 1.0000 |
| Long horizon | 5 | 3 | 0.6000 |
| Tool recovery | 13 | 3 | 0.2308 |
| HDL | 8 | 0 | 0.0000 |

The initial manifest recorded $0.314694. A long-horizon prompt-format
correction (`manifest_long_horizon_v3.json`) superseded its initial 0/5 row
with the 3/5 result above and added $0.042692; cumulative measured-run cost
is therefore **$0.357386**, below the $1 cap. Artifact SHA-256 values:
terminal `4431453018dadabe5b7787ca753d0b13332315652d9423f5cf9e3065bb85315a`,
SWE `a557792cee15c5b27570219dc32acfaec24543edcf3815e33f3b71a9e0ff564c`,
long horizon `9a3cdb5a2ace2b13fceb1528ba8e617271b9f2ed45d7407b53c38fb686817877`,
tool recovery `7597f34457e0988b9c2f17cea7759207cf17eb0d99feeeaec3d2c9a900f40c5c`,
HDL `7c7aca42ce339e1c46257cfda2ca10863e625c9ed52e6c628dba0727302554eb`.
Long-horizon rows carry per-episode `trajectory_sha256`; tool-recovery rows
carry `reply_sha256`. Terminal/SWE retain their command transcripts and HDL
retains generated samples, but this run did not allocate usage cost or an
individual hash to every one of those episodes; the artifact-level hashes
above are the available integrity evidence, not a claim of finer provenance.
These subset pass rates are not combined into a HeliosLM candidate Agentic
Index and are not comparable with vendor-published benchmark scores.

```bash
export MOONSHOT_API_KEY=sk-...
python3 examples/k31_reference_run.py --smoke
python3 examples/k31_reference_run.py --episodes 50 \
    --out results/reference/<date>_k31-pro/
```

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
