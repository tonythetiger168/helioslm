# Agentic eval expansion plan — Terminal-Bench track (v5.272+)

Implementation status: Phase 1 is implemented with 50 tasks, three
difficulty tiers, exact and seeded hidden-oracle verification, a per-task
trap gate in the committed std-bench reference, and a calibration-board row
embedded in each terminal API artifact that carries confidence. Phase 3's
aggregation contract, native-artifact adapters and reproducible CLIs are
implemented; Phase 2 remains a plan. There is still no five-component
HeliosLM model baseline and no upstream Terminal-Bench score.

Why: 2026 agentic leaderboards weight long-horizon terminal/repository
work highest (Artificial Analysis puts agentic coding at 30% of its
intelligence index), and the strongest open-weight agentic models
(K3.1 Pro et al.) are differentiated there, not on knowledge tasks.
HeliosLM already has the harness pieces; this plan scales them into a
Terminal-Bench-comparable track.

## What exists today

| Piece | File | Status |
|---|---|---|
| Terminal-style sandbox tasks (10 original tasks, state-verified, verbalized confidence) | `helioslm_v5/eval/term_bench.py` (v5.56) | working; exempt from the daily gate (binds 127.0.0.1) |
| Long-horizon multi-step file tasks through the real AgentLoop, with injection/self-grading negative controls | `helioslm_v5/eval/long_horizon_bench.py` (#174) | working, deterministic CPU |
| SWE-style RL env + repair bench | `helioslm_v5/eval/swe_bench.py`, `examples/swe_agent_rl_train.py` | working |
| Tool-call parse recovery audit | `helioslm_v5/agent/tool_recovery.py` | working |
| Verbalized-confidence capture | `DONE confidence=` in term_bench; `calibration_board.py` | working |
| Fixed-weight agentic aggregation contract | `helioslm_v5/eval/agentic_index.py` | implemented; no model baseline yet |
| Native run-artifact to component adapters | `helioslm_v5/eval/agentic_adapters.py` | implemented; no candidate run recorded yet |

Remaining gaps against Terminal-Bench-style evaluation: no upstream
Terminal-Bench task adapter; no multi-step tasks requiring package installs
or service orchestration; no measured five-component run from one candidate
checkpoint; no contamination control against upstream task text.

## Phase 1 — scale the in-house miniature (implemented, no new deps)

1. Grow `term_bench.py` from 10 to ~50 tasks across families: log
   surgery, config repair, data pipeline (csv/json/jsonl), git
   archaeology on a fixture repo, cron/systemd-unit authoring (static
   check only), multi-file refactor with a failing-test oracle.
2. Add difficulty tiers: tasks where the verifier checks an exact final
   state AND tasks where it checks invariants over a distribution
   (seeded), so the bench measures robustness, not memorized command
   sequences.
3. Wire results into `calibration_board.py` so verbalized confidence vs
   outcome is reported per family (we already record the confidence;
   the board join is the missing piece).

Acceptance: 50 tasks, every verifier covered by a unit test that a
no-op run fails and the scripted solution passes; runtime < 10 min CPU
for the full set; board row emitted per run.

## Phase 2 — upstream Terminal-Bench adapter (pod only)

1. Adapter `helioslm_v5/eval/tbench_adapter.py`: read upstream task
   dirs (task.toml + tests), translate into the term_bench sandbox
   protocol, run the upstream pytest-style verifier after DONE.
2. Upstream tasks assume docker; the pod has none. Run with the same
   discipline as CVDP: only tasks whose harness needs the stock image,
   executed directly with paths pointed at a temp dir; `list` prints
   skipped ids with reasons.
3. Decontamination: hash task instruction text against training
   mixtures (`decontam.py`) before reporting any number; contaminated
   tasks are excluded and counted.

Acceptance: adapter runs the docker-free subset end-to-end on the pod;
report is labeled "Terminal-Bench (subset, no-docker)" and never mixed
with full-split vendor numbers.

## Phase 3 — one agentic index (aggregation and adapters implemented)

`helioslm_v5/eval/agentic_index.py` aggregates term_bench (scaled),
long_horizon_bench, swe_bench, tool recovery and the HDL benches into
`results/agentic_index.json` with fixed weights: terminal 30%, SWE 30%,
long-horizon 20%, tool-recovery 10% and HDL 10%. Every component carries the
model name, checkpoint SHA-256, task-set name and SHA-256, sample count and a
source run artifact with its SHA-256. A changed task set therefore cannot
silently compare against an old run, and components from different models or
checkpoints cannot be mixed.

A results directory contains these five component descriptors:
`terminal.json`, `swe.json`, `long_horizon.json`, `tool_recovery.json` and
`hdl.json`. Source artifacts are addressed by normalized paths relative to
that directory. For example:

```json
{
  "schema": "agentic-component/1",
  "component": "terminal",
  "model": "helioslm-candidate",
  "checkpoint_sha256": "<64 lowercase hex characters>",
  "task_set": "term-bench-50-v1",
  "task_set_sha256": "<64 lowercase hex characters>",
  "score": 0.5,
  "n": 50,
  "evidence_type": "model_run",
  "verification_tier": "measured",
  "source": {
    "artifact": "artifacts/terminal.json",
    "sha256": "<64 lowercase hex characters>"
  }
}
```

`helioslm_v5/eval/agentic_adapters.py` converts native benchmark output into
those descriptors. It recomputes every score from the source rows instead of
trusting a summary field. For example, after an actual terminal candidate run:

```bash
python -m helioslm_v5.eval.agentic_adapters \
  --results results/<run> --component terminal \
  --artifact artifacts/terminal.json \
  --model helioslm-candidate \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --task-set term-bench-50-v1 \
  --task-set-sha256 "$TERM_TASK_SET_SHA256" \
  --evidence-type model_run
python -m helioslm_v5.eval.agentic_adapters \
  --results results/<run> --component terminal \
  --artifact artifacts/terminal.json \
  --model helioslm-candidate \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --task-set term-bench-50-v1 \
  --task-set-sha256 "$TERM_TASK_SET_SHA256" \
  --evidence-type model_run --check
```

The accepted source shapes are the row lists returned by `run_term_bench()` and
`run_swe_bench()`, the dictionary returned by `evaluate_policy()`, the
`hdl-bench/1` dictionary returned by `hdl_bench.score()`, and a strict
`tool-recovery-run/1` object with `rows: [{task, correct}, ...]`. Terminal/SWE
use mean per-task pass@1; long-horizon uses scenario success rate; tool
recovery uses exact-recovery rate; HDL uses pass@1. Component `n` counts all
scored episodes or generated HDL samples. Duplicate task IDs, non-boolean
outcomes, inconsistent counts and summary/row mismatches are hard failures.
The long-horizon `run_suite()` shape and tool-recovery test-report shape are
rejected outright. Adapter `--check` is byte-for-byte and non-writing, just
like index `--check`; `schema_fixture` remains available only for isolated
schema tests.

Generate and later verify a recorded index with:

```bash
python -m helioslm_v5.eval.agentic_index \
  --results results/<run> --output results/<run>/agentic_index.json
python -m helioslm_v5.eval.agentic_index \
  --results results/<run> --output results/<run>/agentic_index.json --check
```

`--check` compares the complete serialization byte-for-byte and never rewrites
a stale output. Missing, duplicate or unknown components; malformed hashes;
NaN, infinite, boolean or out-of-range scores; mixed checkpoint/model/evidence
identities; and source hash mismatches are hard failures, not zero scores.

**Evidence boundary:** a publishable model index requires all five components
to use `model_run` + `measured`. Scripted std-bench references, the
long-horizon `run_suite()` oracle, tool-recovery unit-test reports and HDL
`self_check()` results validate harnesses; they are not candidate model runs
and must not be relabeled to manufacture an index. A partial index is not a
lower or provisional total.

The daily-gate `case_agentic_index.py` builds an isolated `schema_fixture` +
`schema-fixture` run and requires `is_model_score: false`. Its 0.625 is only
the hand-computed math oracle (0.15 + 0.225 + 0.12 + 0.09 + 0.04), not a
HeliosLM baseline. Four tests pin the weighting math, deterministic order,
canonical component hash, strict validation, source tampering, CLI generation
and non-writing byte-for-byte `--check`. No static fixture index is committed
because there is no recorded five-component candidate run to publish yet.

**Phase 3 first measured run (2026-11-23, done):** the toy checkpoint
(`checkpoints/toy_v5.13.pt`) was run through all five benchmarks by
`examples/agentic_candidate_run.py`; artifacts, `model_run`/`measured`
descriptors and the aggregate index (**0.000000**, `is_model_score: true`)
are committed under `results/agentic/2026-11-23_toy-v5.13/` — see
`docs/AGENTIC_BENCH.md` for the numbers. The index verifies byte-for-byte
with `--check`. The run also exposed and fixed one contract drift: the
long-horizon adapter now accepts `evaluate_policy`'s real int `violations`
counts (previously only the list-form fixture rows had ever been scored).
**Remaining Phase 3 work:** score a trained candidate on the same five
tracks (needs an authorized checkpoint) to lift the index above the toy
floor.

## Explicitly out of scope

- OSWorld-style GUI control (needs a display stack the pod lacks).
- Buying into vendor-reported Terminal-Bench numbers as comparators;
  only same-harness, same-subset runs are compared.
