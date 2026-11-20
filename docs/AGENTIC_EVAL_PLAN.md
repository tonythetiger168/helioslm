# Agentic eval expansion plan — Terminal-Bench track (v5.272+)

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

Gaps against Terminal-Bench-style evaluation: only 10 miniature tasks;
no upstream Terminal-Bench task adapter; no multi-step tasks requiring
package installs or service orchestration; results not aggregated into
one agentic index; no contamination control against upstream task text.

## Phase 1 — scale the in-house miniature (no new deps)

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

## Phase 3 — one agentic index

Aggregate term_bench (scaled), long_horizon_bench, swe_bench and the
HDL benches into `results/agentic_index.json` with fixed weights
(terminal 30%, swe 30%, long-horizon 20%, tool-recovery 10%, HDL 10% —
mirroring public weighting so external comparisons stay interpretable).
Every row carries the model's checkpoint hash and the task-set hash, so
a changed task set never silently compares against an old run.

Acceptance: `python -m helioslm_v5.eval.agentic_index --results
results/...` reproduces the committed index JSON byte-for-byte for a
recorded run (same reproducibility contract as `calibration_board
--check`).

## Explicitly out of scope

- OSWorld-style GUI control (needs a display stack the pod lacks).
- Buying into vendor-reported Terminal-Bench numbers as comparators;
  only same-harness, same-subset runs are compared.
