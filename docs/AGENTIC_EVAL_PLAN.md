# Agentic eval expansion plan — Terminal-Bench track (v5.272+)

Implementation status: Phase 1 is implemented with 50 tasks, three
difficulty tiers, exact and seeded hidden-oracle verification, a per-task
trap gate in the committed std-bench reference, and a calibration-board row
embedded in each terminal API artifact that carries confidence. Phase 3's
aggregation contract, native-artifact adapters and reproducible CLIs are
implemented. Phase 2A's pinned Terminal-Bench 2.1 read-only discovery and
capability inventory are implemented. Phase 2B.1's isolated agent-bridge core
has synthetic protocol oracles; actual container-only oracle/no-op smoke,
Harbor verifier integration and full-split execution remain pending. There is
still no upstream Terminal-Bench score.

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
| Fixed-weight agentic aggregation contract | `helioslm_v5/eval/agentic_index.py` | implemented; toy and trained-candidate runs recorded below |
| Native run-artifact to component adapters | `helioslm_v5/eval/agentic_adapters.py` | implemented; candidate run evidence remains outside the repo |
| Terminal-Bench 2.1 read-only task identity and capability inventory | `helioslm_v5/eval/tbench_adapter.py` | Phase 2A implemented; no execution or scoring |
| Isolated Harbor command/observation agent core | `helioslm_v5/eval/tbench_harbor.py` | Phase 2B.1 synthetic contract only; no container smoke, verifier run or score |

Remaining gaps against Terminal-Bench-style evaluation: no completed
container-only Harbor oracle/no-op smoke or full-split execution result; no
multi-step tasks requiring package installs or service orchestration; no
contamination control against upstream task text.

## Phase 1 — scale the in-house miniature (implemented, no new deps)

1. Grow `term_bench.py` from 10 to ~50 tasks across families: log
   surgery, config repair, data pipeline (csv/json/jsonl), git
   archaeology on a fixture repo, cron/systemd-unit authoring (static
   check only), multi-file refactor with a failing-test oracle.
2. Add difficulty tiers: tasks where the verifier checks an exact final
   state AND tasks where it checks invariants over a distribution
   (seeded), so the bench measures robustness not memorized command
   sequences.
3. Wire results into `calibration_board.py` so verbalized confidence vs
   outcome is reported per family (we already record the confidence;
   the board join is the missing piece).

Acceptance: 50 tasks, every verifier covered by a unit test that a
no-op run fails and the scripted solution passes; runtime < 10 min CPU
for the full set; board row emitted per run.

## Phase 2 — upstream Terminal-Bench adapter (container only)

The 2026-11-25 feasibility audit is in
`docs/TBENCH_PHASE2_FEASIBILITY.md`. It pins Terminal-Bench 2.1 at source
commit `12b7565679ce5006230b820666353673a133db93` and statically classifies
all 89 tasks. No upstream code was executed and no score was produced.
Terminal-Bench 2.1 revises 28 of the 89 tasks in 2.0, so versions, scores and
task-set hashes must remain separate.

1. Build the Harbor-native execution bridge as a later container-only phase.
   Preserve the task image/build definition, resources, network declaration,
   timeouts, `tests/test.sh` lifecycle and raw Harbor reward. Do not replace
   these with `term_bench.task["verify"](root)` or a pytest-only shortcut.
2. Phase 2A read-only `list`/inventory is implemented in
   `helioslm_v5/eval/tbench_adapter.py`. It requires the source commit,
   canonical task-set SHA-256 and (for the pinned split) Git tasks-tree ID;
   an identity mismatch is a hard failure. The only CLI subcommand is `list`.
   It classifies capabilities without importing Harbor or executing task
   content, treats symlinks as data rather than following them, and prints
   every skipped task ID with stable machine-readable reasons. Its JSON
   report has schema `tbench-inventory/1` and contains no score or reward.
3. Phase 2B.1 implements the agent-side core in
   `helioslm_v5/eval/tbench_harbor.py`, audited against Harbor source commit
   `90f2ee3c18d394eb0da0070415c6297eb13b87f8`. It requires an explicit
   disposable-runtime/no-host-exposure attestation and a native
   `eligible_static` inventory row. Commands go only to `environment.exec`
   without replacing Harbor's cwd, environment, user or timeout policy;
   episodes retain task-set identity and populate `AgentContext.metadata`.
   Harbor remains lazy-imported. Synthetic fake-environment tests cover DONE,
   invalid replies, budget exhaustion, isolation rejections and observation
   clipping. They do not invoke a verifier and produce no reward or score.
4. Execute only inside an existing disposable container/pod with an isolated
   Harbor runtime. Start with container-only oracle/no-op smoke controls,
   then the pinned full split when that runtime is available. Never expose
   host credentials, paths or the host Docker socket to a task container.
   The Phase 2B.1 synthetic oracles are not this runtime smoke.
5. The former automatic docker-free acceptance is withdrawn: all 89 pinned
   tasks declare prebuilt images and Dockerfiles, allow Internet access and
   install verifier dependencies, so the conservative static no-docker gate
   auto-admits 0/89. Any future manually audited no-docker exception is a
   separate custom subset, not a native or full-split result.
6. Decontaminate task instructions against training mixtures before a
   candidate rollout; record contaminated IDs explicitly. Run artifacts must
   retain source/task-set hashes, Harbor and runtime versions, resolved image
   digests, per-task rewards/failures and all subset/skip identities.

Acceptance: read-only discovery covers the complete pinned set with
synthetic-fixture tests; native execution preserves Harbor semantics; a full
2.1 report accounts for all 89 tasks. Any subset is labeled with its exact
version, runtime and task IDs. In particular, a no-docker report is labeled
"Terminal-Bench (subset, no-docker)" and is never mixed with native or
full-split vendor numbers. Static parsing or partial runs produce no score.

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
and non-writing byte-for-byte `--check`. No static fixture index is committed;
the toy and trained-candidate runs described below remain run evidence, not a
schema fixture relabeled as a publishable model index.

**Phase 3 pipeline smoke (2026-11-23, done):** the toy checkpoint
(`checkpoints/toy_v5.13.pt`) was run through all five benchmarks by
`examples/agentic_candidate_run.py` to exercise the measured pipeline end
to end; its outputs are kept out of this repository (`results/agentic/`
is git-ignored). The run also exposed and fixed one contract drift: the
long-horizon adapter now accepts `evaluate_policy`'s real int `violations`
counts (previously only the list-form fixture rows had ever been scored).

**Phase 3 trained candidate (2026-11-23, done):** the mid SFT checkpoint
(`mid_sft_v5.33`, 360M params, BPE, authorized for this run) was scored on
the same five tracks via the runner's `--candidate mid` wrapper
(`examples/mid_candidate.py`). Agentic Index **0.000000** — equal to the
toy floor, but for a different reason: the mid model's SFT distribution is
the chat tool protocol, so all five harness protocols are
out-of-distribution (bash prompts EOS after a token or two, policy prompts
emit unparseable `@@tool@@`-flavoured text, repair and Verilog outputs are
off-protocol). Driven **in-distribution**, the same checkpoint scores
9/12 correct + 3/12 safe abstentions on grounded chat (GroundingGate) vs
0/12 ungrounded (it confabulates tool arguments) — the sharpest evidence
yet that the failure mode to fix is unverified self-trust, not incapacity.
Full numbers, controls and roadmap implications:
`docs/AGENTIC_MID_CANDIDATE.md`.
**Remaining Phase 3 work:** produce candidates SFT-trained on the harness
protocols themselves (term_bench bash, long-horizon single-call policy,
tool-recovery repair format, Verilog spec-to-RTL) to lift the trained
index above the toy floor; keep the grounded-chat eval as a standing
regression gate.

## Explicitly out of scope

- OSWorld-style GUI control (needs a display stack the pod lacks).
- Buying into vendor-reported Terminal-Bench numbers as comparators;
  only same-harness, same-subset runs are compared.
