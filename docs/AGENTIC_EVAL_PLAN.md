# Agentic eval expansion plan — Terminal-Bench track (v5.272+)

Implementation status: Phase 1 is implemented with 50 tasks, three
difficulty tiers, exact and seeded hidden-oracle verification, a per-task
trap gate in the committed std-bench reference, and a calibration-board row
embedded in each terminal API artifact that carries confidence. Phase 3's
aggregation contract, native-artifact adapters and reproducible CLIs are
implemented. Phase 2A's pinned Terminal-Bench 2.1 read-only discovery and
capability inventory are implemented. Phase 2B.1's isolated agent-bridge core
and Phase 2B.2's built-in oracle/no-op smoke contracts have synthetic
protocol oracles. Phase 2C's decontamination ledger and result-integration
contracts are implemented with synthetic fixtures; actual mixture scanning,
container-only smoke execution, Harbor verifier runs and full-split execution
remain pending. There is still no upstream
Terminal-Bench score.

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
| Isolated Harbor command/observation agent core and oracle/no-op smoke gates | `helioslm_v5/eval/tbench_harbor.py` | Phase 2B.1/2B.2 synthetic contracts only; no container smoke, verifier run or score |
| Decontamination ledger and native-run result integration | `helioslm_v5/eval/tbench_integration.py` | Phase 2C synthetic contracts only; no mixture scan, model run or score |

Remaining gaps against Terminal-Bench-style evaluation: no completed
container-only Harbor oracle/no-op smoke or full-split execution result; no
real training-mixture scan against the upstream task text; no
multi-step tasks requiring package installs or service orchestration.

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
   Phase 2B.2 adds pure-data contracts for Harbor's built-in `oracle` and
   `nop` controls: serialized `TrialConfig` values must keep the pinned local
   task source, disposable environment, task resources and timeout policy;
   reject model credentials, custom agents/verifiers, mounts and Compose
   overlays; and require the upstream verifier. Serialized `TrialResult`
   evidence must be exception-free, retain Harbor's trial name and task
   checksum, have completed verifier timing and a valid raw reward
   (`oracle`=1, `nop`=0), retain the resolved image digest,
   and include hashed `result.json` plus verifier-log artifacts. These checks
   validate the control contract only and still execute nothing.
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
6. Phase 2C contracts are implemented in
   `helioslm_v5/eval/tbench_integration.py`. The decontamination ledger keeps
   every inventory task exactly once; a contaminated task is recorded as
   excluded with its match evidence and must be skipped with the stable
   `contaminated` reason — it never silently becomes a zero or leaves the
   accounting. Native run reports must account for every `eligible_static`
   row exactly once (scored or skipped with taxonomy reasons); the terminal
   component descriptor recomputes the mean reward from the scored rows and
   preserves the source/tree/task-set identity, Harbor identity, resolved
   image digests and hashed raw artifacts. Identity mismatches between the
   inventory, decontamination and run reports are hard failures. Synthetic
   fixtures only: no mixture scan, model rollout, container, verifier or
   score was executed.

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

## Phase 3.5 — harness-protocol candidate SFT (cost-tiered plan)

No base model: the HeliosLM pipeline is self-trained end to end, and this
track stays on that line. The failure mode measured above is protocol
out-of-distribution, not incapacity, so the next candidate is SFT on
machine-generated harness-protocol data. Spending is tiered; every tier is
a hard stop that needs explicit authorization before the next one starts.
GPU hours are quoted for an A100 80GB (US$1.1–2.75/hr) with an RTX 4090
(US$0.7–0.9/hr) as the budget alternative.

| Tier | Cap | Content | Output | Gate to proceed |
|---|---|---|---|---|
| T0 | US$0 (CPU sandbox) | Generate the five-track protocol dataset (~100k rows: term_bench bash, long-horizon policy, tool-recovery repair, Verilog spec-to-RTL, grounded-chat retention), manifest SHA-256, format/decontamination validation in CI | training-ready `harness_sft` dataset | validation suite green and merged |
| T1 | ~US$5 | 5% subset smoke run (1 epoch): loss curve, no NaN, checkpoint save/load, eval hook | smoke checkpoint (never scored) | loss decreases monotonically |
| T2 | ~US$1–2 expected, US$25 hard cap | full 1 epoch (~176k rows, ~27M tokens at the T1-observed average) = ~5,560 steps at 32 rows/step | first testable candidate | ≥2 tracks non-zero; otherwise fix data, not budget |
| T3 | ~US$30 | 3 epochs + full Agentic Index + GroundingGate regression check | first **measured** trained-candidate index | index clearly above toy floor (≥3 tracks non-zero), GroundingGate holds 9/12 + 3 abstentions ±1, parse rate ≥95% |
| T4 | US$50 hard cap | one full re-run after infrastructure failure only | — | stop and report at the cap; no auto-renewal |
| T5+ | by proposal | mixture/lr ablations, 1B/1–2B-token Stage 2 pilot (US$100–300), Stage 2 proper (US$300–3,000), 7B from-scratch (US$15k–45k), frontier scale (US$0.5M+) | — | each tier is separately proposed and authorized |

**T0 builder implemented (synthetic/generator data only, 2026-11-26).**
`helioslm_v5/eval/harness_sft_data.py` emits the five tracks and verifies
each against its own oracle: term_bench rows are prompt/reply pairs from
replaying every task's scripted solution in a real sandbox (the replay must
pass the task verifier), long-horizon rows are correct-policy trajectories
re-executed through the real `AgentLoop` (scripted scenarios plus
parameterised variants; injection scenarios teach refusal-by-ignoring),
tool-recovery rows are corrupt-then-repair pairs whose target round-trips
exactly under `recover_tool_call`, the HDL track is a hand-written plus
parameterised generator suite in the pinned VerilogEval prompt envelope
(Icarus-gated when the simulator is installed; no upstream text or code),
and grounded-chat rows retain the answer-or-abstain behaviour. The v1 build
wrote 99,134 rows (`term_bench` 102, `long_horizon` 90,022,
`tool_recovery` 8,000, `hdl` 1,004 with all 1,004 Icarus-verified,
`grounded_chat` 6); manifest SHA-256
`2dbceb152ab339140987313f74c4149f733a38bc93d45310c8ccd2351e4d1410` covers the
written bytes. No model, network, upstream benchmark text or teacher API is
involved; the dataset itself lands outside the repo (results are
git-ignored). T1 (smoke run) and above still need explicit authorization
and a GPU budget.

**T1 smoke completed (2026-11-27, ~US$0.23 total, pod deleted).** Two
rounds on an RTX 4090 against `mid_sft_v5.33`: r1 trained on harness rows
only and FAILED the grounded-chat regression gate (9/12 → 6/12 with three
wrong answers); r2 added 3,200 mid original-SFT replay rows (40%) and
PASSED (loss 7.86 → 0.74 over 200 steps, checkpoint reload finite,
grounded 9/12 + 3 abstained before and after — no regression). Format
smoke on r2: long_horizon 4/4, tool_recovery 3/4, grounded_chat 3/4,
**hdl 1/4, term_bench 0/4**. The replay and eval sets share one toy
generator (different seeds), so the 9/12 proves *no regression*, not
capability. Lessons now hard requirements: replay ≥40% of any mixture,
grounded regression is a hard gate, and the hdl/term_bench format tracks
needed reinforcement.

**T2 weak-track format drills (synthetic/generator data only,
2026-11-27).** `harness_sft_data.py` gains two opt-in builders behind
`build_dataset(n_format=...)` (default 0, so the T0 byte stream is
unchanged): `gen_term_bench_format` emits zero-shot drills — step-1
prompts rendered by the real `term_bench.render_prompt` paired with the
first scripted command, plus verifier-passed DONE envelopes (the T0
replays always embed commands in multi-step history, which T1 showed is
not enough to elicit the zero-shot protocol); `gen_hdl_format` re-pairs
Icarus-verified suite specs with strict-envelope replies ([BEGIN]-only /
code-only / no-fence / bare-module variants, never a Markdown fence).
Every drill target re-parses under its harness parser. T2 mixture
(~176k rows): T0 full 99,134 + 1,024 format drills (`n_format=512`) +
~76k replay (43%, hard floor 40%), warm start from the T1 r2 checkpoint
(sha256 `d6f7444b…` in the private HF dataset). **Step arithmetic
(corrected 2026-11-27):** at micro 8 × accum 4 = 32 rows/step, one
epoch over ~176k rows is **~5,560 steps** — an earlier 20k-step draft
figure was 3.6 epochs and violated the one-epoch limit. At the T1 r2
measured rate (200 steps in 91.3 s on an RTX 4090), one epoch trains in
~45–90 min including checkpoint IO and eval, ~US$1–2 at US$0.89/h; the
US$25 cap stands as rerun buffer. Execution order: step 0 (free, CPU)
weak-track failure classification on the r2 checkpoint, then a single
one-epoch run. T2 pod run still requires explicit approval before any
spend.

Data boundaries: all T0 rows are machine-generated from in-repo harness
oracles — no upstream Terminal-Bench text, no teacher API, no external
corpus. Decontamination against upstream task text still runs through the
Phase 2C ledger rules before any candidate is scored on the pinned split.
Checkpoints are fingerprinted by SHA-256; API keys, if a later authorized
tier needs them, come only from environment variables.

## Explicitly out of scope

- OSWorld-style GUI control (needs a display stack the pod lacks).
- Buying into vendor-reported Terminal-Bench numbers as comparators;
  only same-harness, same-subset runs are compared.
