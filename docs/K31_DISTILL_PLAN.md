# K3.1 Pro as teacher / reference — distillation experiment plan (v5.272+)

K3.1 Pro is the current open-weight agentic reference point: its gains
over K3 came from post-training scale (large joint RL over hundreds of
domains), not architecture. Its weights are public (1.56 TB MXFP4
checkpoint on Hugging Face). This plan fixes how HeliosLM uses it:
as a **teacher for agentic trajectories** and as an **external
reference row**, not as a logit-KD teacher.

## Hard constraints (decided up front)

1. **No logit-level KD.** `helioslm_v5/src/training/logit_kd.py` moves
   teacher top-k distributions onto student tokens; it needs teacher
   tokenizer access and per-token vocab mapping. K3.1's vocab does not
   match HeliosLM tokenizers, and running a 1.56 TB checkpoint for
   logits is not pod-feasible. Sequence-level and trajectory-level
   distillation only.
2. **Teacher output is untrusted training data.** Every teacher
   trajectory passes the existing data-quality gates
   (`data_quality.py` filters, `decontam.py` against all eval sets)
   before entering any mixture.
3. **License check first.** Record the checkpoint's license in
   `docs/HDL_DATA.md`-style provenance before any generated data is
   committed; if the license forbids derivative training data, K3.1 is
   used as an eval reference only.
4. **Vendor-reported numbers stay labeled as such** — the same
   separation rule used for all external results.

## Route A — agentic trajectory distillation (primary)

Reuses the existing `distill_agent.py` pipeline (`run_tasks` against an
OpenAI-compatible endpoint):

1. Serve or rent K3.1 Pro behind an OpenAI-compatible endpoint (the
   1.56 TB checkpoint needs ~8×H200-class serving; a hosted endpoint is
   the realistic path — record cost per 1k trajectories before
   starting).
2. Task sources, in priority order:
   - `term_bench.py` Phase-1 tasks (see `docs/AGENTIC_EVAL_PLAN.md`) —
     teacher must produce the multi-turn command transcript, and the
     **sandbox verifier must pass the final state**, not just the
     teacher's claim of success;
   - `swe_bench.py` repair tasks with the repo test suite as the
     verifier;
   - H4 HDL tasks where `hdl_reward.reward` (post-audit, Gate 0 of
     `docs/H4_KL_RL_RUNBOOK.md`) is the verifier.
3. Keep only trajectories the verifier accepts (`distill_filter.py`
   contract), then SFT with `examples/sft_hf.py` on the verified
   traces; report pass-rate of the teacher itself per task family —
   that number is the ceiling the student can approach, and it is also
   the K3.1-as-reference measurement (Route C comes free here).
4. Ablate: student SFT on verified K3.1 traces vs student SFT on the
   existing CodeV-R1/H3 data, same token budget, same eval table.

Acceptance: teacher pass-rate per family reported; student trained on
verified traces beats the same-budget control on the Phase-1 term_bench
set without regressing the H3 table.

### Route A offline pipeline (implemented 2026-11-24)

`helioslm_v5.src.training.agentic_distill` and
`examples/agentic_distill.py` now cover all four improvement-3 protocols:
terminal, SWE, long-horizon and tool recovery. The task generator produces
fresh parameterized training tasks with `agentic-distill-disjoint-v1`
provenance; it does not clone `term_bench.TASKS`, `swe_bench.TASKS`,
`long_horizon_bench.SCENARIOS` or the seeded tool-recovery eval corpus.
Collection records the exact prompt/reply at every model step. Terminal and
SWE acceptance requires a DONE reply plus final-state or visible+hidden test
verification; long-horizon acceptance additionally requires a fresh-sandbox
replay, no OracleGuard violation, no injection and no parse error; tool
recovery requires strict parsing and exact equality with the original calls.

```bash
python3 examples/agentic_distill.py tasks --per-protocol 25 \
    --out runs/route-a/tasks.jsonl
# Collection executes untrusted commands: use an existing disposable container.
python3 examples/agentic_distill.py collect --tasks runs/route-a/tasks.jsonl \
    --out runs/route-a/trajectories.jsonl --container-confirmed \
    --max-cost-usd 1.0
python3 examples/agentic_distill.py export \
    --trajectories runs/route-a/trajectories.jsonl --out runs/route-a/sft.jsonl
```

Collection checks the new tasks against all four built-in formal eval sets
before the first API call. Long-horizon SFT inputs preserve the
candidate-facing `MidCandidate.policy` text verbatim; AgentLoop internals and
teacher-wrapper private API instructions are not substituted as student input.
Export requires a non-empty eval index, drops foreign/unverified trajectories
and malformed targets, applies
`data_quality.heuristic_check`, then applies the 13-gram `decontam.py` gate
to every prompt+target. The report exposes unchecked short eval items and
per-protocol survivor counts; an empty result is refused rather than written
as a placeholder. Extra `--eval` JSONL files can extend, but cannot disable,
the built-in index. API keys come from the named environment variable only;
the CLI neither rents an endpoint nor starts SFT.

The existing K3.1 reference JSON artifacts are **not training input**. They
are same-harness measurements of an external model, and the terminal/SWE
artifacts contain commands but not the original per-step prompts/replies.
Using those eval-derived traces for SFT and then reporting gains on the same
tasks would be contamination. Candidate training, its sizing/cost review and
the subsequent five-track acceptance run remain separate authorized steps.

## Route B — preference pairs for the trust/decision layer

K3.1's published hallucination-rate improvement is the behavior
HeliosLM's `trust_gate` / `grounding` stack targets. Generate paired
accept/abstain decisions on `clarify_missing_param`-style probes and
`oracle_guard` negative controls; train with the existing `dpo.py`
trainer. Smaller scope than Route A; do it only if Route A's teacher
endpoint is already running.

## Route C — external reference row (cheap, do first)

Before any training: run the K3.1 endpoint on the current agentic eval
matrix (term_bench 10, long_horizon_bench, swe_bench repair) exactly as
any model under test, and add the row to the results table labeled
"external, API, <date>". This calibrates how far the in-house models
are from the open-weight frontier on *our* harness — the number that
matters for prioritizing Route A.

## Explicitly out of scope

- Distilling K3.1's long-context behavior (1M context is an
  architecture/property gap, not a trajectory gap).
- Any use of K3.1 generations in the pretraining mixture.
- Claiming comparability with K3.1's vendor-reported benchmark scores;
  Route C produces same-harness numbers only.
