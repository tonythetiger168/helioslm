# HeliosLM Milestones

Single tracker for every HeliosLM plan, oldest first:

1. [Phase 0–3](#phase-03--local-decision-engine-plan-2026-10-06) of the
   Local Decision Engine plan ([`PHASE0_3_PLAN.md`](PHASE0_3_PLAN.md)) —
   complete;
2. the [2026-10-13 landscape-review milestones](#2026-10-13-landscape-review-milestones)
   M1–M6;
3. the [2026 Q4 roadmap phases](#roadmap-2026-q4-phases)
   ([`ROADMAP_2026Q4.md`](ROADMAP_2026Q4.md)) — in progress.

Update this file in the same PR that lands or changes a milestone.

> Not tracked here: the "M1–M11" IDs in `docs/helioslm_v5_code_review.md`
> and `docs/V5.1_FIXES_REPORT.md` (v5.1 code-review findings), and the
> "Phase 0–2" commits of the rev-skills integration plan (`ddf1851`,
> `41ef523`: skill template, validator, model-security skills).

## Phase 0–3 — Local Decision Engine plan (2026-10-06)

All four phases are complete. Commit hashes cited in the plan predate
this repository's history; the CHANGELOG entries and tests below are the
surviving evidence. Scope notes are in [`PHASE0_3_PLAN.md`](PHASE0_3_PLAN.md).

| Phase | Item | Status | Landed in | Tests |
|---|---|---|---|---|
| 0 | Merge the diverged lines, version v5.41, all tests green | ✅ Done | v5.41 | full suite |
| 1.1 | DecisionHead module (`src/decision_head.py`) | ✅ Done | v5.41 | `test_lm_decision_head.py` |
| 1.2 | DecisionHead unit tests | ✅ Done | v5.41 | `test_lm_decision_head.py` (5) |
| 1.3 | DecisionHead wired onto real HeliosLMv5 hidden states | ✅ Done | v5.42 | `test_decision_head_integration.py` (3) |
| 1.4 | DPO trainer (`src/training/dpo.py`) | ✅ Done | v5.41 | `test_dpo.py` |
| 2.1 | Multi-env registry: math / code / alignment-audit (`src/training/envs.py`) | ✅ Done (end-to-end loop v5.43) | v5.42 | `test_envs.py`, `case_trainers.py` |
| 2.2 | Long-context passkey eval (`eval/longctx.py`) | ✅ Done — tested at 128–512 tokens; no 32K–256K result recorded | v5.42 | `test_longctx.py`, `test_longctx_harness.py` |
| 2.3 | Calibration × quantization cross (`eval/quant_calib.py`) | ✅ Done | v5.42 | `test_quant_calib.py` |
| 3.1 | Prefix-pool accounting ledger (`src/inference/pool_ledger.py`) | ✅ Done | v5.42 | `test_pool_ledger.py` |
| 3.2 | TrustGate v2, calibrated abstention (`agent/trust_gate_v2.py`) | ✅ Done | v5.42 | `test_trust_gate_v2.py` |
| 3.3 | Mini-benchmark harness (`eval/mini_bench.py`) | ✅ Done | v5.42 | `test_mini_bench.py` |
| 3.4 | README demo GIF (`docs/demo.gif`) | ✅ Done | v5.42 | — |
| 3.5 | Launch post (`docs/launch_post_reddit.md`) | ✅ Drafted — publication not recorded | v5.42 | — |

## 2026-10-13 landscape-review milestones

| ID | Milestone | Status | Landed in | Tests |
|---|---|---|---|---|
| M1 | Self-grading invariant: OracleGuard keeps graders / answer keys out of tool reach | ✅ Done | v5.51 (#17) | T82 |
| M2 | Artifact verification tiers: every benchmark JSON carries a registered tier, enforced in CI | ✅ Done | v5.51 (#17) | T83 |
| M3 | TrustGate CLARIFY three-state routing for calls with missing parameters | ✅ Done | v5.53 (#19), hardened in the 2026-10-15 review fixes | T86, T87 |
| M5 | API probe refreshed to the 2026-10 model generation, with cost accounting | ✅ Done | v5.51 (#17) | T84 |
| M6 | Cost-per-completed-task baseline | ✅ Done | v5.52 (#18) | T85 |

M4 was dropped from the plan; the remaining IDs are kept as-is, not
renumbered.

## Roadmap 2026 Q4 phases

| Phase | Item | Status |
|---|---|---|
| A | Housekeeping: v5.52/v5.53 CHANGELOG entries, this milestone file | ✅ Done (v5.54) |
| A | Calibration leaderboard over committed probe artifacts | ✅ Done (v5.54, T88) |
| A | Calibration leaderboard: new frontier API rows (opus-5.5, sonnet-5.5, astra, sol, argon) | ⏳ Needs an API key run of `examples/probe_api.py` |
| A | Calibration leaderboard on RLCDAlignBench (7,193 items) | ⏳ Dataset (`all.jsonl`) is not in the repo |
| A | Standard eval harness: GSM8K / HumanEval (pinned subsets, sandboxed execution, reference gates) | ✅ Done (v5.55, T89) |
| A | Standard eval harness: first model rows (frontier APIs, local mid model) | ⏳ Needs an API key; the mid checkpoint is not in the repo |
| A | Standard eval harness: terminal-style sandbox tasks (10 tasks, verbalized confidence) | ✅ Done (v5.56, T90) |
| B | `k3_block` preset: K3 layer order, Gated MLA, per-channel decay, attention residuals, LatentMoE | ✅ Done (v5.57, T91) |
| B | `v4_csa`: CSA in compress-then-select order + CSA/HCA stack preset with IndexShare, passkey A/B | ✅ Done (v5.58, T92) |
| B | `v4_csa`: learned compressor — 4× passkey recall 0.13 → 0.67; 16× does not learn | ✅ Done (v5.59, T93) |
| B | `v4_csa`: compressor round 2 + oracle ceiling — 16× 0.13 → 0.39, 4× mixed; perfect pooling would reach 0.86–0.90 (dense 0.91) | ✅ Done (v5.60, T94) |
| B | `v4_csa`: compressor round 3 — 4× 0.825/0.800, about 92% of the ceiling | ✅ Done (v5.61, T95) |
| B | `v4_csa`: 16× compressor (0.375/0.390 vs a 0.86/0.845 ceiling; more training does not help) | ⬜ Not started |
| B | Stable LatentMoE study | ⬜ Not started |
| B2 | B2-1 per-layer sliding-window / global interleave + `mimo_hybrid_config()` (MiMo) — 5:1 hybrid 0.424 vs dense 0.428 at 39% of the keys | ✅ Done (v5.62, T96) |
| B2 | B2-2 learnable attention sink bias (MiMo) — no effect at this scale (0.424 vs 0.424) | ✅ Done (v5.62, T96) |
| B2 | B2-3 `moe.first_k_dense` (GLM-5.3, MiMo) | ⬜ Not started |
| B2 | B2-4 learnable mHC with Sinkhorn projection (GLM-5.3-Flash) | ⬜ Not started |
| B2 | B2-5 MiMo-style light MTP head | ⬜ Not started |
| B2 | B2-6 on-policy distillation trainer (MiMo MOPD; confirm the definition first) | ⬜ Not started |
| B3 | Q1 short conv in GDA linear-attention layers (Qwen3-Next) | ⬜ Not started |
| B3 | Q2 zero-centered RMSNorm with weight decay (Qwen3-Next) | ⬜ Not started |
| B3 | Q3 global-batch load-balancing loss (Qwen3 MoE) | ⬜ Not started |
| B3 | Q4 `qwen_next_config()` preset | ⬜ Not started |
| B3 | Q5 (optional) MTP speculative-decoding acceptance measurement | ⬜ Not started |
| AC | AC1 mini SWE task set (20 small repos, hidden tests, trap gates) | ✅ Done (v5.62, T97) |
| AC | AC2 agent RL loop (GRPO + sandbox, test-pass reward) | 🔄 In progress |
| AC | AC3 tool-call recovery layer (7 named repairs, fuzz-tested) | ✅ Done (v5.62, T99) |
| AC | AC4 repo-scale long-context probe (dense vs hybrid vs CSA) | ⬜ Not started |
| AC | AC5 知止: stop-decision metrics (false DONE, tested-before-done) on terminal/SWE runs | ✅ Done (v5.62, T98) — frontier rows need an API key |
| C | CUDA verification (#1), fused quant benchmark (#2), BPE tokenizer (#5), HF Hub (#4), ~1B GPU run | ⬜ Not started |
| D | DecisionHead at 1B, multi-env GRPO training at real scale (tooling exists since Phase 2.1), TrustGate CLARIFY threshold calibration | ⬜ Not started |
