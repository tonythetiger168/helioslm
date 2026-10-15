# HeliosLM Milestones

Status of the numbered milestones from the 2026-10-13 landscape review,
plus the phases of [`ROADMAP_2026Q4.md`](ROADMAP_2026Q4.md). Update this
file in the same PR that lands or changes a milestone.

> The older "M1–M11" IDs in `docs/helioslm_v5_code_review.md` and
> `docs/V5.1_FIXES_REPORT.md` are a separate list (v5.1 code-review
> findings), not these milestones.

## 2026-10-13 landscape-review milestones

| ID | Milestone | Status | Landed in | Tests |
|---|---|---|---|---|
| M1 | Self-grading invariant: OracleGuard keeps graders / answer keys out of tool reach | ✅ Done | v5.51 (#17) | T82 |
| M2 | Artifact verification tiers: every benchmark JSON carries a registered tier, enforced in CI | ✅ Done | v5.51 (#17) | T83 |
| M3 | TrustGate CLARIFY three-state routing for calls with missing parameters | ✅ Done | v5.53 (#19), hardened in the 2026-10-15 review fixes | T86, T87 |
| M4 | **Undefined** — the source review is not in the repo and no commit names M4 | ❓ Needs definition | — | — |
| M5 | API probe refreshed to the 2026-10 model generation, with cost accounting | ✅ Done | v5.51 (#17) | T84 |
| M6 | Cost-per-completed-task baseline | ✅ Done | v5.52 (#18) | T85 |

M4 is left undefined on purpose rather than guessed. Whoever has the
2026-10-13 review should fill in this row.

## Roadmap 2026 Q4 phases

| Phase | Item | Status |
|---|---|---|
| A | Housekeeping: v5.52/v5.53 CHANGELOG entries, this milestone file | ✅ Done (v5.54) |
| A | Calibration leaderboard over committed probe artifacts | ✅ Done (v5.54, T88) |
| A | Calibration leaderboard: new frontier API rows (opus-5.5, sonnet-5.5, astra, sol, argon) | ⏳ Needs an API key run of `examples/probe_api.py` |
| A | Calibration leaderboard on RLCDAlignBench (7,193 items) | ⏳ Dataset (`all.jsonl`) is not in the repo |
| A | Standard eval harness (GSM8K / HumanEval subsets, terminal-style sandbox tasks) | ⬜ Not started |
| B | `k3_block` / `v4_csa` presets, Stable LatentMoE study | ⬜ Not started |
| C | CUDA verification (#1), fused quant benchmark (#2), BPE tokenizer (#5), HF Hub (#4), ~1B GPU run | ⬜ Not started |
| D | DecisionHead at 1B, multi-env GRPO, TrustGate CLARIFY threshold calibration | ⬜ Not started |
