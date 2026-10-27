# Phase 0–3 Execution Plan — Local Decision Engine

> **Status (2026-10-15): complete.** Every phase landed by v5.42
> (2026-10-06). This file is kept as the plan of record; live status for
> all milestones is tracked in [`MILESTONES.md`](MILESTONES.md), and the
> follow-on plan is [`ROADMAP_2026Q4.md`](ROADMAP_2026Q4.md).
>
> **Reconstruction note (2026-10-06):** same provenance as
> `docs/ROADMAP.md` — drafted overnight, never pushed, reconstructed from
> `LLM_HANDOFF.md`.
>
> **Commit hashes:** the hashes cited below (`224fe36`, `fd754c1`,
> `72cf579`) are from the pre-unification line and are not reachable in
> this repository's history (it starts at `dbcb53c`, which imported the
> tree). The CHANGELOG entries for v5.41 / v5.42 and the listed tests are
> the evidence that survives.

## Phase overview

| Phase | Window | Key tasks | Status |
|-------|--------|-----------|--------|
| **0** | Day 1–2 | Merge diverged lines (ahead 4 / behind 160+), version v5.41, all tests green, push | ✅ done 2026-10-06 (`224fe36`, `fd754c1`) |
| **1** | Week 1–2 | DecisionHead module, unit tests, DPO trainer, integrate into HeliosLM mid/large | ✅ done — 1.1/1.2/1.4 in v5.41 (`72cf579`), 1.3 in v5.42 |
| **2** | Week 2–4 | Multi-env GRPO (math / code / alignment-audit), long-context eval (32K–256K), calibration × quantization cross experiment | ✅ done in v5.42 — tooling and oracles; see scope notes below |
| **3** | Week 4–6 | Prefix pool accounting, TrustGate v2 (calibrated probabilistic), mini-benchmark harness, README GIF, r/LocalLLaMA post | ✅ done in v5.42 — the post is drafted, its publication is not recorded in the repo |

## Phase 1 detail

### 1.1 DecisionHead module — `helioslm_v5/src/decision_head.py` ✅
Spec (from the direction doc):

```python
class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, num_choices: int = 255):
        self.noul = nn.Linear(hidden_size, 1)      # Binary: P(yes)
        self.choice = nn.Linear(hidden_size, num_choices)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        return {
            'noul': torch.sigmoid(self.noul(hidden)).squeeze(-1),
            'choice': torch.softmax(self.choice(hidden), dim=-1),
            'score': self.score(hidden).squeeze(-1),
        }
```

Shipped v5.41 with loud-error guards and arbitrary leading-dim support.

### 1.2 Unit tests
Handoff named `helioslm_v5/tests/test_decision_head.py`, but that name was
already taken by the v5.32 agent-variant tests, so the shipped file is
`tests/test_lm_decision_head.py` (5/5 oracles).

### 1.3 Integration into HeliosLM mid/large ✅ (v5.42)
Wire the head onto the model's final-norm hidden states
(`model(input_ids)` returns `(logits, hidden_states, past)` per the
documented contract) and prove end-to-end gradient flow through the real
model. Deliverable: `tests/test_decision_head_integration.py`.

Shipped v5.42 (3/3). Recorded finding: the decision path bypasses
`lm_head` by design, and the aux-loss-free `route_bias` carries no
backprop gradient (heuristic-only).

### 1.4 DPO Trainer — `helioslm_v5/src/training/dpo.py` ✅
Handoff named `src/trainers/dpo_trainer.py`; the repo's training-module
convention (`src/training/grpo.py`, `fp8_trainer.py`, …) won, noted
honestly in the module docstring. Shipped v5.41 with 6/6 oracles
including the analytic anchor π=ref ⇒ loss = log 2.

## Phase 2 — delivered v5.42

1. **Multi-env GRPO** ✅ `src/training/envs.py`: environment registry
   with MathEnv, CodeEnv (literal compare, never exec) and
   AlignmentAuditEnv (consumes RLCDAlignBench records, refuses synthetic
   data), plus MultiEnvBatch mixing. Tests: `test_envs.py`. The
   end-to-end loop that wires the registry into the GRPO update followed
   in v5.43 (`src/training/env_grpo.py`; async variant v5.44), tested in
   `test_v5.py` via `case_trainers.py`. *Scope:* these are toy-scale
   oracles; a multi-env GRPO training run at real scale is roadmap
   Phase D.
2. **Long-context eval** ✅ `eval/longctx.py`: passkey-by-loglikelihood
   harness, refuses rotary extrapolation without explicit
   `rope_scaling`. Tests: `test_longctx.py`, `test_longctx_harness.py`.
   *Scope:* passkey only (no GraphWalks-style BFS generator), and tests
   run at 128–512 tokens; the 32K–256K range is supported by the
   interface but no result at those lengths is recorded.
3. **Calibration × quantization cross** ✅ `eval/quant_calib.py`: ECE and
   accuracy for bf16 vs NVFP4 fake-quant twins of the same weights.
   Tests: `test_quant_calib.py`.

## Phase 3 — delivered v5.42

- Prefix-pool accounting ✅ `src/inference/pool_ledger.py` (prompt tokens
  reused vs. computed, effective prompt-token discount, per-request
  entries; decode cost out of scope). Tests: `test_pool_ledger.py`.
- TrustGate v2 ✅ `agent/trust_gate_v2.py` (cost-sensitive threshold,
  abstain band widened by the head's measured ECE). Tests:
  `test_trust_gate_v2.py`.
- Mini-benchmark harness ✅ `eval/mini_bench.py` (task-level JSONL,
  choice-loglikelihood scoring). Tests: `test_mini_bench.py`.
- README demo GIF ✅ `docs/demo.gif`.
- Launch post ✅ drafted in `docs/launch_post_reddit.md`; whether and
  when it was posted is not recorded in the repo.
