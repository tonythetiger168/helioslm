# Phase 0–3 Execution Plan — Local Decision Engine

> **Reconstruction note (2026-10-06):** same provenance as
> `docs/ROADMAP.md` — drafted overnight, never pushed, reconstructed from
> `LLM_HANDOFF.md`. The Phase 0/1 rows are updated to their *actual*
> completed state so the plan reads as a living document.

## Phase overview

| Phase | Window | Key tasks | Status |
|-------|--------|-----------|--------|
| **0** | Day 1–2 | Merge diverged lines (ahead 4 / behind 160+), version v5.41, all tests green, push | ✅ done 2026-10-06 (`224fe36`, `fd754c1`) |
| **1** | Week 1–2 | DecisionHead module, unit tests, DPO trainer, integrate into HeliosLM mid/large | 1.1/1.2/1.4 done (`72cf579`); 1.3 integration smoke = this plan's next action |
| **2** | Week 2–4 | Multi-env GRPO (math / code / alignment-audit), long-context eval (32K–256K), calibration × quantization cross experiment | pending |
| **3** | Week 4–6 | Prefix pool accounting, TrustGate v2 (calibrated probabilistic), mini-benchmark harness, README GIF, r/LocalLLaMA post | pending |

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

### 1.3 Integration into HeliosLM mid/large
Wire the head onto the model's final-norm hidden states
(`model(input_ids)` returns `(logits, hidden_states, past)` per the
documented contract) and prove end-to-end gradient flow through the real
model. Deliverable: `tests/test_decision_head_integration.py`.

### 1.4 DPO Trainer — `helioslm_v5/src/training/dpo.py` ✅
Handoff named `src/trainers/dpo_trainer.py`; the repo's training-module
convention (`src/training/grpo.py`, `fp8_trainer.py`, …) won, noted
honestly in the module docstring. Shipped v5.41 with 6/6 oracles
including the analytic anchor π=ref ⇒ loss = log 2.

## Phase 2 preview

1. **Multi-env GRPO**: environment registry (math / code /
   alignment-audit); the alignment-audit env consumes RLCDAlignBench
   records directly.
2. **Long-context eval**: GraphWalks-style BFS / passkey generators over
   32K–256K; honest reporting only (no unverified context claims).
3. **Calibration × quantization cross**: run the same calibration probe
   against bf16 and NVFP4-QAT checkpoints; report drift (the unique
   positioning no frontier lab occupies).

## Phase 3 preview

Prefix-pool accounting (hit rate, token savings, effective cache
discount), TrustGate v2, mini-benchmark harness (`eval/score_stream`
upgraded to task-level JSONL), README demo GIF, launch post.
