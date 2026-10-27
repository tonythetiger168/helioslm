# HeliosLM Roadmap — Local Decision Engine Direction

> **Reconstruction note (2026-10-06):** this document was drafted in the
> 2026-10-06 overnight session but never reached `origin/main` (the push
> carried the code changes only). It was reconstructed from
> `LLM_HANDOFF.md` after a disk-wide search found no surviving copy. If the
> original sandbox copy resurfaces, diff and prefer the more complete one.

## Background

| External system | Positioning | Problem |
|-----------------|-------------|---------|
| **Jev (TypeSafe AI)** | Cloud decision model | Closed-source, API dependency, cannot be verified locally |
| **Harness-MU** | Multi-user agent safety architecture | Gatekeeper is hard rules, no calibration capability |

**HeliosLM's opportunity:** become the locally deployable "Jev +
Harness-MU" — a verifiable, calibratable, correctness-first **decision
engine**.

## Why HeliosLM is positioned for this

- Correctness-first reference implementation; every mechanism carries
  oracle tests (the "verified" half of "verifiable")
- Calibration assets already in-tree: TrustGate (calibrated abstention),
  continuous noul, RLCD outcome loop, decision-layer audit toolkit,
  RLCDAlignBench (7,193 instances, 16 benchmarks)
- Level 1 result (0.796 AUROC, SFT + linear probe) is the current best
  calibration detector; Level 2 (RLCD-FT of a general base) failed at
  ~0.50 AUROC across seven configurations — root cause: a 0.6B general
  base lacks the alignment-failure inductive bias. Conclusion: **the
  decision readout must live on a decision-shaped model, not a general
  LM fine-tuned into one.**

## Core architecture: Decision Head

Typed, non-autoregressive decisions over the model's own hidden states —
one parallel pass answers all decision questions (the Jev property:
decisions don't need token-by-token generation).

```python
class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, num_choices: int = 255):
        self.noul = nn.Linear(hidden_size, 1)      # Binary: P(yes)
        self.choice = nn.Linear(hidden_size, num_choices)  # routing
        self.score = nn.Linear(hidden_size, 1)      # continuous score
```

Implemented as `helioslm_v5/src/decision_head.py` (v5.41, LM-backed).
The toy-scale self-contained variant remains at
`helioslm_v5/agent/decision_head.py` (v5.32).

## Training path

```
Stage 1: SFT        (HeliosLM 360M SFT, 0.796 AUROC starting point)
Stage 2: RLCD + Brier  (on the dedicated Decision Head, not a general LM)
Stage 3: DPO + Multi-env GRPO  (Math / Code / Alignment-audit environments)
```

`src/training/dpo.py` (v5.41) covers the DPO stage. Multi-env GRPO and
the environment registry are pending (Phase 2).

## Evidence-first culture (carried forward)

- ECE/Brier reported, never assumed
- honest no-logprobs stance: without logprobs, say calibration is not
  possible — do not compute it anyway
- anti-shortcut checks on every confidence claim
- the failed Level 2 experiment stays in `docs/BENCHMARKS.md` as the
  recorded reason the engine direction exists
