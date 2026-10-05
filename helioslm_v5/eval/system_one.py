"""Decision-layer audit toolkit (v5.22) — calibration and routing gates.

Context: System One decision models (TypeSafe AI's Jev, 2026-09-15) make
fast, typed, probabilistic decisions for agent loops. Their credibility
rests on CALIBRATION (RLCD), but published validation uses the *average of
the most expensive LLMs* as reference probabilities — LLM-as-judge again.
This module provides the deterministic complement:

1. `MockSystemOne` — a controllable stand-in for the API (which is
   waitlisted): per-question probabilities over a verifiable task family,
   with a temperature knob that degrades calibration. Ground truth is
   known BY CONSTRUCTION, so calibration is measured, not judged.

2. Calibration metrics: ECE, Brier, reliability curve — against ground
   truth, no reference model required.

3. `ThresholdRouter` + the ROUTING MONOTONICITY GATE: with an always-
   correct escalation path, raising the direct-action threshold tau can
   only INCREASE correctness (it moves marginal cases to the oracle) at
   higher cost. This is a provable invariant — `routing_gate()` checks it
   empirically per tau grid, the same role the bitwise gate plays for
   cache/draft/tier: an optimization that violates monotonicity is a bug,
   not a trade-off.

4. `evolve_threshold()` — cost model (cheap decision vs expensive oracle)
   under a correctness floor, the decision-layer analog of
   `HarnessEvolver` (ModularRSI-style: the evolvable module is the
   routing policy, the gate is monotonicity + the correctness floor).

This is not a Jev competitor; it is the audit kit a Jev-style deployment
should run in CI: calibration drift, threshold regression, and routing
monotonicity as testable invariants.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch

__all__ = ["MockSystemOne", "expected_calibration_error", "brier_score",
           "reliability_curve", "ThresholdRouter", "routing_gate",
           "evolve_threshold"]


# ---------------------------------------------------------------------------
# Mock System One decision model (verifiable task family, controllable miscal)
# ---------------------------------------------------------------------------

class MockSystemOne:
    """State -> typed probabilistic decisions, ground truth known.

    Task: "which bucket does this vector belong to?" — buckets are fixed
    random directions; the model sees the vector plus noise. Logit
    temperature > 1 degrades calibration systematically (overconfidence
    or underconfidence), letting the metrics discriminate.

    Determinism: buckets AND the injected noise are drawn from per-instance
    seeded generators (never the global RNG), so a ``seed`` fully
    reproduces an audit run — two fresh instances with the same seed give
    identical outputs, and ``routing_gate``/``evolve_threshold`` numbers
    are comparable across processes.
    """

    def __init__(self, n_buckets: int = 4, dim: int = 16, noise: float = 0.4,
                 temperature: float = 1.0, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.buckets = torch.randn(n_buckets, dim, generator=g)
        self._noise_gen = torch.Generator().manual_seed(seed)
        self.noise = noise
        self.temperature = temperature
        self.n_buckets = n_buckets
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns {probs [B, K], correct [B] (bool), gold [B]}."""
        logits = x @ self.buckets.T / math.sqrt(self.dim)
        gold = logits.argmax(dim=-1)
        noisy = x + torch.randn(x.shape, generator=self._noise_gen,
                                dtype=x.dtype) * self.noise
        model_logits = noisy @ self.buckets.T / math.sqrt(self.dim)
        probs = torch.softmax(model_logits / self.temperature, dim=-1)
        pred = probs.argmax(dim=-1)
        return {"probs": probs, "correct": pred == gold, "gold": gold}

    def sample_batch(self, n: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(n, self.dim, generator=g)


# ---------------------------------------------------------------------------
# Calibration metrics (ground truth, not reference models)
# ---------------------------------------------------------------------------

def reliability_curve(probs: torch.Tensor, correct: torch.Tensor,
                      n_bins: int = 10) -> List[Dict[str, float]]:
    """Per-bin mean predicted confidence vs empirical accuracy."""
    conf, pred = probs.max(dim=-1)
    bins = []
    edges = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append({"bin": i, "lo": lo, "hi": hi, "n": n,
                     "mean_conf": float(conf[mask].mean()),
                     "acc": float(correct[mask].float().mean())})
    return bins


def expected_calibration_error(probs: torch.Tensor, correct: torch.Tensor,
                               n_bins: int = 10) -> float:
    bins = reliability_curve(probs, correct, n_bins)
    total = sum(b["n"] for b in bins)
    return sum(b["n"] / total * abs(b["mean_conf"] - b["acc"]) for b in bins)


def brier_score(probs: torch.Tensor, correct: torch.Tensor) -> float:
    """Multi-class Brier: mean squared error of the correctness event."""
    conf = probs.max(dim=-1).values
    return float(((conf - correct.float()) ** 2).mean())


# ---------------------------------------------------------------------------
# Threshold router + monotonicity gate
# ---------------------------------------------------------------------------

@dataclass
class RoutingReport:
    tau: float
    direct_rate: float        # fraction answered directly (no oracle)
    correctness: float
    cost: float
    gate_passed: bool         # monotonicity: correctness >= lower-tau run
    notes: List[str] = field(default_factory=list)


class ThresholdRouter:
    """Confidence-threshold escalation router.

    probs >= tau  -> act directly on the model's argmax (cheap)
    probs <  tau  -> escalate to the oracle (always correct, expensive)
    """

    def __init__(self, oracle_cost: float = 10.0, decision_cost: float = 1.0):
        self.oracle_cost = oracle_cost
        self.decision_cost = decision_cost

    def __call__(self, result: Dict[str, torch.Tensor],
                 tau: float) -> RoutingReport:
        conf = result["probs"].max(dim=-1).values
        direct = conf >= tau
        correct_direct = result["correct"][direct]
        # escalation path is ALWAYS correct by construction (oracle)
        n = len(conf)
        n_direct = int(direct.sum())
        correctness = ((float(correct_direct.sum()) + (n - n_direct)) / n)
        cost = (n_direct * self.decision_cost
                + (n - n_direct) * self.oracle_cost) / n
        return RoutingReport(tau=tau, direct_rate=n_direct / n,
                             correctness=correctness, cost=cost,
                             gate_passed=True)   # set by routing_gate


def routing_gate(model: MockSystemOne, router: ThresholdRouter,
                 taus: List[float], n_samples: int = 512,
                 seed: int = 0) -> List[RoutingReport]:
    """Evaluate the tau grid AND enforce the monotonicity invariant.

    With an always-correct escalation path, correctness must be
    non-decreasing in tau. A violation means the routing layer has a bug
    (e.g., oracle itself noisy) — exactly the role the bitwise gate plays
    for cache/draft/tier: an 'optimization' that breaks the invariant is
    rejected on the spot.
    """
    batch = model.sample_batch(n_samples, seed=seed)
    result = model(batch)
    reports = []
    prev_correct = 0.0
    for tau in sorted(taus):
        rep = router(result, tau)
        rep.gate_passed = rep.correctness >= prev_correct - 1e-9
        if not rep.gate_passed:
            rep.notes.append("monotonicity violated: correctness dropped "
                             "when tau increased — escalation path is not "
                             "trustworthy")
        prev_correct = max(prev_correct, rep.correctness)
        reports.append(rep)
    return reports


def evolve_threshold(model: MockSystemOne, router: ThresholdRouter,
                     correctness_floor: float = 0.95,
                     taus: List[float] = None,
                     n_samples: int = 512, seed: int = 0) -> Dict:
    """Cheapest tau meeting the correctness floor (decision-layer analog
    of HarnessEvolver: the evolvable module is the routing policy, the
    gate is monotonicity + the floor)."""
    taus = taus or [round(0.5 + 0.05 * i, 2) for i in range(11)]
    reports = routing_gate(model, router, taus, n_samples, seed)
    feasible = [r for r in reports if r.correctness >= correctness_floor
                and r.gate_passed]
    best = min(feasible, key=lambda r: r.cost) if feasible else None
    return {
        "best_tau": best.tau if best else None,
        "best_cost": best.cost if best else None,
        "correctness": best.correctness if best else None,
        "direct_rate": best.direct_rate if best else None,
        "evaluated": [{"tau": r.tau, "cost": round(r.cost, 3),
                       "correctness": round(r.correctness, 4),
                       "gate": r.gate_passed} for r in reports],
        "floor": correctness_floor,
        "oracle": "routing monotonicity + correctness floor "
                  "(escalation path assumed always correct)",
    }
