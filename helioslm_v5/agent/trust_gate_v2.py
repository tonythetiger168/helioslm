"""HeliosLM v5.43 — TrustGate v2: calibrated probabilistic abstention
(Phase 3.2; v5.43 adds the quant-drift merge helper `apply_quant_drift`).

v1 (``trust_gate.py``) routes on hard thresholds hi/lo against P(yes).
Two things v1 cannot say: WHY the threshold is what it is, and how much to
trust the probability itself. v2 replaces both gaps explicitly:

1. Cost-sensitive threshold (the "probabilistic" half): DIRECT is chosen
   when the EXPECTED loss of acting is no worse than escalating —
   DIRECT iff (1 - p) * cost_wrong <= cost_escalate, i.e.
   ``p >= 1 - cost_escalate / cost_wrong``. The threshold is now an
   answer to a deployment question, not a magic number.

2. Calibration-uncertainty band (the "calibrated" half): a threshold
   computed from a probability is only as sharp as that probability's
   calibration. With a recorded ECE from held-out evaluation, v2 widens
   an abstain band around the threshold by exactly that ECE: inside the
   band it ESCALATES rather than pretending the probability is sharper
   than it is. Without any calibration record the band uses a
   conservative default and every decision is tagged
   ``"calibrated": False`` — loud, not silent.

Behavior tiers (high/medium/low) are kept for observability. The Gate
contract (decide/escalate/ask) is unchanged.
"""
try:
    from .calibrated_router import CalibratedRouter
    from .decision_head import DecisionHead
    from .gate import Gate, Route
except ImportError:
    from calibrated_router import CalibratedRouter
    from decision_head import DecisionHead
    from gate import Gate, Route

# Conservative abstain half-width when no calibration record is provided.
# 0.25 = "a quarter of the probability scale of doubt"; it is a declared
# policy, not an estimate.
_UNCALIBRATED_MARGIN = 0.25


def apply_quant_drift(calibration: dict, ece_drift: float) -> dict:
    """Widen a recorded abstain band by a quantization-induced ECE drift.

    Pairs TrustGate v2 with the v5.42 quant-calib cross probe: when an
    NVFP4 fake-quant twin measures WORSE calibration than its bf16 origin
    (``ece_drift = ece_quant - ece_bf16 > 0`` from
    ``eval/quant_calib.compare_bf16_vs_nvfp4``), a gate that keeps the
    bf16 band pretends the quantized probability is sharper than the
    measurement says it is. This helper merges the drift into the
    calibration record so the band widens accordingly.

    Declared policy — WIDEN-ONLY: a negative drift (quantization helped
    calibration) does NOT shrink the band below the bf16 record; trusting
    a lucky measurement to make the gate MORE aggressive is a policy
    choice, and the conservative default is to keep it out of the math.

    Args:
        calibration: {"ece": float in [0, 0.5], "n": int} measured on
            HELD-OUT data (same rule as TrustGateV2; ``n`` is provenance).
            None is a loud error — a drift against NO base record is
            exactly the uncalibrated case the gate already tags loudly.
        ece_drift: signed drift in ECE points from the quant-calib probe.

    Returns:
        A NEW dict (input never mutated) with the merged "ece", the raw
        "quant_drift", the "policy" name, and a "source" string — every
        field kept for audit provenance.
    """
    if calibration is None:
        raise ValueError(
            "apply_quant_drift needs a base calibration record; "
            "widening an absent record is the uncalibrated case — "
            "construct TrustGateV2 without calibration instead")
    ece = float(calibration.get("ece", -1))
    if not (0.0 <= ece <= 0.5):
        raise ValueError("base calibration ece must be recorded in "
                         "[0, 0.5]")
    drift = float(ece_drift)
    if drift != drift or abs(drift) == float("inf"):
        raise ValueError(f"ece_drift must be a finite float, got "
                         f"{ece_drift!r}")
    merged = dict(calibration)
    merged.update({
        "ece": min(0.5, ece + max(drift, 0.0)),
        "quant_drift": drift,
        "policy": "widen-only",
        "source": "eval/quant_calib.compare_bf16_vs_nvfp4",
    })
    return merged


class TrustGateV2(Gate):
    """Cost-sensitive, calibration-aware trust gate.

    Args:
        head: trained DecisionHead with a continuous-noul trust question.
        trust_q: name of the trust question.
        cost_wrong: loss incurred when DIRECT executes a wrong action.
        cost_escalate: loss incurred by escalating (human/oracle time).
        calibration: optional {"ece": float, "n": int} measured on
            HELD-OUT data (never the tuning set). ``n`` is recorded for
            provenance; the band uses ``ece`` only.
        inner: fallback gate consulted by ``escalate`` (like v1).

    Expected-loss rule (per decision, p = P(action correct)):
        loss(DIRECT)  = (1 - p) * cost_wrong
        loss(ESCALATE) = cost_escalate
        p* = 1 - cost_escalate / cost_wrong
    Abstain band: [p* - margin, p* + margin] -> ESCALATE, where
    margin = calibration["ece"] if calibrated else _UNCALIBRATED_MARGIN.
    """

    def __init__(self, head: DecisionHead, trust_q: str = "trust",
                 cost_wrong: float = 1.0, cost_escalate: float = 0.2,
                 calibration: dict = None, inner: Gate = None):
        assert head.qmeta.get(trust_q, (None,))[0] == "noul", \
            "trust question must be a continuous noul"
        if cost_wrong <= 0 or cost_escalate < 0:
            raise ValueError("costs must satisfy cost_wrong > 0, "
                             "cost_escalate >= 0")
        if cost_escalate >= cost_wrong:
            raise ValueError("cost_escalate >= cost_wrong makes DIRECT "
                             "never worth it — fix the costs, not the gate")
        if calibration is not None and \
                not (0.0 <= float(calibration.get("ece", -1)) <= 0.5):
            raise ValueError("calibration ece must be recorded in [0, 0.5]")
        self.router = CalibratedRouter(head)
        self.trust_q = trust_q
        self.cost_wrong = float(cost_wrong)
        self.cost_escalate = float(cost_escalate)
        self.calibration = calibration
        self.inner = inner
        self.log: list = []

    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        return 1.0 - self.cost_escalate / self.cost_wrong

    @property
    def margin(self) -> float:
        if self.calibration:
            return float(self.calibration["ece"])
        return _UNCALIBRATED_MARGIN

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def _trust(self, call, context) -> float:
        p, _ = self.router.head.forward(self.router._ids(call, context)) \
            [self.trust_q]
        return p

    # ------------------------------------------------------------------
    def decide_explain(self, call, context) -> dict:
        p = self._trust(call, context)
        p_star = self.threshold
        m = self.margin
        if p < p_star - m or p > p_star + m:
            route = Route.DIRECT if p > p_star else Route.ESCALATE
        else:
            route = Route.ESCALATE     # calibration uncertainty: do not act
        rec = {"call": call.name, "p_trust": p, "p_star": p_star,
               "margin": m, "band": [p_star - m, p_star + m],
               "route": route.value,
               "tier": "high" if p >= p_star + m else
                       "medium" if p >= p_star - m else "low",
               "calibrated": self.is_calibrated,
               "expected_loss": {
                   "direct": (1 - p) * self.cost_wrong,
                   "escalate": self.cost_escalate}}
        self.log.append(rec)
        return rec

    def decide(self, call, context) -> Route:
        return Route[self.decide_explain(call, context)["route"]]

    def escalate(self, call, context):
        if self.inner is not None:
            return self.inner.escalate(call, context)
        raise NotImplementedError(
            "TrustGateV2 abstained and no oracle is configured")
