"""HeliosLM v5.37 - TrustGate: calibrated abstention (the System-One
behavior tiers, deployed).

Assembles the v5.32-36 pieces into the production pattern from the
TypeSafe docs: confidence tiers drive BEHAVIOR, not just logging.
    high trust   -> DIRECT (auto-execute)
    medium       -> DIRECT but flag for logging
    low          -> ESCALATE (do not act on a low-confidence route)
The head's P(correct) comes from OUTCOMES (RLCD, v5.35b); the model's
own confidence is not consulted (measured to carry no signal at mid).
"""

try:
    from .calibrated_router import CalibratedRouter
    from .decision_head import DecisionHead
    from .gate import Gate, Route
except ImportError:
    from calibrated_router import CalibratedRouter
    from decision_head import DecisionHead
    from gate import Gate, Route


class TrustGate(Gate):
    """Wraps a trained DecisionHead; the trust question is a continuous
    Noul whose P(yes) IS P(the routed action will be correct)."""

    def __init__(self, head: DecisionHead, trust_q: str = "trust",
                 hi: float = 0.7, lo: float = 0.3, inner: Gate = None):
        assert head.qmeta.get(trust_q, (None,))[0] == "noul", \
            "trust question must be a continuous noul"
        self.router = CalibratedRouter(head)
        self.trust_q = trust_q
        self.hi, self.lo = hi, lo
        self.inner = inner          # optional fallback for abstains
        self.log: list = []

    def _trust(self, call, context):
        p, _ = self.router.head.forward(self.router._ids(call, context)) \
            [self.trust_q]
        return p

    def decide(self, call, context):
        p = self._trust(call, context)
        route = Route.DIRECT if p >= self.lo else Route.ESCALATE
        self.log.append({"call": call.name, "p_trust": p,
                         "route": route.value,
                         "tier": "high" if p >= self.hi else
                                 "medium" if p >= self.lo else "low"})
        return route

    def escalate(self, call, context):
        if self.inner is not None:
            return self.inner.escalate(call, context)
        raise NotImplementedError(
            "TrustGate abstained (low trust) and no oracle is configured")

    def ask(self, call, context, kind, question):
        return self.router.ask(call, context, kind, question)
