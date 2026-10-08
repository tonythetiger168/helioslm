"""CalibratedPrefetcher - Colibri x HeliosLM RLCD integration (v5.41).

The prefetch decision ("which experts will the next layer need?") is a
decision-layer problem. This module replaces Colibri's frequency-based
ExpertPrefetcher with a calibrated one: a DecisionHead reads the current
layer's state and emits P(each expert will be needed). Low-confidence
predictions widen the prefetch set -- the inference-time analog of
TrustGate's "abstain -> gather more evidence".

CRITICAL RLCD property: the training signal is a REAL outcome, not a
label. A prefetch MISS (expert actually used but not prefetched) is an
observable downstream event -- the exact condition under which T28 says
the Brier calibration term has teeth (unlike AlignBench label==outcome).
"""
import torch

try:
    from .decision_head import DecisionHead
except ImportError:
    from decision_head import DecisionHead


class CalibratedPrefetcher:
    """Calibrated expert prefetcher for tiered MoE caches.

    Interface-compatible with Colibri's ExpertPrefetcher.predict().
    """

    def __init__(self, num_experts: int, d: int = 64, top_k: int = 2,
                 hi: float = 0.7, lo: float = 0.3, widen_factor: int = 2,
                 vocab: int = 1024):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hi, self.lo = hi, lo
        self.widen = widen_factor
        self.questions = {f"expert_{i}": ("noul", None)
                          for i in range(num_experts)}
        self.head = DecisionHead(self.questions, d=d, vocab=vocab,
                                 brier_lambda=1.0)

    def state_ids(self, layer_state_text: str):
        from trajectory import text_to_ids
        return torch.tensor(text_to_ids(layer_state_text[:600]))

    def predict(self, layer_state_text: str, current_experts=None):
        ids = self.state_ids(layer_state_text)
        out = self.head.forward(ids)
        probs = {name: p for name, (p, _) in out.items()}
        ranked = sorted(probs, key=lambda k: -probs[k])
        top_p = probs[ranked[0]]
        k = self.top_k
        if top_p < self.lo:
            k = min(self.num_experts, self.top_k * self.widen)
        elif top_p < self.hi:
            k = min(self.num_experts, self.top_k + 1)
        chosen = [int(n.split("_")[1]) for n in ranked[:k]]
        return chosen, {n.split("_")[1]: round(p, 3)
                        for n, p in probs.items()}

    def record_outcome(self, records: list):
        """RLCD training entry: records carry replay-verified outcomes."""
        self.head.fit(records, epochs=800, lr=1e-1)
