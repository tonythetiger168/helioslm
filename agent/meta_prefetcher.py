"""MetaPrefetcher - RSI route (6): the improver becomes the optimized object.

Extends CalibratedPrefetcher: its OWN hi/lo thresholds are meta-learned
from the prefetch hit-rate outcome. The survey's meta-level improvement
in minimal form -- the system that decides is itself adjusted by a
measurable outcome (miss rate).
"""


class MetaPrefetcher:
    def __init__(self, base_prefetcher, hi=0.7, lo=0.3,
                 adapt_rate=0.05, target_miss=0.15):
        self.pf = base_prefetcher
        self.hi, self.lo = hi, lo
        self.adapt = adapt_rate
        self.target_miss = target_miss
        self.history = []

    def predict(self, state_text, current_experts=None):
        self.pf.hi, self.pf.lo = self.hi, self.lo
        return self.pf.predict(state_text, current_experts)

    def record_outcome(self, hit, n_prefetched, n_actually_used):
        self.history.append({"hit": hit, "k": n_prefetched,
                             "used": n_actually_used})
        if len(self.history) < 8:
            return
        recent = self.history[-20:]
        miss_rate = 1 - sum(h["hit"] for h in recent) / len(recent)
        over_fetch = sum(h["k"] - h["used"] for h in recent) / len(recent)
        if miss_rate > self.target_miss:
            self.lo = max(0.05, self.lo - self.adapt)
        elif over_fetch > 1.0:
            self.hi = min(0.95, self.hi + self.adapt)
