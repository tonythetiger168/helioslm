"""Prefix-pool accounting ledger (Phase 3.1) — the K3-style prompt-cache
economics book for ``PrefixPool``.

``PrefixPool`` already counts hits/misses/hit_tokens; what it does NOT
answer is the deployment question: "what fraction of prompt-token compute
did the pool eliminate?" (the number a prompt-cache discount is made of).
``PoolLedger`` wraps a pool, records per-request token reuse via
``stats()`` deltas — no probe calls, no LRU side effects, no changes to
the verified pool itself — and reports the effective discount.

Honesty notes:
  - Generation (decode) tokens are EXCLUDED from the account: the ledger
    measures prefix reuse only. A claim about total serving cost would
    need decode accounting too; this module does not make that claim.
  - ``effective_discount`` is reuse/(reuse+computed) over PROMPT tokens;
    it is comparable to advertised cache discounts only under the same
    workload mix, which is why every report carries the entry list.
"""
from typing import Dict, List


class PoolLedger:
    """Accounting wrapper around a ``PrefixPool`` instance."""

    def __init__(self, pool):
        self.pool = pool
        self.requests = 0
        self.tokens_reused = 0      # prompt tokens served from pool
        self.tokens_computed = 0    # prompt tokens actually forwarded
        self.entries: List[dict] = []

    def generate(self, ids, **kwargs):
        s0 = self.pool.stats()
        out = self.pool.generate(ids, **kwargs)
        s1 = self.pool.stats()
        reused = int(s1["hit_tokens"] - s0["hit_tokens"])
        prompt = len(ids)
        self.tokens_reused += reused
        self.tokens_computed += max(prompt - reused, 0)
        self.requests += 1
        self.entries.append({
            "request": self.requests,
            "prompt_tokens": prompt,
            "reused_tokens": reused,
            "computed_tokens": max(prompt - reused, 0),
            "hit": reused > 0,
        })
        return out

    def report(self) -> Dict:
        total = self.tokens_reused + self.tokens_computed
        return {
            "requests": self.requests,
            "tokens_reused": self.tokens_reused,
            "tokens_computed": self.tokens_computed,
            "effective_discount": (self.tokens_reused / total)
            if total else 0.0,
            "scope": "prompt tokens only; decode cost excluded",
            "entries": list(self.entries),
        }
