"""Content-addressed KV prefix pool (v5.18, roadmap #6 final component).

Cross-session prefix reuse: many requests sharing a system prompt pay
prefill once. Token ids are chunked into fixed-size blocks; each block's
KV is keyed by a content hash of the token ids **plus a config
fingerprint** (model version, quant/tiering scheme — the same metadata
discipline proposed to colibri in P2), so a stale or differently-quantized
entry can never be served. Values are deep-copied cache snapshots; the
pool is model-level (per-session engine COW forks remain the hot path).

Correctness contract (oracle-tested): greedy generation seeded from a
pooled prefix is bitwise-identical to from-scratch generation, for full
hits, partial-prefix hits, post-eviction misses, and fingerprint
mismatches — a pooled prefix changes latency, never the answer.

Usage:
    pool = PrefixPool(model, block_size=16, max_blocks=64,
                      fingerprint="v5.18/fp32")
    out = pool.generate(ids, max_new_tokens=24, temperature=0)
    report = pool.stats()   # hits, misses, hit_tokens, evictions

Limitations (honest): CPU tensors only (deepcopy of cache); single
process; values are per-(model, fingerprint) — attach one pool per model.
"""

import hashlib
from collections import OrderedDict

import torch

__all__ = ["PrefixPool"]


class PrefixPool:
    """LRU pool of KV snapshots keyed by hashed token-id blocks."""

    def __init__(self, model, block_size=16, max_blocks=64,
                 fingerprint=""):
        self.model = model
        self.model.eval()
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.fingerprint = fingerprint
        self._pool = OrderedDict()   # key -> (n_tokens, past_key_values)
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0        # prompt tokens covered by prefix hits
        self.evictions = 0

    # ------------------------------------------------------------------
    def _key(self, block_ids):
        h = hashlib.blake2b(digest_size=16)
        h.update(self.fingerprint.encode("utf-8"))
        h.update(bytes(int(t) for t in block_ids))
        return h.hexdigest()

    def _blocks(self, ids):
        return [ids[i: i + self.block_size]
                for i in range(0, len(ids) - self.block_size + 1,
                               self.block_size)]

    # ------------------------------------------------------------------
    @staticmethod
    def _clone_past(past):
        return [tuple(t.clone() for t in layer) for layer in past]

    # ------------------------------------------------------------------
    def lookup(self, ids):
        """Longest pooled prefix of ``ids``.

        Returns (n_tokens, past, last_logits) or None. ``past`` covers
        EXACTLY the first n_tokens (per-block incremental prefill — never
        positions beyond the matched prefix).
        """
        matched = 0
        best = None
        for block in self._blocks(ids):
            key = self._key(block)
            if key not in self._pool:
                break
            n_tok, past, logits = self._pool[key]
            matched = n_tok
            best = (past, logits)
            self._pool.move_to_end(key)
        if best is None:
            return None
        past, logits = best
        return matched, self._clone_past(past), logits.clone()

    def store(self, ids):
        """Incremental per-block prefill; each block stores its own
        (cumulative length, exact-length past snapshot, last logits)."""
        logits = None
        past = None
        n = 0
        for block in self._blocks(ids):
            x = torch.tensor([block], dtype=torch.long)
            if past is None:
                lg, _, past = self.model(input_ids=x, use_cache=True)
            else:
                lg, _, past = self.model(input_ids=x, past_key_values=past,
                                         use_cache=True)
            logits = lg[0, -1].clone()
            n += len(block)
            key = self._key(block)
            self._pool[key] = (n, self._clone_past(past), logits.clone())
            self._pool.move_to_end(key)
            while len(self._pool) > self.max_blocks:
                self._pool.popitem(last=False)
                self.evictions += 1
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32, temperature=0.0):
        """Greedy/nucleus generation with prefix reuse.

        Seeded from the longest pooled prefix; extends the pool with the
        full prompt afterward. Returns token ids [len-matched prefix ids
        are NOT re-generated; output covers prompt tail + new tokens] —
        callers wanting the full row should concatenate ids + new tokens.
        """
        ids = [int(t) for t in ids]
        hit = self.lookup(ids)
        if hit is not None:
            matched, past, logits = hit
            self.hits += 1
            self.hit_tokens += matched
            if matched < len(ids):
                x = torch.tensor([ids[matched:]], dtype=torch.long)
                lg, _, past = self.model(input_ids=x,
                                         past_key_values=past,
                                         use_cache=True)
                logits = lg[0, -1]
            # matched == len(ids): reuse the stored last logits as-is
        else:
            self.misses += 1
            matched = 0
            self.store(ids)
            full = (len(ids) // self.block_size) * self.block_size
            if full == 0:
                lg, _, past = self.model(
                    input_ids=torch.tensor([ids], dtype=torch.long),
                    use_cache=True)
                logits = lg[0, -1]
            elif full < len(ids):
                # prefill the tail remainder on top of the deepest block
                _, past, _ = self.lookup(ids)
                lg, _, past = self.model(
                    input_ids=torch.tensor([ids[full:]], dtype=torch.long),
                    past_key_values=past, use_cache=True)
                logits = lg[0, -1]
            else:
                _, past, logits = self.lookup(ids)

        out = []
        for _ in range(max_new_tokens):
            if temperature == 0:
                nxt = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            out.append(nxt)
            logits, _, past = self.model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long),
                past_key_values=past, use_cache=True)
            logits = logits[0, -1]
        self.store(ids + out)   # extend pool with what we just computed
        return torch.tensor([ids + out], dtype=torch.long)

    def stats(self):
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_tokens": self.hit_tokens,
            "hit_rate": self.hits / max(self.hits + self.misses, 1),
            "evictions": self.evictions,
            "blocks_resident": len(self._pool),
            "max_blocks": self.max_blocks,
        }
