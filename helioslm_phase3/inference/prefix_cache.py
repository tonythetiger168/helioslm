"""Prefix caching for common prompts.

Caches KV cache for frequently used prefixes (system prompts, 
instruction templates) to avoid recomputation.
"""
import hashlib
from typing import Dict, Optional, List, Tuple
import torch


class PrefixCache:
    """
    LRU cache for prompt prefixes.

    When a prompt starts with a cached prefix, reuse the prefix's
    KV cache instead of recomputing from scratch.
    """

    def __init__(self, max_cache_size: int = 100, max_prefix_len: int = 1024):
        self.max_cache_size = max_cache_size
        self.max_prefix_len = max_prefix_len

        # Cache: prefix_hash -> (kv_cache, prefix_length)
        self.cache: Dict[str, Tuple] = {}
        self.access_order: List[str] = []  # LRU tracking

    def _hash_prefix(self, tokens: List[int]) -> str:
        """Hash token sequence for cache lookup."""
        prefix_str = ",".join(map(str, tokens[:self.max_prefix_len]))
        return hashlib.md5(prefix_str.encode()).hexdigest()

    def get(self, tokens: List[int]) -> Optional[Tuple]:
        """Get cached KV cache for prefix."""
        key = self._hash_prefix(tokens)
        if key in self.cache:
            # Update LRU
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, tokens: List[int], kv_cache: Tuple):
        """Cache KV cache for a prefix."""
        key = self._hash_prefix(tokens)

        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_cache_size:
            # Evict LRU
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = kv_cache
        self.access_order.append(key)

    def find_longest_prefix(self, tokens: List[int]) -> Tuple[Optional[str], int]:
        """
        Find the longest cached prefix for given tokens.

        Returns: (cache_key, prefix_length) or (None, 0)
        """
        best_key = None
        best_len = 0

        for length in range(min(len(tokens), self.max_prefix_len), 0, -1):
            key = self._hash_prefix(tokens[:length])
            if key in self.cache:
                best_key = key
                best_len = length
                break

        return best_key, best_len

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "hit_rate": 0.0,  # Would need external tracking
        }
