"""Long-Term Memory v2 - P3 (Production-Ready)"""
import torch
import time
from typing import List, Dict, Optional


class LongTermMemory:
    """
    Long-term memory with FAISS-based retrieval and importance scoring.

    Improvements:
      - Proper tensor handling
      - Batch retrieval support
      - Memory decay based on time
    """

    def __init__(self, config):
        self.max_entries = config.agentic.memory_max_entries
        self.compression_ratio = config.agentic.memory_compression_ratio
        self.hidden_size = config.hidden_size
        self.memories = []
        self.use_faiss = False
        self.index = None

        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.hidden_size)
            self.use_faiss = True
        except ImportError:
            pass

    def add(self, key, value, importance=1.0):
        """
        Add memory entry.

        Args:
            key: tensor or string for retrieval
            value: any stored value
            importance: float importance score
        """
        # Evict least important if at capacity
        if len(self.memories) >= self.max_entries:
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i]["importance"])
            self.memories.pop(min_idx)
            # Rebuild FAISS index if needed
            if self.use_faiss:
                self._rebuild_index()

        entry = {
            "key": key.detach().cpu() if isinstance(key, torch.Tensor) else key,
            "value": value,
            "importance": importance,
            "timestamp": time.time(),
        }
        self.memories.append(entry)

        # Add to FAISS index
        if self.use_faiss and isinstance(key, torch.Tensor):
            self.index.add(key.detach().cpu().numpy().reshape(1, -1))

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most relevant memories.

        Args:
            query: tensor or string query
            top_k: int

        Returns:
            list of memory entries
        """
        if not self.memories:
            return []

        if self.use_faiss and isinstance(query, torch.Tensor):
            q = query.detach().cpu().numpy().reshape(1, -1)
            k = min(top_k, len(self.memories))
            distances, indices = self.index.search(q, k)
            return [self.memories[i] for i in indices[0] if i < len(self.memories) and i >= 0]

        # Fallback: importance-based retrieval
        return sorted(self.memories, key=lambda m: m["importance"], reverse=True)[:top_k]

    def retrieve_batch(self, queries, top_k=5):
        """Batch retrieval."""
        return [self.retrieve(q, top_k) for q in queries]

    def _rebuild_index(self):
        """Rebuild FAISS index after eviction."""
        if not self.use_faiss:
            return
        import faiss
        self.index = faiss.IndexFlatIP(self.hidden_size)
        for mem in self.memories:
            if isinstance(mem["key"], torch.Tensor):
                self.index.add(mem["key"].numpy().reshape(1, -1))

    def decay_importance(self, decay_factor=0.99):
        """Apply time-based importance decay."""
        current_time = time.time()
        for mem in self.memories:
            age = current_time - mem["timestamp"]
            mem["importance"] *= (decay_factor ** (age / 3600))  # Hourly decay
