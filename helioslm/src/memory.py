"""Long-Term Memory - P3 Production with FAISS + Importance Scoring"""
import torch
import time
from typing import List, Dict, Optional
import numpy as np


class LongTermMemory:
    """Production long-term memory with FAISS vector indexing, importance-based eviction, temporal decay"""
    def __init__(self, config):
        self.max_entries = config.agentic.memory_max_entries
        self.compression_ratio = config.agentic.memory_compression_ratio
        self.hidden_size = config.hidden_size
        self.memories: List[Dict] = []
        self.use_faiss = False
        self.index = None
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.hidden_size)
            self.use_faiss = True
        except ImportError:
            pass
        self.importance_threshold = 0.3

    def compute_importance(self, key: torch.Tensor, value: str, access_count: int = 1) -> float:
        recency = 1.0
        frequency = min(access_count / 10.0, 1.0)
        salience = torch.norm(key).item() / (self.hidden_size ** 0.5)
        salience = min(salience, 1.0)
        return 0.3 * recency + 0.3 * frequency + 0.4 * salience

    def add(self, key: torch.Tensor, value: str, importance: Optional[float] = None,
            metadata: Optional[Dict] = None):
        if len(self.memories) >= self.max_entries:
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i]["importance"])
            self.memories.pop(min_idx)
            if self.use_faiss:
                self._rebuild_index()
        if importance is None:
            importance = self.compute_importance(key, value)
        entry = {
            "key": key.detach().cpu() if isinstance(key, torch.Tensor) else key,
            "value": value, "importance": importance, "timestamp": time.time(),
            "access_count": 0, "metadata": metadata or {},
        }
        self.memories.append(entry)
        if self.use_faiss and isinstance(key, torch.Tensor):
            key_np = key.detach().cpu().numpy().reshape(1, -1)
            self.index.add(key_np)

    def retrieve(self, query: torch.Tensor, top_k: int = 5, min_relevance: float = 0.5) -> List[Dict]:
        if not self.memories:
            return []
        results = []
        if self.use_faiss and isinstance(query, torch.Tensor):
            query_np = query.detach().cpu().numpy().reshape(1, -1)
            distances, indices = self.index.search(query_np, min(top_k * 2, len(self.memories)))
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.memories):
                    memory = self.memories[idx]
                    similarity = dist / (self.hidden_size ** 0.5)
                    if similarity > min_relevance:
                        memory["access_count"] += 1
                        results.append({**memory, "similarity": similarity})
        else:
            query_flat = query.flatten()
            scored = []
            for i, mem in enumerate(self.memories):
                mem_key = mem["key"].flatten() if isinstance(mem["key"], torch.Tensor) else torch.tensor(mem["key"]).flatten()
                sim = torch.dot(query_flat, mem_key).item() / (torch.norm(query_flat).item() * torch.norm(mem_key).item() + 1e-8)
                if sim > min_relevance:
                    scored.append((sim, i))
            scored.sort(reverse=True)
            for sim, idx in scored[:top_k]:
                self.memories[idx]["access_count"] += 1
                results.append({**self.memories[idx], "similarity": sim})
        results.sort(key=lambda x: x["similarity"] * 0.6 + x["importance"] * 0.4, reverse=True)
        return results[:top_k]

    def _rebuild_index(self):
        if not self.use_faiss:
            return
        import faiss
        self.index = faiss.IndexFlatIP(self.hidden_size)
        keys = []
        for mem in self.memories:
            if isinstance(mem["key"], torch.Tensor):
                keys.append(mem["key"].numpy().reshape(1, -1))
        if keys:
            self.index.add(np.vstack(keys))

    def compress_old_memories(self):
        current_time = time.time()
        for mem in self.memories:
            age = current_time - mem["timestamp"]
            if age > 86400 * 7:
                mem["importance"] *= 0.9
