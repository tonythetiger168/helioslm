"""Advanced memory system with multiple memory types."""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    embedding: Optional[torch.Tensor] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5  # 0-1, learned or manually set
    memory_type: str = "episodic"  # episodic, semantic, procedural
    metadata: Dict = field(default_factory=dict)


class EpisodicMemory:
    """Memory of specific experiences and conversations."""

    def __init__(self, embedding_dim: int = 12288, max_entries: int = 10000):
        self.entries: List[MemoryEntry] = []
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries

    def add(self, content: str, embedding: torch.Tensor, importance: float = 0.5):
        """Add an episodic memory."""
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            memory_type="episodic",
        )
        self.entries.append(entry)

        # Prune if too many
        if len(self.entries) > self.max_entries:
            self._prune()

    def _prune(self):
        """Remove least important memories."""
        self.entries.sort(key=lambda e: e.importance, reverse=True)
        self.entries = self.entries[:self.max_entries]

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve most relevant episodic memories."""
        if not self.entries:
            return []

        # Compute similarities
        similarities = []
        for entry in self.entries:
            if entry.embedding is not None:
                sim = torch.cosine_similarity(query_embedding, entry.embedding, dim=-1)
                similarities.append((sim.item(), entry))

        similarities.sort(reverse=True)
        return [entry for _, entry in similarities[:top_k]]


class SemanticMemory:
    """Memory of facts and general knowledge."""

    def __init__(self, embedding_dim: int = 12288):
        self.facts: Dict[str, MemoryEntry] = {}  # key -> entry
        self.embedding_dim = embedding_dim

    def store_fact(self, key: str, value: str, embedding: torch.Tensor):
        """Store a factual memory."""
        self.facts[key] = MemoryEntry(
            content=value,
            embedding=embedding,
            memory_type="semantic",
        )

    def query(self, query_embedding: torch.Tensor) -> Optional[str]:
        """Query for relevant facts."""
        if not self.facts:
            return None

        best_match = None
        best_sim = -1

        for key, entry in self.facts.items():
            if entry.embedding is not None:
                sim = torch.cosine_similarity(query_embedding, entry.embedding, dim=-1).item()
                if sim > best_sim:
                    best_sim = sim
                    best_match = entry.content

        return best_match if best_sim > 0.7 else None


class ProceduralMemory:
    """Memory of how to perform tasks (skills/workflows)."""

    def __init__(self):
        self.procedures: Dict[str, Dict] = {}  # task_name -> procedure

    def learn_procedure(self, task_name: str, steps: List[str], success_rate: float = 1.0):
        """Learn a new procedure."""
        self.procedures[task_name] = {
            "steps": steps,
            "success_rate": success_rate,
            "execution_count": 1,
        }

    def get_procedure(self, task_name: str) -> Optional[List[str]]:
        """Retrieve procedure steps."""
        proc = self.procedures.get(task_name)
        return proc["steps"] if proc else None

    def update_success(self, task_name: str, success: bool):
        """Update success rate based on execution."""
        if task_name in self.procedures:
            proc = self.procedures[task_name]
            proc["execution_count"] += 1
            # Exponential moving average
            alpha = 0.1
            proc["success_rate"] = (1 - alpha) * proc["success_rate"] + alpha * (1.0 if success else 0.0)


class AdvancedMemory:
    """
    Unified memory system combining all memory types.

    Usage:
        memory = AdvancedMemory()
        memory.episodic.add("User likes Python", embedding)
        memory.semantic.store_fact("python_creator", "Guido van Rossum", embedding)
        memory.procedural.learn_procedure("deploy_model", ["build", "test", "deploy"])
    """

    def __init__(self, embedding_dim: int = 12288):
        self.episodic = EpisodicMemory(embedding_dim)
        self.semantic = SemanticMemory(embedding_dim)
        self.procedural = ProceduralMemory()
        self.working_memory: List[str] = []  # Current context

    def add_to_working_memory(self, content: str):
        """Add to working memory (limited capacity)."""
        self.working_memory.append(content)
        if len(self.working_memory) > 10:  # Keep last 10 items
            self.working_memory.pop(0)

    def consolidate(self):
        """
        Consolidate working memory into episodic memory.

        Called periodically to transfer short-term to long-term memory.
        """
        if not self.working_memory:
            return

        # Combine working memory into a single episodic entry
        content = " | ".join(self.working_memory)
        # In real implementation, compute embedding
        # self.episodic.add(content, embedding)

        self.working_memory.clear()

    def retrieve_relevant(self, query_embedding: torch.Tensor) -> Dict[str, Any]:
        """Retrieve relevant memories from all types."""
        return {
            "episodic": self.episodic.retrieve(query_embedding),
            "semantic": self.semantic.query(query_embedding),
            "procedural": list(self.procedural.procedures.keys()),
            "working": self.working_memory,
        }
