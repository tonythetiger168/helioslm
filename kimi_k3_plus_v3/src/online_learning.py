"""Online Learning - P6: UserLoRA + Memory Consolidation"""
import torch
import torch.nn as nn
import time


class UserLoRA(nn.Module):
    """P6: Per-user LoRA adapter for personalization"""
    def __init__(self, config, user_id="default"):
        super().__init__()
        self.config = config
        self.user_id = user_id
        self.hidden_size = config.hidden_size
        self.rank = config.online_learning.lora_rank
        self.alpha = config.online_learning.lora_alpha
        self.lora_A = nn.Parameter(torch.randn(self.hidden_size, self.rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.hidden_size))
        self.preference_embed = nn.Parameter(torch.zeros(self.rank))

    def forward(self, hidden_states):
        if self.training:
            delta = torch.matmul(hidden_states, self.lora_A)
            delta = torch.matmul(delta, self.lora_B)
            return hidden_states + (self.alpha / self.rank) * delta
        return hidden_states

    def adapt(self, feedback_score, learning_rate=1e-4):
        with torch.no_grad():
            self.preference_embed += learning_rate * feedback_score * torch.randn_like(self.preference_embed)


class MemoryConsolidation:
    """P6: Sleep-phase memory consolidation"""
    def __init__(self, config):
        self.config = config
        self.short_term = []
        self.long_term = []
        self.consolidation_threshold = config.online_learning.consolidation_threshold
        self.compression_ratio = config.online_learning.memory_compression_ratio

    def add_short_term(self, key, value, importance=1.0):
        self.short_term.append({
            "key": key.detach() if isinstance(key, torch.Tensor) else key,
            "value": value,
            "importance": importance,
            "timestamp": time.time()
        })

    def consolidate(self):
        if len(self.short_term) < self.consolidation_threshold:
            return
        self.short_term.sort(key=lambda x: x["importance"], reverse=True)
        compressed = []
        for mem in self.short_term[:int(len(self.short_term) * self.compression_ratio)]:
            compressed.append(mem)
        self.long_term.extend(compressed)
        self.short_term = []

    def retrieve_long_term(self, query, top_k=5):
        if not self.long_term:
            return []
        scores = []
        for mem in self.long_term:
            if isinstance(query, torch.Tensor) and isinstance(mem["key"], torch.Tensor):
                sim = torch.cosine_similarity(query.flatten(), mem["key"].flatten(), dim=0)
                scores.append((sim.item(), mem))
        scores.sort(reverse=True)
        return [m for _, m in scores[:top_k]]


class OnlineLearningManager:
    """P6: Main online learning manager"""
    def __init__(self, config):
        self.config = config
        self.enabled = config.online_learning.enabled
        self.user_adapters = {}
        self.memory = MemoryConsolidation(config)

    def get_user_adapter(self, user_id):
        if user_id not in self.user_adapters:
            self.user_adapters[user_id] = UserLoRA(self.config, user_id)
        return self.user_adapters[user_id]

    def process_feedback(self, user_id, hidden_states, feedback_score):
        if not self.enabled:
            return hidden_states
        adapter = self.get_user_adapter(user_id)
        adapter.adapt(feedback_score)
        return adapter(hidden_states)

    def sleep_consolidate(self):
        self.memory.consolidate()
