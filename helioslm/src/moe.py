"""Stable LatentMoE+ - P0: Dynamic Sparsity + P2: Domain Expert Groups"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SiTUGLU(nn.Module):
    def __init__(self, h: int, eh: int):
        super().__init__()
        self.gate_proj = nn.Linear(h, eh, bias=False)
        self.up_proj = nn.Linear(h, eh, bias=False)
        self.down_proj = nn.Linear(eh, h, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * torch.tanh(self.up_proj(x)))

class StableLatentMoE(nn.Module):
    """Production MoE with Dynamic Sparsity + Domain Grouping + Load Balancing"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.num_shared = config.moe.num_shared_experts
        self.top_k = config.moe.num_activated_experts
        self.min_k = config.moe.min_experts
        self.max_k = config.moe.max_experts
        self.expert_hidden = config.moe.expert_hidden_size
        self.capacity_factor = config.moe.capacity_factor
        
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.router_noise = nn.Parameter(torch.zeros(1))
        self.experts = nn.ModuleList([SiTUGLU(self.hidden_size, self.expert_hidden) for _ in range(self.num_experts)])
        self.shared_experts = nn.ModuleList([SiTUGLU(self.hidden_size, self.expert_hidden) for _ in range(self.num_shared)])
        
        self.difficulty_estimator = nn.Sequential(nn.Linear(self.hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.load_balance_loss_coef = config.moe.load_balance_loss_coef
        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.domain_groups = config.moe.domain_groups if config.moe.domain_grouping else {}

    def forward(self, hidden_states: torch.Tensor, task_type: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, seq, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        normed = self.input_norm(flat)
        
        difficulty = self.difficulty_estimator(normed).squeeze(-1)
        dynamic_k = (self.min_k + (self.max_k - self.min_k) * difficulty).round().long().clamp(self.min_k, self.max_k)
        
        router_logits = self.router(normed)
        if self.training:
            noise = torch.randn_like(router_logits) * torch.sigmoid(self.router_noise)
            router_logits = router_logits + noise
        
        if task_type and self.domain_groups:
            for domain, (start, end) in self.domain_groups.items():
                if domain == task_type and end < self.num_experts:
                    router_logits[:, start:end + 1] += 2.0
        
        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_k_values, dim=-1)
        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = self.num_experts * (router_prob.mean(dim=0) ** 2).sum() * self.load_balance_loss_coef
        
        capacity = int(self.capacity_factor * flat.shape[0] / self.num_experts)
        output = torch.zeros_like(flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = routing_weights[:, i:i + 1]
            for eid in range(self.num_experts):
                mask = (expert_indices == eid)
                if not mask.any(): continue
                if mask.sum() > capacity:
                    overflow_indices = torch.where(mask)[0][capacity:]
                    mask[overflow_indices] = False
                expert_input = normed[mask]
                expert_output = self.experts[eid](expert_input)
                output[mask] += expert_weights[mask] * expert_output
        
        for shared in self.shared_experts:
            output += shared(normed)
        
        return output.view(B, seq, h), aux_loss
