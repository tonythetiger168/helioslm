"""
Stable LatentMoE+ with Dynamic Sparsity
Phase 1: 动态稀疏度降低推理成本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SiTUGLU(nn.Module):
    """SiTU-GLU 激活"""
    def __init__(self, hidden_size, expert_hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        up = torch.tanh(self.up_proj(x))
        return self.down_proj(gate * up)


class StableLatentMoE(nn.Module):
    """动态稀疏度 MoE"""
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

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.expert_hidden)
            for _ in range(self.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.expert_hidden)
            for _ in range(self.num_shared)
        ])

        # 动态稀疏度估计器
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        self.load_balance_loss_coef = config.moe.load_balance_loss_coef
        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, task_type=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)
        normed = self.input_norm(flat)

        # Phase 1: 动态稀疏度 - 根据任务难度调整激活专家数
        difficulty = self.difficulty_estimator(normed).squeeze(-1)
        dynamic_k = (self.min_k + (self.max_k - self.min_k) * difficulty).round().long().clamp(self.min_k, self.max_k)

        router_logits = self.router(normed)

        # 领域偏置
        if task_type == "programming":
            router_logits[:, 0:224] += 2.0
        elif task_type == "mathematics":
            router_logits[:, 224:448] += 2.0
        elif task_type == "science":
            router_logits[:, 448:672] += 2.0
        elif task_type == "creative":
            router_logits[:, 672:896] += 2.0

        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_k_values, dim=-1)

        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = self.num_experts * (router_prob.mean(dim=0) ** 2).sum() * self.load_balance_loss_coef

        output = torch.zeros_like(flat)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = routing_weights[:, i:i+1]
            for eid in range(self.num_experts):
                mask = (expert_indices == eid)
                if mask.any():
                    output[mask] += expert_weights[mask] * self.experts[eid](normed[mask])

        for shared in self.shared_experts:
            output += shared(normed)

        return output.view(batch_size, seq_len, hidden_size), aux_loss
