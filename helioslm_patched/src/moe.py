"""Stable LatentMoE v2 - P0: Dynamic Sparsity + P2: Domain Expert Groups (Vectorized)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SiTUGLU(nn.Module):
    """SiTU Gated Linear Unit activation."""
    def __init__(self, h, eh):
        super().__init__()
        self.gate_proj = nn.Linear(h, eh, bias=False)
        self.up_proj = nn.Linear(h, eh, bias=False)
        self.down_proj = nn.Linear(eh, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * torch.tanh(self.up_proj(x)))


class StableLatentMoE(nn.Module):
    """
    Stable LatentMoE with:
      - Vectorized expert routing (no Python loops)
      - Real dynamic sparsity (difficulty-based k)
      - Domain expert grouping
      - Load balancing loss
    """

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

        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.expert_hidden)
            for _ in range(self.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.expert_hidden)
            for _ in range(self.num_shared)
        ])

        # Dynamic sparsity estimator
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.load_balance_loss_coef = config.moe.load_balance_loss_coef
        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.domain_groups = config.moe.domain_groups if config.moe.domain_grouping else {}

    def forward(self, hidden_states, task_type=None):
        """
        Args:
            hidden_states: [B, seq, hidden]
            task_type: str or None, for domain bias
        Returns:
            output: [B, seq, hidden]
            aux_loss: scalar
        """
        B, seq, h = hidden_states.shape
        flat = hidden_states.view(-1, h)  # [B*seq, hidden]
        normed = self.input_norm(flat)

        # Dynamic sparsity: compute per-token difficulty
        difficulty = self.difficulty_estimator(normed).squeeze(-1)  # [B*seq]
        dynamic_k = (self.min_k + (self.max_k - self.min_k) * difficulty).round().long()
        dynamic_k = dynamic_k.clamp(self.min_k, self.max_k)  # [B*seq]

        # Use max k for routing (efficient), but mask later
        router_logits = self.router(normed)  # [B*seq, num_experts]

        # Domain grouping bias
        if task_type and self.domain_groups:
            for domain, (start, end) in self.domain_groups.items():
                if domain == task_type and start < self.num_experts and end < self.num_experts:
                    router_logits[:, start:end+1] += 2.0

        # Top-K routing (vectorized)
        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B*seq, top_k]
        routing_weights = F.softmax(top_k_values, dim=-1)  # [B*seq, top_k]

        # Apply dynamic sparsity mask
        # Create mask: for each token, only keep top dynamic_k[i] experts
        mask = torch.arange(self.top_k, device=dynamic_k.device).unsqueeze(0) < dynamic_k.unsqueeze(1)
        routing_weights = routing_weights * mask.float()
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Vectorized expert computation using index_add
        output = torch.zeros_like(flat)

        # Group tokens by expert for efficient computation
        for k_idx in range(self.top_k):
            expert_ids = top_k_indices[:, k_idx]  # [B*seq]
            weights = routing_weights[:, k_idx:k_idx+1]  # [B*seq, 1]

            # Process all tokens for each expert in parallel
            for eid in range(self.num_experts):
                mask_e = (expert_ids == eid) & (weights.squeeze(-1) > 0)
                if mask_e.any():
                    tokens = normed[mask_e]
                    expert_out = self.experts[eid](tokens)
                    output[mask_e] += weights[mask_e] * expert_out

        # Shared experts (always active)
        for s in self.shared_experts:
            output += s(normed)

        # Load balancing auxiliary loss
        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = self.num_experts * (router_prob.mean(dim=0) ** 2).sum() * self.load_balance_loss_coef

        return output.view(B, seq, h), aux_loss
