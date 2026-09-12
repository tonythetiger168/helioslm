"""
Kimi Delta Attention (KDA) + Gated MLA + Attention Residuals
Phase 1: 速度增强核心
"""

import math
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


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention: 3:1 比例交替线性/全局注意力
    支持动态稀疏度 + NoPE 外推
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.num_kv_heads = config.attention.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.residual_depth = config.attention.residual_depth

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.gate = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.linear_q_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_k_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)

        if config.attention.attn_res_enabled:
            self.residual_gates = nn.Parameter(torch.zeros(config.num_hidden_layers, self.residual_depth))
            self.residual_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(self.residual_depth)
            ])

        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def linear_attention(self, q, k, v):
        """O(n) 线性复杂度注意力"""
        q = F.elu(self.linear_q_kernel(q)) + 1
        k = F.elu(self.linear_k_kernel(k)) + 1
        kv_state = torch.einsum('bhsk,bhsv->bhkv', k, v)
        z = torch.einsum('bhqd,bhkd->bhq', q, k.sum(dim=2)) + 1e-6
        out = torch.einsum('bhqd,bhkd,bhkv->bhqv', q, k, kv_state) / z.unsqueeze(-1)
        return out

    def global_attention(self, q, k, v, mask=None):
        """O(n^2) 全局注意力"""
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkv->bhqv', attn_weights, v)
        return out

    def forward(self, hidden_states, layer_idx, past_residuals=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        normed = self.input_norm(hidden_states)

        q = self.q_proj(normed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        gate_values = torch.sigmoid(self.gate(normed)).unsqueeze(-1).transpose(1, 2)
        use_global = (layer_idx % 4 == 0)

        if use_global:
            attn_output = self.global_attention(q, k, v, attention_mask)
        else:
            attn_output = self.linear_attention(q, k, v)

        attn_output = attn_output * gate_values

        if self.config.attention.attn_res_enabled and past_residuals is not None:
            residual_output = torch.zeros_like(attn_output)
            for i, (residual, proj) in enumerate(zip(past_residuals[-self.residual_depth:], self.residual_proj)):
                gate = torch.sigmoid(self.residual_gates[layer_idx, i])
                residual_output += gate * proj(residual).view_as(attn_output)
            attn_output = attn_output + residual_output

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        return output, hidden_states
