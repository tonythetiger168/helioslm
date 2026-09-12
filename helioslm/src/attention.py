"""Helios Attention - P0: FlashAttention-compatible + Gated MLA + AttnRes"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class HeliosAttention(nn.Module):
    """Production-grade attention with Gated MLA + AttnRes + RoPE + FlashAttention"""
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
        
        self.rope_theta = config.rope_theta
        if config.attention.attn_res_enabled:
            self.residual_gates = nn.Parameter(torch.zeros(config.num_hidden_layers, self.residual_depth))
            self.residual_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(self.residual_depth)
            ])
        
        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # Reshape to (1, 1, seq_len, head_dim) for broadcasting with x: (batch, num_heads, seq_len, head_dim)
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin

    def forward(self, hidden_states: torch.Tensor, layer_idx: int,
                past_residuals: Optional[List[torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, seq, _ = hidden_states.shape
        normed = self.input_norm(hidden_states)
        q = self.q_proj(normed).view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA: repeat KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        q = self._apply_rope(q, seq)
        k = self._apply_rope(k, seq)
        gate = torch.sigmoid(self.gate(normed)).unsqueeze(-1).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention') and attention_mask is None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out * gate
        if self.config.attention.attn_res_enabled and past_residuals:
            res_out = torch.zeros_like(attn_out)
            for i, (res, proj) in enumerate(zip(past_residuals[-self.residual_depth:], self.residual_proj)):
                gated_res = torch.sigmoid(self.residual_gates[layer_idx, i])
                projected = proj(res).view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
                res_out += gated_res * projected
            attn_out = attn_out + res_out
        
        output = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, seq, self.hidden_size))
        return output, hidden_states
