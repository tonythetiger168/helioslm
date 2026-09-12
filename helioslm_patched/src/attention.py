"""Kimi Delta Attention v2 - P0: KDA + Gated MLA + AttnRes + KV-Cache + FlashAttention + RoPE"""
import math
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


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) implementation."""

    def __init__(self, dim, max_seq_len=32768, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        # Precompute cos/sin for common sequence lengths
        self._precompute(max_seq_len)

    def _precompute(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]  # [B, H, T, D]
        if seq_len > self.max_seq_len:
            self._precompute(seq_len * 2)
            self.max_seq_len = seq_len * 2
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to q and k."""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention v2 with:
      - KV-Cache for efficient autoregressive generation
      - FlashAttention (torch SDPA) for training
      - RoPE positional embeddings
      - Gated MLA
      - AttnRes with configurable depth
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.num_kv_heads = config.attention.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.residual_depth = config.attention.residual_depth

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Gated MLA
        self.gate = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Linear attention kernels (for every 4th layer)
        self.linear_q_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_k_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # AttnRes
        if config.attention.attn_res_enabled:
            self.residual_gates = nn.Parameter(torch.zeros(config.num_hidden_layers, self.residual_depth))
            self.residual_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(self.residual_depth)
            ])

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta)

        self.input_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attention_dropout = config.attention.attention_dropout

    def linear_attention(self, q, k, v):
        """Linear attention for long sequences (every 4th layer)."""
        q = F.elu(self.linear_q_kernel(q)) + 1
        k = F.elu(self.linear_k_kernel(k)) + 1
        kv_state = torch.einsum("bhsk,bhsv->bhkv", k, v)
        z = torch.einsum("bhqd,bhkd->bhq", q, k.sum(dim=2)) + 1e-6
        return torch.einsum("bhqd,bhkd,bhkv->bhqv", q, k, kv_state) / z.unsqueeze(-1)

    def flash_attention(self, q, k, v, attention_mask=None, is_causal=True):
        """FlashAttention using torch SDPA with GQA support."""
        # q: [B, num_heads, T, D], k/v: [B, num_kv_heads, T, D]
        # GQA: repeat k,v heads to match q
        if self.num_kv_heads != self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)  # [B, num_heads, T, D]
            v = v.repeat_interleave(reps, dim=1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            mask = mask.expand(q.shape[0], 1, q.shape[2], k.shape[2])
            attn_mask = mask.bool()
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )
        return attn_out

    def forward(self, hidden_states, layer_idx, past_residuals=None, attention_mask=None, past_key_value=None, use_cache=False, position_ids=None):
        """
        Args:
            hidden_states: [B, seq, hidden]
            layer_idx: current layer index
            past_residuals: list of past residual states for AttnRes
            attention_mask: [B, seq] or None
            past_key_value: tuple of (past_k, past_v) for KV-Cache
            use_cache: whether to return present_key_value
            position_ids: [B, seq] position indices
        """
        B, seq, _ = hidden_states.shape
        normed = self.input_norm(hidden_states)

        # Project Q, K, V
        q = self.q_proj(normed).view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, seq_len=seq)
        if position_ids is not None:
            cos = cos[:, :, position_ids, :]
            sin = sin[:, :, position_ids, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV-Cache: concatenate with past
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # Gating
        gate = torch.sigmoid(self.gate(normed)).unsqueeze(-1).transpose(1, 2)

        # Attention: FlashAttention for most layers, linear for every 4th
        if layer_idx % 4 == 0:
            attn_out = self.flash_attention(q, k, v, attention_mask, is_causal=(past_key_value is None))
        else:
            attn_out = self.linear_attention(q, k, v)

        attn_out = attn_out * gate

        # AttnRes
        if self.config.attention.attn_res_enabled and past_residuals is not None:
            res_out = torch.zeros_like(attn_out)
            for i, (res, proj) in enumerate(zip(past_residuals[-self.residual_depth:], self.residual_proj)):
                res_out += torch.sigmoid(self.residual_gates[layer_idx, i]) * proj(res).view_as(attn_out)
            attn_out = attn_out + res_out

        # Output projection
        output = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, seq, self.hidden_size))

        return output, hidden_states, present_key_value
