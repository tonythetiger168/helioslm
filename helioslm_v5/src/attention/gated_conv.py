"""Gated short-convolution token mixer (v5.126, LB9 of the 2026-10-23
leaderboard review; LFM2 / LFM2.5, Liquid AI).

LFM2 replaces attention in most layers of its small models (10 of 16 at
350M-1.2B) with a double-gated, causal, depthwise short convolution, and
keeps a few attention layers for long-range retrieval. It is cheap on
CPUs and phones: a kernel of 3 and a fixed-size cache.

    B, C, x = split(in_proj(h))          three [B, T, d] streams
    u_t     = B_t * x_t                  input gate
    y_t     = sum_{j < k} w_{k-1-j} * u_{t-j}   depthwise causal conv
    out_t   = out_proj(C_t * y_t)        output gate

Cache: the last k - 1 gated inputs u, [B, k - 1, d], marked as a
``StateTensor`` (fixed size, no sequence axis), so ``past_seq_len``, MTP
rollback and engine batching treat it like the delta-rule state.

Packed sequences: a tap reaches back j tokens only when the positions
step by exactly j (a position restart starts a new document), so the
convolution never mixes documents. Padding (attention_mask 0) writes a
zero input. Same forward signature as MLA / GatedDeltaAttention.
"""
import torch
import torch.nn as nn

from .linear_attention import StateTensor


class GatedShortConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.kernel = int(config.gated_conv_kernel)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        # depthwise taps, Conv1d default init (the model-wide init only
        # touches Linear / Embedding)
        conv = nn.Conv1d(d, d, self.kernel, groups=d, bias=False)
        self.conv_weight = nn.Parameter(conv.weight.detach()[:, 0, :].clone())

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None):
        B, T, d = hidden_states.shape
        if T == 0:
            raise ValueError("GatedShortConv received an empty sequence")
        k = self.kernel
        bg, cg, xv = self.in_proj(hidden_states).chunk(3, dim=-1)
        u = bg * xv
        if attention_mask is not None:
            m = attention_mask[:, -T:].to(u.dtype)
            u = u * m.unsqueeze(-1)
        if past_key_value is not None:
            prev = past_key_value[0].as_subclass(torch.Tensor)
            if prev.shape != (B, k - 1, d):
                raise ValueError(f"conv state shape {tuple(prev.shape)} != "
                                 f"{(B, k - 1, d)}")
            prev = prev.to(u.dtype)
        else:
            prev = u.new_zeros(B, k - 1, d)
        full = torch.cat([prev, u], dim=1)          # [B, k - 1 + T, d]
        valid = None
        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            pos = pos.expand(B, T)
            lead = pos[:, :1] - torch.arange(k - 1, 0, -1, device=pos.device)
            valid = torch.cat([lead, pos], dim=1)   # [B, k - 1 + T]
        y = torch.zeros_like(u)
        for j in range(k):
            tap = full[:, k - 1 - j:k - 1 - j + T]
            if valid is not None and j:
                same = (valid[:, k - 1:] - valid[:, k - 1 - j:k - 1 - j + T]
                        == j)
                tap = tap * same.unsqueeze(-1).to(tap.dtype)
            y = y + self.conv_weight[:, k - 1 - j] * tap
        out = self.out_proj(cg * y)
        present = None
        if use_cache:
            present = (full[:, -(k - 1):].contiguous()
                       .as_subclass(StateTensor),)
        return out, present
