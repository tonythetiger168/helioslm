"""HeliosLM v5.31 — compressed-attention references (DS-V4 direction).

Two toy-scale, causality-exact reference modules distilled from
DeepSeek-V4's attention redesign. Deviations from V4 recorded (not hidden):

- V4 uses CSA (4x KV compression + FP4 lightning indexer top-k)
  interleaved with HCA (128x compression dense). We implement the HCA
  mechanism (chunk-pooled memory slots + local window) and a CSA-style
  indexer, but with plain bf16/fp32 linear projections instead of FP4,
  and mean-pooling instead of learned compressors. The *structural*
  properties — compression ratio, top-k selection, causality — are the
  reference contribution; the quant/precision engineering is not the
  point at 8.5M params.

Both modules are CAUSAL EXACT:
- within the current chunk: full causal attention;
- across chunks: queries see only slots of PAST chunks (slot c pools
  chunk c's K/V; queries in chunk c never see slot c);
- CSA indexer: top-k is taken over past positions only.
"""
import math

import torch
import torch.nn as nn


class CompressedGlobalAttention(nn.Module):
    """HCA-style: chunked local attention + compressed global memory.

    Sequence is split into chunks of `window`; each past chunk contributes
    one mean-pooled K/V memory slot. Query i (in chunk c) attends to:
    slots of chunks < c (global memory) and keys of chunk c up to i
    (local, causal).
    """

    def __init__(self, dim: int, n_heads: int = 2, window: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.window = n_heads, window
        self.dh = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        """x: (B, L, D). attention_mask: (B, L) with 1 = real, 0 = pad
        (padding is excluded from pooling and from local keys). Returns
        (B, L, D)."""
        B, L, D = x.shape
        w, H, dh = self.window, self.n_heads, self.dh
        C = (L + w - 1) // w                     # number of chunks
        pad = C * w - L
        if pad:
            x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)
            if attention_mask is None:
                attention_mask = torch.cat(
                    [x.new_ones(B, L, dtype=torch.long),
                     x.new_zeros(B, pad, dtype=torch.long)], dim=1)
        qkv = self.qkv(x).view(B, C * w, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)              # (B, C*w, H, dh)
        if attention_mask is None:
            attention_mask = torch.ones(B, C * w, dtype=torch.long,
                                        device=x.device)
        m = attention_mask.view(B, C, w, 1, 1).to(q.dtype)  # (B,C,w,1,1)

        # --- memory slots: mean-pooled K/V per chunk (past chunks only) ---
        kc = (k.view(B, C, w, H, dh) * m).sum(2) / m.sum(2).clamp(min=1.0)
        vc = (v.view(B, C, w, H, dh) * m).sum(2) / m.sum(2).clamp(min=1.0)
        # cumulative slots: slot_c must pool only chunks < c for queries in c
        # cumsum includes chunk c itself -> shift right by one chunk.
        k_slots = torch.cumsum(kc, dim=1) - kc   # (B, C, H, dh), exclusive
        v_slots = torch.cumsum(vc, dim=1) - vc
        cnt = torch.arange(C, device=x.device).view(1, C, 1, 1)
        k_slots = k_slots / cnt.clamp(min=1)
        v_slots = v_slots / cnt.clamp(min=1)

        # --- global memory scores: (B, C, H, w_q, C_slots) ---
        s_mem = torch.einsum("bcwhd,bshd->bchws",
                             q.view(B, C, w, H, dh),
                             k_slots) / math.sqrt(dh)
        # k_slots[s] pools chunks < s (exclusive cumsum), so a query in
        # chunk c may attend slots 1 <= s <= c (s=0 is the empty pool)
        s_idx = torch.arange(C, device=x.device)
        valid = (s_idx.view(1, 1, 1, 1, C) >= 1) & \
                (s_idx.view(1, 1, 1, 1, C) <= s_idx.view(1, C, 1, 1, 1))
        s_mem = s_mem.masked_fill(~valid, float("-inf"))

        # --- local causal scores within each chunk: (B, C, H, w_q, w_z) ---
        s_loc = torch.einsum("bcwhd,bczhd->bchwz",
                             q.view(B, C, w, H, dh),
                             k.view(B, C, w, H, dh)) / math.sqrt(dh)
        # causal mask over (q-in-chunk, k-in-chunk)
        causal = torch.tril(torch.ones(w, w, device=x.device, dtype=torch.bool))
        s_loc = s_loc.masked_fill(~causal.view(1, 1, 1, w, w), float("-inf"))
        # pad mask over keys: m (B,C,w_k,1,1) -> (B,C,1,1,w_k)
        s_loc = s_loc.masked_fill(m.transpose(2, 3).view(B, C, 1, 1, w) < 0.5,
                                  float("-inf"))

        s = torch.cat([s_mem, s_loc], dim=-1)      # (B, C, H, w_q, C+w)
        p = torch.softmax(s, dim=-1)
        p_mem, p_loc = p.split([C, w], dim=-1)     # (B, C, H, w_q, .) each
        o_mem = torch.einsum("bchws,bshd->bchwd", p_mem, v_slots)
        o_loc = torch.einsum("bchqz,bczhd->bchqd", p_loc,
                             v.view(B, C, w, H, dh))
        # (B, C, H, w, dh) -> permute to (B, C, w, H, dh) BEFORE merging C*w
        o = (o_mem + o_loc).permute(0, 1, 3, 2, 4).reshape(B, C * w, H, dh)
        o = o.reshape(B, C * w, D)
        o = self.out(o)
        if pad:
            o = o[:, :L]
        return o


class IndexedSparseAttention(nn.Module):
    """CSA-style: cheap indexer head selects top-k global tokens.

    Each query computes indexer scores over PAST positions via a tiny
    scorer s_j = u . tanh(W k_j) (V4's FP4 lightning indexer simulated
    by a linear head — deviation recorded). Attention then runs over:
    the local window (always) + the top-k indexer-selected past tokens.
    Causality: selection set depends only on past positions.
    """

    def __init__(self, dim: int, n_heads: int = 2, window: int = 8,
                 topk: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.window, self.topk = n_heads, window, topk
        self.dh = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.indexer = nn.Linear(dim, dim)   # simulates the lightning head

    def forward(self, x, attention_mask=None):
        B, L, D = x.shape
        H, dh, w, kmax = self.n_heads, self.dh, self.window, self.topk
        qkv = self.qkv(x).view(B, L, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)              # (B, L, H, dh)
        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.long, device=x.device)
        pos = torch.arange(L, device=x.device)

        # indexer scores per position (content-only; causal use below)
        idx = self.indexer(x).view(B, L, H, dh)
        idx_score = (idx * k).sum(-1) / math.sqrt(dh)   # (B, L, H)

        outs = []
        for b in range(B):
            ob = []
            for i in range(L):
                if attention_mask[b, i] == 0:
                    ob.append(q.new_zeros(H, dh))
                    continue
                lo = max(0, i - w + 1)
                cand = [j for j in range(lo, i + 1)]
                past = [j for j in range(0, lo)]
                if past and kmax > 0:
                    sc = idx_score[b, past]                  # (P, H)
                    # per-head selection, union of heads' picks (reference
                    # simplification: shared selection via head-0 scores)
                    order = sc[:, 0].argsort(descending=True)[:kmax]
                    cand = [past[t] for t in order.tolist()] + cand
                kk = k[b, cand]                              # (S, H, dh)
                vv = v[b, cand]
                s = torch.einsum("hd,shd->hs", q[b, i], kk) / math.sqrt(dh)
                keep = [attention_mask[b, j] > 0 for j in cand]
                s = s.masked_fill(~torch.tensor(keep, device=x.device)
                                  .view(1, -1), float("-inf"))
                p = torch.softmax(s, dim=-1)
                ob.append(torch.einsum("hs,shd->hd", p, vv))
            outs.append(torch.stack(ob))                     # (L, H, dh)
        o = torch.stack(outs).reshape(B, L, D)
        return self.out(o)
