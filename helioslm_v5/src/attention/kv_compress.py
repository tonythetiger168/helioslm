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


class HighCompressionAttention(nn.Module):
    """HCA at production compression ratio (B1-2): chunk-pooled
    memory slots at `chunk`x KV compression + trailing local window +
    sparse top-k RESCUE of exact tokens from the compressed region.

    Motivation: CompressedGlobalAttention above is the HCA mechanism at
    toy ratio (one slot per `window`=8 tokens). DeepSeek-V4's HCA runs
    ~128x compression, where mean-pooled slots destroy exact per-token
    information (a 4-digit code, a name). V4's answer is that HCA layers
    are INTERLEAVED with CSA layers whose indexer keeps exact top-k
    access; this module instead attaches the rescue directly to the HCA
    layer — one slot budget, one rescue budget, one ablation knob — so
    the quality breakpoint of high compression can be measured with and
    without rescue in a single module.

    Attention set for query i in chunk c = i // chunk:
      - LOCAL:  positions max(0, i - window + 1) .. i  (exact, causal);
      - SLOTS:  one mean-pooled K/V slot per chunk < c  (exclusive —
                queries in chunk c never see slot c, same contract as
                CompressedGlobalAttention);
      - RESCUE: top-`rescue_topk` exact tokens from positions
                j <= i - window AND j < c * chunk (i.e. strictly inside
                the compressed region, outside the local window — no
                double-counting), selected by a cheap indexer.

    The rescue indexer is a CSA-style linear scorer: idx = W_idx x,
    score(i, j) = mean_h (idx_i . k_j) / sqrt(dh), with the selection
    SHARED across heads (head-mean — the documented simplification, in
    the same spirit as IndexedSparseAttention's head-0-shared selection
    and MLA's "head_mean" free indexer). `rescue_topk=0` removes the
    rescue path entirely (pure HCA at `chunk`x compression).

    CAUSAL EXACT, same non-negotiable as the v5.31 modules: every term a
    query i consumes (the local band, slots of chunks < c, rescue picks
    with j <= i - window < i) depends only on positions <= i, and the
    rescue SELECTION scores only positions inside strictly-past chunks,
    so perturbing position p cannot change outputs at positions < p.

    Deviations from DeepSeek-V4 (recorded, not hidden): mean-pooling
    instead of learned compressors; an fp32 linear indexer instead of an
    FP4 lightning indexer; rescue attached to the HCA layer instead of
    living in separate interleaved CSA layers. The structural properties
    — 128x-class compression, top-k exact rescue, causality — are the
    reference contribution.
    """

    def __init__(self, dim: int, n_heads: int = 2, window: int = 8,
                 chunk: int = 128, rescue_topk: int = 0):
        super().__init__()
        assert dim % n_heads == 0
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not isinstance(rescue_topk, int) or rescue_topk < 0:
            raise ValueError(
                f"rescue_topk must be a non-negative integer, got "
                f"{rescue_topk!r}")
        self.n_heads, self.window = n_heads, window
        self.chunk, self.rescue_topk = chunk, rescue_topk
        self.dh = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        if rescue_topk > 0:
            self.rescue_indexer = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        """x: (B, L, D). attention_mask: (B, L) with 1 = real, 0 = pad
        (padding is excluded from pooling, from the local band, and from
        rescue eligibility; padded query rows return finite zeros via
        the nan_to_num guard, same convention as the MLA absorbed path).
        Returns (B, L, D)."""
        B, L, D = x.shape
        H, dh, W, S, R = self.n_heads, self.dh, self.window, \
            self.chunk, self.rescue_topk
        C = (L + S - 1) // S                     # number of chunks
        pad = C * S - L
        if pad:
            x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask,
                     x.new_zeros(B, pad, dtype=attention_mask.dtype)],
                    dim=1)
        Lp = C * S
        qkv = self.qkv(x).view(B, Lp, 3, H, dh)
        q, k, v = qkv.unbind(dim=2)              # (B, Lp, H, dh)
        if attention_mask is None:
            attention_mask = torch.ones(B, Lp, dtype=torch.long,
                                        device=x.device)
        key_ok = attention_mask.bool()           # (B, Lp)

        # --- memory slots: exclusive cumsum mean-pool per chunk -------
        m = attention_mask.view(B, C, S, 1, 1).to(q.dtype)
        kc = (k.view(B, C, S, H, dh) * m).sum(2) / m.sum(2).clamp(min=1.0)
        vc = (v.view(B, C, S, H, dh) * m).sum(2) / m.sum(2).clamp(min=1.0)
        k_slots = torch.cumsum(kc, dim=1) - kc   # (B, C, H, dh), exclusive
        v_slots = torch.cumsum(vc, dim=1) - vc
        cnt = torch.arange(C, device=x.device).view(1, C, 1, 1)
        k_slots = (k_slots / cnt.clamp(min=1)).transpose(1, 2)  # (B,H,C,dh)
        v_slots = (v_slots / cnt.clamp(min=1)).transpose(1, 2)

        qh = q.transpose(1, 2)                   # (B, H, Lp, dh)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(dh)
        pos = torch.arange(Lp, device=x.device)
        q_chunk = pos // S                       # (Lp,)
        i_j = pos.view(Lp, 1) - pos.view(1, Lp)  # (Lp, Lp) distance i-j

        # --- slot scores: (B, H, Lp, C) --------------------------------
        s_mem = torch.matmul(qh, k_slots.transpose(-2, -1)) * scale
        # slot s pools chunks < s (exclusive cumsum), so a query in chunk
        # c may attend slots 1 <= s <= c (slot 0 is the empty pool).
        s_idx = torch.arange(C, device=x.device)
        valid_slot = (s_idx.view(1, C) >= 1) & \
            (s_idx.view(1, C) <= q_chunk.view(Lp, 1))            # (Lp, C)
        s_mem = s_mem.masked_fill(~valid_slot.view(1, 1, Lp, C),
                                  float("-inf"))

        # --- exact-token scores over the local band: (B, H, Lp, Lp) ---
        s_tok = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        local_band = (i_j >= 0) & (i_j < W)      # j in [i-W+1, i]
        keep_loc = local_band.view(1, 1, Lp, Lp) & key_ok.view(B, 1, 1, Lp)
        s_loc = s_tok.masked_fill(~keep_loc, float("-inf"))

        terms = [s_mem, s_loc]

        # --- rescue: top-R exact tokens from the compressed region ----
        if R > 0:
            idx = self.rescue_indexer(x).view(B, Lp, H, dh)
            s_idx_q = torch.matmul(idx.transpose(1, 2),
                                   kh.transpose(-2, -1)) * scale
            s_idx_q = s_idx_q.mean(1)            # (B, Lp, Lp) head-mean
            eligible = (i_j >= W) & \
                (pos.view(1, Lp) < (q_chunk * S).view(Lp, 1))
            eligible = eligible.view(1, Lp, Lp) & key_ok.view(B, 1, Lp)
            s_res_all = s_idx_q.masked_fill(~eligible, float("-inf"))
            r_eff = min(R, Lp)
            top = torch.topk(s_res_all, r_eff, dim=-1)   # (B, Lp, r_eff)
            r_idx = top.indices                          # deterministic
            s_res = top.values.view(B, 1, Lp, r_eff) \
                .expand(B, H, Lp, r_eff)
            # ineligible picks carry -inf scores -> exactly 0 weight
            gidx = r_idx.view(B, 1, Lp, r_eff, 1) \
                .expand(B, H, Lp, r_eff, dh)
            v_res = torch.gather(
                vh.unsqueeze(2).expand(B, H, Lp, Lp, dh), 3, gidx)
            terms.append(s_res)

        s = torch.cat(terms, dim=-1)              # (B,H,Lp,C+Lp[+R])
        p = torch.nan_to_num(torch.softmax(s, dim=-1))
        p_mem = p[..., :C]
        p_loc = p[..., C:C + Lp]
        o = torch.matmul(p_mem, v_slots) + torch.matmul(p_loc, vh)
        if R > 0:
            p_res = p[..., C + Lp:]
            o = o + (p_res.unsqueeze(-1) * v_res).sum(3)
        o = o.transpose(1, 2).reshape(B, Lp, D)
        o = self.out(o)
        if pad:
            o = o[:, :L]
        return o

    def get_compression_stats(self, seq_len: int) -> dict:
        """Truthful per-token state accounting (values, not bytes — same
        convention as MLA.get_kv_cache_size; no unmeasured speed claims).

        At sequence length L, per-token STORED state:
          dense reference: 2 * H * dh values per token (full K and V);
          this module: 2 * H * dh / chunk amortized slot values per
          token for the compressed region, plus rescue_topk * 2 * H * dh
          values per QUERY for the rescued exact tokens (a selection
          budget, not a cache — reported separately).
        """
        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError(
                f"seq_len must be a positive integer, got {seq_len!r}")
        H, dh = self.n_heads, self.dh
        dense_per_token = 2 * H * dh
        slot_per_token = 2 * H * dh / self.chunk
        rescue_per_query = self.rescue_topk * 2 * H * dh
        return {
            "seq_len": seq_len,
            "chunk": self.chunk,
            "rescue_topk": self.rescue_topk,
            "dense_values_per_token": dense_per_token,
            "slot_values_per_token": slot_per_token,
            "rescue_values_per_query": rescue_per_query,
            "compression_vs_dense": dense_per_token / slot_per_token,
        }
