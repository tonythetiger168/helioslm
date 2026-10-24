"""Gated Delta-Rule Linear Attention (v5.5) - simplified Gated Delta Rule

Hybrid-interleave companion to MLA (DeepSeek-V3 style): layers that are not
full-attention layers run a linear-time recurrent attention whose "cache" is
a FIXED-SIZE state tensor ``S`` of shape [B, H, Dk, Dv] instead of a
growing K/V sequence.

Math (per head h, per token t):
    k_t = L2_normalize(W_k x_t)          q_t = W_q x_t / sqrt(Dk)
    decay_t = sigmoid(W_g x_t + b_g)     in (0, 1); per-head scalar, or a
                                         per-CHANNEL [Dk] vector when
                                         hybrid_attention.per_channel_decay
                                         is set (v5.21, KDA direction)
    r_t   = k_t . S_{t-1}                (retrieval, [B, H, Dv])
    S_t   = decay_t * S_{t-1} + k_t (x) (v_t - r_t)     (delta-rule write)
    o_t   = q_t . S_t                    (readout from the UPDATED state)

Cache contract (cross-module, relied upon by MTP rollback / the engine):
  - The cache tuple is ``(state,)`` where ``state`` is a ``StateTensor``
    ([B, H, Dk, Dv]) carrying the marker attribute
    ``_is_recurrent_state = True``.
  - StateTensors are NOT sliceable along dim 2 (there is no sequence axis);
    downstream code must check ``getattr(t, "_is_recurrent_state", False)``
    before applying the dim-2 truncation used for MLA caches. Rollback for a
    state cache = restore a clone taken BEFORE the speculative segment
    (plus replay of the committed prefix; see inference/mtp.py).
  - torch.cat along the batch dim, row slicing, .clone() and .contiguous()
    all preserve the subclass marker (verified in the test suite), so the
    engine/MTP batch helpers work unchanged.

Training path: a sequential loop over T (identical op order to the decode
path, so both are numerically identical — verified at atol 1e-5). This is
O(T) sequential. v5.107: ``hybrid_attention.chunk_size`` > 0 switches
multi-token forwards to ``chunked_delta_rule`` (T / chunk steps, each a
triangular solve), equal to the loop to float tolerance.

Numerical stability: the decay gate is a sigmoid (so 0 < decay < 1 by
construction) additionally clamped to [m, 1 - m] with a dtype-aware margin
m = max(1e-6, finfo(dtype).eps) — a fixed 1 - 1e-6 would round to exactly
1.0 in fp16/bf16, silently allowing a non-contracting state; the gate bias
is initialized to +4 (decay ~= 0.982) so the
state starts near a slow-decay accumulator. The bias is a raw
nn.Parameter (not a Linear bias) so the model-wide Linear init
(std=0.02 / zero bias) does not clobber it.

Packed-sequence support (v5.6): unlike MLA, the recurrence cannot segment
its fixed-size state at document boundaries — but it does not need to.
When ``position_ids`` restart mid-sequence (packed training's per-document
positions), the recurrent state is ZEROED at each document boundary before
that document's first token is processed, so documents never leak into each
other. The packed layout is then exactly equivalent to running each
document as its own sequence (verified in the test suite, mirroring the
MLA packed-isolation test). 1D ``[seq]`` position_ids follow the shared
layer contract (same resolution as MLA's ``_resolve_positions``): they are
unsqueezed to ``[1, seq]`` so packed detection and boundary validation
apply uniformly. A document start while a non-empty ``past_key_value``
is supplied (cached decode at a document boundary) is still a loud
``ValueError``: continuing from a carried-over state and resetting it at
the same time is contradictory. Explicit positions supplied against a
carried state must CONTINUE the cache — strictly increasing, starting at
or past the cached length (e.g. chunked-prefill ``past_len..past_len+seq-1``)
— never restart at 0; the recurrence has no access to the cached length
(the state carries no sequence axis), so any restart/fresh-document layout
is rejected outright rather than risk leaking the old document's state.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTensor(torch.Tensor):
    """Marker subclass of torch.Tensor for fixed-size recurrent state.

    Any tensor carrying ``_is_recurrent_state = True`` is a state cache:
    fixed shape [B, H, Dk, Dv], no sequence axis. Downstream cache plumbing
    (MTP rollback, vLLM engine batching) uses the marker to distinguish it
    from dim-2-sequence MLA caches. Plain Tensor subclasses preserve their
    type through torch.cat / slicing / clone / contiguous, so generic
    batch-dim helpers keep working.
    """

    _is_recurrent_state = True


def is_recurrent_state(t) -> bool:
    """Duck-typed marker check (avoids importing StateTensor everywhere)."""
    return bool(getattr(t, "_is_recurrent_state", False))


def past_seq_len(past_key_values) -> int:
    """Sequence length covered by a per-layer cache list (0 for None).

    Shared by ``EvastarV5.forward`` and ``MTPDecoder`` (m11: single source
    of truth). Recurrent-state tensors (fixed shape, no sequence axis) and
    non-tensor entries are skipped; a sequence cache tensor must have
    ``dim() >= 3`` with the sequence axis at dim 2. Every layer is
    scanned, and config validation guarantees at least one MLA layer
    (even with ``first_layer_full=False``, v5.57), so a sequence-length
    tensor always exists in a hybrid cache.
    """
    if past_key_values is None:
        return 0
    for layer_past in past_key_values:
        for t in layer_past:
            if torch.is_tensor(t) and t.dim() >= 3 and not is_recurrent_state(t):
                return t.shape[2]
    return 0


def chunked_delta_rule(q, k, v, decay, keep, seg_start, state, chunk):
    """Chunked-parallel form of the recurrence in the module docstring
    (v5.107, ``hybrid_attention.chunk_size``); same results as the
    per-token loop up to float rounding.

    Shapes: q, k, v [B, H, T, D]; decay [B, H, T] (per head) or
    [B, H, T, Dk] (per channel); keep [B, 1, T] or None (masked tokens:
    decay already 1, write strength 0); seg_start [B, T] bool or None;
    state [B, H, Dk, Dv]. Returns (out [B, H, T, Dv], new state).

    With the write ``u_t = keep_t (v_t - S_{t-1}^T k_t)`` the state after
    token t of a chunk is ``S_t = Lam_t S_0 + sum_{j<=t} Lam_{t,j} k_j u_j^T``
    where ``Lam_{t,j} = diag(prod_{i=j+1..t} decay_i)``. Substituting into
    the retrieval gives a unit lower-triangular system for the u of a
    chunk, ``(I + diag(keep) A) u = keep (v - (k * Lam_{t-1}) S_0)`` with
    ``A_tj = k_t^T Lam_{t-1,j} k_j`` (j < t), solved with one triangular
    solve instead of t sequential steps. Every decay product is formed
    as ``exp`` of a difference of cumulative log-decays over j <= t, so
    all factors are <= 1 (no division by small decays). A document
    boundary (seg_start) zeroes every pair (t, j) across it, and the
    carried state only for tokens still in the chunk's first document.
    Computed in fp32 (fp64 for fp64 inputs) whatever the input dtype.
    """
    out_dtype, state_dtype = q.dtype, state.dtype
    ct = torch.promote_types(q.dtype, torch.float32)   # fp64 stays fp64
    q, k, v, state = q.to(ct), k.to(ct), v.to(ct), state.to(ct)
    B, H, T, _ = q.shape
    per_channel = decay.dim() == 4
    g = torch.log(decay.to(ct))
    if not per_channel:
        g = g[..., None]                                     # [B, H, T, 1]
    keep = keep.to(ct) if keep is not None else None
    seg = torch.cumsum(seg_start.long(), 1) if seg_start is not None else None
    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        n = c1 - c0
        qc, kc, vc = q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1]
        G = g[:, :, c0:c1].cumsum(2)                        # log Lam_t
        Gp = torch.cat([torch.zeros_like(G[:, :, :1]), G[:, :, :-1]], 2)
        lower = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        strict = lower.tril(-1)
        same = inh = None
        if seg is not None:
            sc = seg[:, c0:c1]
            prev = seg[:, c0 - 1] if c0 > 0 else torch.zeros_like(sc[:, 0])
            same = (sc[:, :, None] == sc[:, None, :])[:, None]   # [B, 1, n, n]
            inh = (sc == prev[:, None])[:, None, :, None].to(ct)   # [B, 1, n, 1]

        def pair(Ga, mask):
            # exp(Ga_t - G_j) on the masked pairs, 0 elsewhere (masked to
            # -inf BEFORE exp, so no overflow from j > t).
            m = mask if same is None else mask & same
            d = Ga[:, :, :, None, :] - G[:, :, None, :, :]          # [B, H, n, n, C]
            return torch.exp(d.masked_fill(~m[..., None], float("-inf")))

        Pk, Pq = pair(Gp, strict), pair(G, lower)
        if per_channel:
            A = torch.einsum("bhtc,bhjc,bhtjc->bhtj", kc, kc, Pk)
            Qm = torch.einsum("bhtc,bhjc,bhtjc->bhtj", qc, kc, Pq)
        else:
            A = (kc @ kc.transpose(-1, -2)) * Pk[..., 0]
            Qm = (qc @ kc.transpose(-1, -2)) * Pq[..., 0]
        kdec, qdec = kc * torch.exp(Gp), qc * torch.exp(G)
        if inh is not None:
            kdec, qdec = kdec * inh, qdec * inh
        rhs = vc - kdec @ state                                     # [B, H, n, Dv]
        if keep is not None:
            kp = keep[:, :, c0:c1, None]
            rhs, A = rhs * kp, A * kp
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        u = torch.linalg.solve_triangular(eye + A, rhs, upper=False,
                                          unitriangular=True)
        outs.append(qdec @ state + Qm @ u)
        w = torch.exp(G[:, :, -1:] - G)                             # Lam_{n-1,j}
        carry = torch.exp(G[:, :, -1])[..., None]                    # [B, H, C, 1]
        if same is not None:
            w = w * same[:, :, -1, :, None].to(ct)
            carry = carry * inh[:, :, -1:, :]
        state = carry * state + (kc * w).transpose(-1, -2) @ u
    return torch.cat(outs, 2).to(out_dtype), state.to(state_dtype)


class GatedDeltaAttention(nn.Module):
    """Simplified Gated Delta Rule linear attention (see module docstring).

    Mirrors the MLA forward signature so ``HeliosLMv5Layer`` can hold either
    module: forward(hidden_states, attention_mask=None, past_key_value=None,
    use_cache=False, position_ids=None) -> (output [B, seq, hidden], present).
    ``position_ids`` needs no positional encoding for the recurrence; it is
    used ONLY for packed-sequence handling — a position restart marks a
    document boundary at which the recurrent state is reset to zero (see the
    module docstring). Normal prefill positions (0..T-1) and decode positions
    (past_len.., strictly increasing) are accepted untouched; a 1D ``[seq]``
    layout is accepted as a single row (shared layer contract, same as MLA).

    ``is_recurrent_attention = True`` lets the model/engine/MTP recognize
    the layer type without importing this module.
    """

    is_recurrent_attention = True

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        hc = config.hybrid_attention
        self.num_heads = hc.linear_num_heads
        self.head_dim = hc.linear_head_dim  # Dk == Dv in this simplified variant

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # KDA-style per-channel decay (v5.21, config
        # .hybrid_attention.per_channel_decay): when True the gate emits one
        # sigmoid per head per KEY CHANNEL (decay becomes a [Dk] vector per
        # head, so state rows forget at independent rates — Kimi Linear's
        # fine-grained eraser, as used by GLM-5.3-Flash); when False the
        # v5.5 per-head scalar gate is kept (bit-identical).
        self.per_channel_decay = bool(
            config.hybrid_attention.per_channel_decay)
        gate_width = self.num_heads * self.head_dim \
            if self.per_channel_decay else self.num_heads
        # Bias-free Linear + a raw bias parameter initialized to +4 so
        # decay ~= sigmoid(4) ~= 0.982 at init. A raw Parameter is used
        # because EvastarV5._init_weights zeroes every Linear bias.
        self.g_proj = nn.Linear(self.hidden_size, gate_width, bias=False)
        self.decay_bias = nn.Parameter(
            torch.full((gate_width,), 4.0))

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_scale = 1.0 / math.sqrt(self.head_dim)
        # v5.107: chunked-parallel recurrence for multi-token forwards.
        self.chunk_size = hc.chunk_size

        # v5.63 (roadmap Q1, Qwen3-Next Gated DeltaNet): causal depthwise
        # short convolution over the q/k/v projections, followed by SiLU.
        # 0 (default) = off: no parameters, cache stays (state,) and the
        # forward is bit-identical. When on, the cache becomes
        # (state, conv_state) where conv_state holds the last K-1 PRE-conv
        # projections [B, K-1, 3*H*D]; it is a StateTensor too (no sequence
        # axis), so MTP rollback and engine batching treat it like the state.
        self.conv_kernel = int(getattr(hc, "short_conv_kernel", 0) or 0)
        if self.conv_kernel:
            ch = 3 * self.num_heads * self.head_dim
            w = torch.zeros(ch, self.conv_kernel)
            w[:, -1] = 1.0           # identity taps at init (then SiLU)
            self.conv_weight = nn.Parameter(w)

    def decay_gate(self, hidden_states):
        """Per-token decay in the open interval (0, 1).

        Returns [B, seq, H] (per-head scalar, v5.5) or [B, seq, H, Dk]
        (per-channel, v5.21 KDA direction). Sigmoid output is clamped away
        from exact 0/1 (fp saturation), which also keeps the state
        contraction strictly below 1. The margin is dtype-aware (m6):
        m = max(1e-6, finfo(dtype).eps) — in fp16/bf16 a fixed 1 - 1e-6
        upper bound would round to exactly 1.0; fp32 keeps the historical
        [1e-6, 1 - 1e-6] bounds.
        """
        g = torch.sigmoid(self.g_proj(hidden_states) + self.decay_bias)
        if self.per_channel_decay:
            B, seq, _ = g.shape
            g = g.view(B, seq, self.num_heads, self.head_dim)
        m = max(1e-6, torch.finfo(g.dtype).eps)
        return g.clamp(min=m, max=1.0 - m)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None):
        """
        Args:
            hidden_states: [B, seq, hidden] — only the NEW tokens when decoding.
            attention_mask: optional [B, total_len] (1 = real, 0 = pad). Only
                the last ``seq`` columns (current tokens) are used: masked
                tokens perform NO state write (decay forced to 1, delta zeroed)
                so padding never pollutes the recurrent state. Outputs at
                masked positions are don't-care, matching the MLA convention.
            past_key_value: optional (state,) tuple with state [B, H, Dk, Dv].
            use_cache: return the updated state.
            position_ids: used only for packed-sequence handling (a
                position restart resets the recurrent state to zero, marking
                a new document; see module docstring).
        Returns:
            (output [B, seq, hidden], (state,) or None)
        """
        B, seq, _ = hidden_states.shape
        if seq == 0:
            # m7: fail loudly instead of torch.stack([])'s cryptic error.
            raise ValueError(
                "GatedDeltaAttention received an empty sequence (seq_len=0); "
                "the recurrence requires at least one token"
            )

        # Packed-sequence handling (v5.6): a position restart marks a
        # document boundary. The fixed-size state cannot be segmented, so
        # it is ZEROED at each boundary instead — the packed layout becomes
        # exactly equivalent to running each document as its own sequence.
        # 1D [seq] position_ids follow the shared layer contract (same
        # resolution as MLA._resolve_positions): unsqueeze to [1, seq] so
        # packed detection and boundary validation apply uniformly.
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, seq)
        seg_start = None
        if position_ids is not None and position_ids.dim() == 2 \
                and position_ids.shape[1] == seq:
            if past_key_value is not None:
                # The carried state covers at least one token (a state
                # cache only exists after a forward processed tokens), so a
                # legitimate continuation — a cached decode step or chunked
                # prefill — is strictly increasing and starts at or past
                # the cached length. A position restart (a decreasing step,
                # or a fresh document starting at 0) would silently leak the
                # old document's state into the new one; MLA rejects the
                # same layouts in _build_attn_mask. The recurrence cannot
                # recover the exact cached length (the state has no
                # sequence axis), so any restart is rejected outright.
                restart = (position_ids[:, 1:] < position_ids[:, :-1])
                if (seq > 1 and bool(restart.any())) \
                        or bool((position_ids[:, 0] == 0).any()):
                    raise ValueError(
                        "GatedDeltaAttention: position_ids do not continue "
                        "the cached recurrent state (a non-empty "
                        "past_key_value is supplied): positions must be "
                        "strictly increasing and start at or past the "
                        "cached length (chunked-prefill continuation, e.g. "
                        "past_len..past_len+seq-1, is fine), but a "
                        "position restart starting a new document was "
                        "given. Continuing from the carried-over state "
                        "would leak the old document into the new one; "
                        "prefill each document separately."
                    )
            elif seq > 1:
                restart = (position_ids[:, 1:] < position_ids[:, :-1])  # [B, seq-1]
                if bool(restart.any()):
                    seg_start = torch.zeros(B, seq, dtype=torch.bool,
                                            device=hidden_states.device)
                    seg_start[:, 0] = True
                    seg_start[:, 1:] = restart
                    # Defensive: a true restart always has pos[t] < pos[t-1] but
                    # segment starts may also begin at 0; both are resets.

        conv_state = None
        if past_key_value is not None:
            want = 2 if self.conv_kernel else 1
            if len(past_key_value) != want:
                raise ValueError(
                    "GatedDeltaAttention expects a "
                    f"{'(state, conv_state)' if self.conv_kernel else '(state,)'} "
                    f"cache tuple of {want} tensor(s), got {len(past_key_value)}"
                )
            state = past_key_value[0]
            if self.conv_kernel:
                conv_state = past_key_value[1].as_subclass(torch.Tensor)
            if state.shape != (B, self.num_heads, self.head_dim, self.head_dim):
                raise ValueError(
                    f"recurrent state shape {tuple(state.shape)} != expected "
                    f"{(B, self.num_heads, self.head_dim, self.head_dim)}"
                )
        else:
            state = hidden_states.new_zeros(
                B, self.num_heads, self.head_dim, self.head_dim
            )

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        new_conv_state = None
        if self.conv_kernel:
            q, k, v, new_conv_state = self._short_conv(
                q, k, v, conv_state, attention_mask, seg_start)
        q = q.view(B, seq, self.num_heads, self.head_dim)
        k = k.view(B, seq, self.num_heads, self.head_dim)
        v = v.view(B, seq, self.num_heads, self.head_dim)
        # k is L2-normalized (delta rule); q is scaled like attention scores.
        k = F.normalize(k.transpose(1, 2), dim=-1)          # [B, H, seq, Dk]
        q = q.transpose(1, 2) * self.q_scale                # [B, H, seq, Dk]
        v = v.transpose(1, 2)                               # [B, H, seq, Dv]
        decay = self.decay_gate(hidden_states).transpose(1, 2)
        # [B, H, seq] (per-head) or [B, H, seq, Dk] (per-channel)

        if attention_mask is not None:
            keep = attention_mask[:, -seq:].to(decay.dtype)  # [B, seq]
            keep = keep[:, None, :]                          # [B, 1, seq]
            # Masked token: decay -> 1 (no forgetting) and no write; the
            # state passes through unchanged.
            if self.per_channel_decay:
                decay = 1.0 + (decay - 1.0) * keep[..., None]
            else:
                decay = 1.0 + (decay - 1.0) * keep
        else:
            keep = None

        # Sequential recurrence by default: training and decode share this
        # exact op order, so a one-shot forward and token-by-token decode
        # agree numerically to float-reassociation noise (asserted at atol
        # 1e-5 in the test suite). At a packed-sequence document boundary
        # the state is zeroed before the boundary token is processed. With
        # chunk_size set, multi-token forwards use the chunked-parallel
        # form instead (same math, see chunked_delta_rule).
        if self.chunk_size and seq > 1:
            out, state = chunked_delta_rule(
                q, k, v, decay, keep, seg_start,
                state.as_subclass(torch.Tensor), self.chunk_size)
        else:
            out, state = self._sequential(q, k, v, decay, keep, seg_start, state)

        # Subclass hygiene: when the incoming state was a StateTensor, plain
        # torch ops keep the subclass (Tensor subclasses are "sticky"), so
        # `out` — and then the residual stream, other layers' MLA caches,
        # everything downstream — would silently carry the
        # _is_recurrent_state marker. Only the returned cache tensor may be
        # marked; drop the subclass at the module boundary.
        out = out.as_subclass(torch.Tensor)
        output = self.o_proj(out.transpose(1, 2).contiguous().view(B, seq, -1))

        present = None
        if use_cache:
            present = (state.as_subclass(StateTensor),)
            if self.conv_kernel:
                present = present + (new_conv_state.as_subclass(StateTensor),)
        return output, present

    def _sequential(self, q, k, v, decay, keep, seg_start, state):
        """The per-token recurrence (default; always used for decode)."""
        B = q.shape[0]
        seq = q.shape[2]
        outs = []
        for t in range(seq):
            if seg_start is not None and bool(seg_start[:, t].any()):
                state = state * (~seg_start[:, t]).view(B, 1, 1, 1).to(state.dtype)
            k_t = k[:, :, t]                                  # [B, H, Dk]
            v_t = v[:, :, t]                                  # [B, H, Dv]
            r_t = torch.einsum("bhk,bhkv->bhv", k_t, state)   # retrieval
            delta = v_t - r_t
            if keep is not None:
                delta = delta * keep[:, :, t, None]
            d_t = decay[:, :, t, None, None] if not self.per_channel_decay \
                else decay[:, :, t, :, None]  # [B,H,1,1] or [B,H,Dk,1]
            state = d_t * state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            o_t = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
            outs.append(o_t)
        out = torch.stack(outs, dim=2)                        # [B, H, seq, Dv]

        return out, state

    def _short_conv(self, q, k, v, conv_state, attention_mask, seg_start):
        """Causal depthwise conv (kernel K) + SiLU over [q|k|v] channels.

        Padding tokens (attention_mask 0) contribute zeros, and taps never
        cross a packed-sequence document boundary, so the conv obeys the
        same isolation rules as the recurrence. Returns the convolved
        q, k, v and the new conv_state (last K-1 pre-conv inputs).
        """
        B, seq, d = q.shape
        K = self.conv_kernel
        x = torch.cat([q, k, v], dim=-1)                      # [B, seq, 3d]
        if attention_mask is not None:
            x = x * attention_mask[:, -seq:, None].to(x.dtype)
        if conv_state is None:
            conv_state = x.new_zeros(B, K - 1, x.shape[-1])
        xp = torch.cat([conv_state.to(x.dtype), x], dim=1)    # [B, K-1+seq, 3d]
        seg = None
        if seg_start is not None:
            seg = torch.cumsum(seg_start.long(), dim=1)       # [B, seq]
        w = self.conv_weight.to(x.dtype)
        y = torch.zeros_like(x)
        for j in range(K):                                    # tap j = lag j
            tap = xp[:, K - 1 - j:K - 1 - j + seq]            # x[t - j]
            if seg is not None and j > 0:
                same = torch.zeros_like(seg, dtype=torch.bool)
                same[:, j:] = seg[:, j:] == seg[:, :-j]
                tap = tap * same[..., None].to(x.dtype)
            y = y + tap * w[:, K - 1 - j]
        y = F.silu(y)
        new_state = xp[:, -(K - 1):] if K > 1 else xp[:, :0]
        return y[..., :d], y[..., d:2 * d], y[..., 2 * d:], new_state
