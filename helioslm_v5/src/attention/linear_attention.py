"""Gated Delta-Rule Linear Attention (v5.5) - simplified Gated Delta Rule

Hybrid-interleave companion to MLA (DeepSeek-V3 style): layers that are not
full-attention layers run a linear-time recurrent attention whose "cache" is
a FIXED-SIZE state tensor ``S`` of shape [B, H, Dk, Dv] instead of a
growing K/V sequence.

Math (per head h, per token t):
    k_t = L2_normalize(W_k x_t)          q_t = W_q x_t / sqrt(Dk)
    decay_t = sigmoid(W_g x_t + b_g)     in (0, 1), per-head scalar
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
O(T) sequential and intended for the lite config; a chunked-parallel scan
is a future optimization, deliberately out of scope here.

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
MLA packed-isolation test). A restart while a non-empty ``past_key_value``
is supplied (cached decode with a position restart) is still a loud
``ValueError``: continuing from a carried-over state and resetting it at
the same time is contradictory.
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

    Shared by ``HeliosLMv5.forward`` and ``MTPDecoder`` (m11: single source
    of truth). Recurrent-state tensors (fixed shape, no sequence axis) and
    non-tensor entries are skipped; a sequence cache tensor must have
    ``dim() >= 3`` with the sequence axis at dim 2. Under the hybrid
    interleave rule layer 0 is always MLA, so a sequence-length tensor
    always exists in a hybrid cache.
    """
    if past_key_values is None:
        return 0
    for layer_past in past_key_values:
        for t in layer_past:
            if torch.is_tensor(t) and t.dim() >= 3 and not is_recurrent_state(t):
                return t.shape[2]
    return 0


class GatedDeltaAttention(nn.Module):
    """Simplified Gated Delta Rule linear attention (see module docstring).

    Mirrors the MLA forward signature so ``HeliosLMv5Layer`` can hold either
    module: forward(hidden_states, attention_mask=None, past_key_value=None,
    use_cache=False, position_ids=None) -> (output [B, seq, hidden], present).
    ``position_ids`` needs no positional encoding for the recurrence; it is
    used ONLY for packed-sequence handling — a position restart marks a
    document boundary at which the recurrent state is reset to zero (see the
    module docstring). Normal prefill positions (0..T-1) and decode positions
    (past_len.., strictly increasing) are accepted untouched.

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
        # Per-head scalar decay gate; bias-free Linear + a raw bias parameter
        # initialized to +4 so decay ~= sigmoid(4) ~= 0.982 at init. A raw
        # Parameter is used because HeliosLMv5._init_weights zeroes every
        # Linear bias.
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.decay_bias = nn.Parameter(torch.full((self.num_heads,), 4.0))

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_scale = 1.0 / math.sqrt(self.head_dim)

    def decay_gate(self, hidden_states):
        """Per-token per-head decay in the open interval (0, 1): [B, seq, H].

        Sigmoid output is clamped away from exact 0/1 (fp saturation), which
        also keeps the state contraction strictly below 1. The margin is
        dtype-aware (m6): m = max(1e-6, finfo(dtype).eps) — in fp16/bf16 a
        fixed 1 - 1e-6 upper bound would round to exactly 1.0; fp32 keeps
        the historical [1e-6, 1 - 1e-6] bounds.
        """
        g = torch.sigmoid(self.g_proj(hidden_states) + self.decay_bias)
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
        seg_start = None
        if position_ids is not None and seq > 1 \
                and position_ids.dim() == 2 and position_ids.shape[1] == seq:
            restart = (position_ids[:, 1:] < position_ids[:, :-1])  # [B, seq-1]
            if bool(restart.any()):
                if past_key_value is not None:
                    raise ValueError(
                        "GatedDeltaAttention: position_ids restart mid-"
                        "sequence while a non-empty past_key_value is "
                        "supplied (cached decode at a document boundary). "
                        "Continuing from a carried-over state and resetting "
                        "it at the same time is contradictory; prefill each "
                        "document separately."
                    )
                seg_start = torch.zeros(B, seq, dtype=torch.bool,
                                        device=hidden_states.device)
                seg_start[:, 0] = True
                seg_start[:, 1:] = restart
                # Defensive: a true restart always has pos[t] < pos[t-1] but
                # segment starts may also begin at 0; both are resets.

        if past_key_value is not None:
            if len(past_key_value) != 1:
                raise ValueError(
                    "GatedDeltaAttention expects a (state,) cache tuple of 1 "
                    f"tensor, got {len(past_key_value)}"
                )
            state = past_key_value[0]
            if state.shape != (B, self.num_heads, self.head_dim, self.head_dim):
                raise ValueError(
                    f"recurrent state shape {tuple(state.shape)} != expected "
                    f"{(B, self.num_heads, self.head_dim, self.head_dim)}"
                )
        else:
            state = hidden_states.new_zeros(
                B, self.num_heads, self.head_dim, self.head_dim
            )

        q = self.q_proj(hidden_states).view(B, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, seq, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, seq, self.num_heads, self.head_dim)
        # k is L2-normalized (delta rule); q is scaled like attention scores.
        k = F.normalize(k.transpose(1, 2), dim=-1)          # [B, H, seq, Dk]
        q = q.transpose(1, 2) * self.q_scale                # [B, H, seq, Dk]
        v = v.transpose(1, 2)                               # [B, H, seq, Dv]
        decay = self.decay_gate(hidden_states).transpose(1, 2)  # [B, H, seq]

        if attention_mask is not None:
            keep = attention_mask[:, -seq:].to(decay.dtype)  # [B, seq]
            keep = keep[:, None, :]                          # [B, 1, seq]
            # Masked token: decay -> 1 (no forgetting) and no write; the
            # state passes through unchanged.
            decay = 1.0 + (decay - 1.0) * keep
        else:
            keep = None

        # Sequential recurrence. Training and decode share this exact op
        # order, so a one-shot forward and token-by-token decode agree
        # numerically to float-reassociation noise (asserted at atol 1e-5
        # in the test suite). At a packed-sequence document boundary the
        # state is zeroed before the boundary token is processed.
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
            d_t = decay[:, :, t, None, None]                  # [B, H, 1, 1]
            state = d_t * state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            o_t = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
            outs.append(o_t)
        out = torch.stack(outs, dim=2)                        # [B, H, seq, Dv]

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
        return output, present
