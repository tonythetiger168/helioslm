"""Multi-Head Latent Attention (MLA) - DeepSeek-V3 Style

Key properties:
- Low-rank KV compression: hidden -> kv_latent_dim via kv_a_proj.
- Decoupled RoPE: K = cat([K_nope (per head, no RoPE), k_rope (shared, RoPE)]);
  V is a separate projection and is NEVER rotated.
- head_dim is decoupled from hidden_size / num_heads:
  head_dim = no_rope_head_dim + rope_head_dim, values use v_head_dim.
- Weight absorption (config.attention.use_absorption, v5.1): the KV cache
  stores only the compressed latent, and kv_b_proj is folded into the Q-side
  and output-side matmuls via associativity.

Two cache modes (per layer), selected by config.attention.use_absorption:

  absorbed (default) — DeepSeek-V2/V3 inference scheme:
      (c_kv   [B, 1, L, kv_latent_dim],   # post-norm latent, shared by heads
       k_rope [B, 1, L, rope_head_dim])   # shared, post-RoPE
      Per-token cost: kv_latent_dim + rope_head_dim values. Math:
        score_nope = q_nope . K_nope^T = (q_nope @ W_UK) . c_kv^T
        output     = P @ V             = (P @ c_kv) @ W_UV^T
      where W_UK / W_UV are per-head slices of kv_b_proj.weight. Decode is
      O(1) in projection compute per step: only the current token is
      projected (q_absorbed once, W_UV once); the cached latents are never
      re-expanded.

  non-absorbed (fallback / numerical reference, v5.1 behaviour):
      (k_nope [B, H, L, no_rope_head_dim],
       k_rope [B, 1, L, rope_head_dim],
       v      [B, H, L, v_head_dim])
      Per-token cost: H*(no_rope_head_dim + v_head_dim) + rope_head_dim.

Contract (relied upon by MTP rollback / inference engines): past_key_value
is a tuple of tensors and dim 2 of EVERY tensor is the sequence-length dim,
so truncating the cache with tensor[:, :, :n] along dim 2 is always legal.
"""
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


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin cache with lazy budgeting.

    Only min(max_position_embeddings, 8192) positions are precomputed at
    construction (a full 1M-position table would cost ~168 MB per layer).
    The cache doubles on demand when longer sequences show up.
    """

    _INIT_BUDGET = 8192

    def __init__(self, dim, max_position_embeddings, base=10000.0, scaling=None):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.scaling = scaling
        if scaling is not None:
            stype = scaling["type"]
            factor = float(scaling["factor"])
            if factor < 1.0:
                raise ValueError(
                    f"rope scaling factor must be >= 1.0, got {factor}"
                )
            if stype == "linear":
                # Position interpolation: position p behaves like p/factor.
                # This is a UNIFORM frequency scaling (every inv_freq_k
                # divided by factor), which is NOT expressible as a base
                # rescale (base^m would weight high frequencies more), so
                # inv_freq is scaled directly below.
                pass
            elif stype == "ntk":
                # Dynamic-NTK style: rescale the base so the HIGHEST
                # frequency sees the full stretch while lower frequencies
                # are progressively less affected. inv_freq_k = base'^(-2k/d)
                # with base' = base * factor^(d/(d-2)); k=0 gives base'^0 = 1
                # exactly, so the lowest frequency is untouched.
                base = base * factor ** (dim / (dim - 2))
            elif stype == "yarn":
                # YaRN (NTK-by-parts, v5.8): split the frequency range by
                # WAVELENGTH relative to the pre-trained context length.
                # Short wavelengths (high frequency) keep the original
                # inv_freq; long wavelengths (low frequency) are fully
                # interpolated (divided by factor); the band in between is
                # blended with a linear ramp, avoiding the discontinuity a
                # hard split would introduce. This matches the HuggingFace
                # `rope_type="yarn"` formula.
                pass  # applied after inv_freq is built below
            else:
                raise ValueError(
                    f"unknown rope scaling type {stype!r}; expected "
                    "'linear', 'ntk' or 'yarn'"
                )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if scaling is not None and scaling["type"] == "linear":
            inv_freq = inv_freq / float(scaling["factor"])
        elif scaling is not None and scaling["type"] == "yarn":
            inv_freq = self._yarn_inv_freq(inv_freq, scaling, max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq)
        # YaRN attention temperature (mscale): scaling cos/sin by this
        # factor scales the rotated q AND k, so attention scores pick up
        # factor^2 — the YaRN paper's sqrt(t) compensation for the
        # softmax sharpness lost when long-context positions are
        # interpolated onto a smaller range. Non-yarn types keep 1.0.
        self.attention_factor = 1.0
        if scaling is not None and scaling["type"] == "yarn":
            self.attention_factor = self._yarn_attention_factor(scaling)
        self.max_seq_len = 0
        budget = min(max_position_embeddings, self._INIT_BUDGET)
        self._precompute(budget)

    @staticmethod
    def _yarn_inv_freq(inv_freq, scaling, max_position_embeddings):
        """NTK-by-parts (YaRN) frequency remapping.

        Keys: ``factor`` (>=1), optional ``original_max_position`` (the
        pre-trained context length; defaults to the config's
        max_position_embeddings, which plays the role of the training
        length for this reference implementation), ``beta_fast`` (32) and
        ``beta_slow`` (1) — the wavelength thresholds in units of
        original_max_position.

        Wavelength law (HuggingFace `rope_type="yarn"`):
          wavelen_k = 2*pi / inv_freq_k
          wavelen_k <  L / beta_fast  -> keep inv_freq_k   (high freq)
          wavelen_k >  L / beta_slow  -> inv_freq_k / factor (low freq)
          in between                   -> linear ramp between the two
        """
        factor = float(scaling["factor"])
        orig_len = float(
            scaling.get("original_max_position", max_position_embeddings)
        )
        if orig_len <= 0:
            raise ValueError(
                f"yarn original_max_position must be positive, got {orig_len}"
            )
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        if not (beta_fast > beta_slow > 0):
            raise ValueError(
                f"yarn requires beta_fast > beta_slow > 0, got "
                f"beta_fast={beta_fast}, beta_slow={beta_slow}"
            )
        low_wavelen = orig_len / beta_fast
        high_wavelen = orig_len / beta_slow
        wavelen = 2.0 * math.pi / inv_freq
        # Fully interpolated beyond the long-wavelength threshold.
        remapped = torch.where(wavelen > high_wavelen,
                               inv_freq / factor, inv_freq)
        # Smooth ramp inside the band: smooth = 1 at the SHORT-wavelength
        # edge (keep original) and 0 at the LONG-wavelength edge (fully
        # interpolated).
        smooth = (orig_len / wavelen - beta_slow) / (beta_fast - beta_slow)
        ramped = (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
        in_band = (wavelen >= low_wavelen) & (wavelen <= high_wavelen)
        return torch.where(in_band, ramped, remapped)

    @staticmethod
    def _yarn_attention_factor(scaling):
        factor = float(scaling["factor"])
        explicit = scaling.get("attention_factor")
        if explicit is not None:
            explicit = float(explicit)
            if explicit <= 0:
                raise ValueError(
                    f"yarn attention_factor must be positive, got {explicit}"
                )
            return explicit
        # YaRN paper default: mscale = 0.1 * ln(s) + 1 (1.0 at factor 1).
        return 0.1 * math.log(factor) + 1.0

    def _precompute(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # persistent=False (M-C4): the table grows lazily with sequence
        # length, so a checkpoint taken after a long sequence would carry an
        # oversized buffer and strict-load into a fresh model would fail.
        # Excluding it keeps state_dict stable; the table is rebuilt on
        # demand by _ensure_capacity at the next forward.
        self.register_buffer("cos_cached",
                             (emb.cos() * self.attention_factor)[None, None, :, :],
                             persistent=False)
        self.register_buffer("sin_cached",
                             (emb.sin() * self.attention_factor)[None, None, :, :],
                             persistent=False)
        self.max_seq_len = seq_len

    def _ensure_capacity(self, seq_len):
        if seq_len > self.max_seq_len:
            self._precompute(max(seq_len, self.max_seq_len * 2))

    def forward(self, position_ids):
        """position_ids: [B, L] or [L] -> (cos, sin) of shape [B, 1, L, dim]."""
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.to(self.inv_freq.device)
        if (position_ids < 0).any():
            raise ValueError(
                f"position_ids must be non-negative, got min "
                f"{int(position_ids.min().item())}"
            )
        self._ensure_capacity(int(position_ids.max().item()) + 1)
        cos = self.cos_cached[0, 0][position_ids].unsqueeze(1)
        sin = self.sin_cached[0, 0][position_ids].unsqueeze(1)
        return cos, sin


def apply_rotary(x, cos, sin):
    """Apply RoPE. x: [B, H, L, D], cos/sin: [B, 1, L, D].

    The rotary dimension must match exactly: silently zero-padding cos/sin
    would zero out (not rotate) the trailing dims, so we fail loudly instead.
    """
    if cos.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"rotary dim mismatch: x has {x.shape[-1]} dims but cos/sin have "
            f"{cos.shape[-1]}; refusing to zero-pad"
        )

    def rotate_half(t):
        t1, t2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
        return torch.cat([-t2, t1], dim=-1)

    # The cache is fp32; cast to x.dtype so bf16/fp16 activations don't
    # silently promote the whole rotary product (or mismatch under CUDA).
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class MLA(nn.Module):
    """Multi-Head Latent Attention (see module docstring for cache layouts).

    ``config.attention.use_absorption`` selects the cache mode:
      - True:  latent cache (c_kv, k_rope) with weight absorption.
      - False: expanded per-head cache (k_nope, k_rope, v), the v5.1
        behaviour, kept as fallback and numerical reference.

    ``config.attention.kv_cache_dtype`` (v5.7) selects the STORAGE dtype of
    c_kv in absorbed mode: "auto" keeps the compute dtype; "fp8" stores the
    latent on the float8_e4m3fn grid (saturating cast, scale-free, <= 6.25%
    relative element error) — the tuple layout is unchanged and dim 2 is
    still the sequence length, so rollback clone/slice and engine
    watermarking keep working; attention always computes over the same
    quantized values that are stored.

    ``config.attention.sparse_top_k`` (v5.8) enables DSA-style top-k
    decode over the latent cache; ``config.attention.logit_soft_cap``
    (v5.9) soft-caps attention logits Gemma-style; ``config.attention.
    qk_norm`` (v5.9) RMSNorms the per-head query parts and the shared
    k_rope before RoPE; ``config.attention.sliding_window`` /
    ``sliding_window_sink`` (v5.9) restrict attention to a trailing
    position window (plus optional sink tokens) without modifying the
    cache.

    Both modes are mathematically identical up to floating-point
    reassociation (verified by the test suite at atol 1e-4).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.num_kv_heads = config.attention.num_key_value_heads

        # MLA dimensions (decoupled from hidden_size / num_heads)
        self.kv_latent_dim = config.attention.kv_latent_dim
        self.q_lora_rank = config.attention.q_lora_rank
        self.rope_head_dim = config.attention.rope_head_dim
        self.no_rope_head_dim = config.attention.no_rope_head_dim
        self.v_head_dim = config.attention.v_head_dim
        self.head_dim = self.no_rope_head_dim + self.rope_head_dim

        # getattr for robustness against configs built before the switch
        # existed; the dataclass default is True.
        self.use_absorption = bool(
            getattr(config.attention, "use_absorption", True)
        )

        # Q compression: d_model -> d_c' -> num_heads * head_dim
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # KV compression: d_model -> d_c (latent)
        self.kv_a_proj = nn.Linear(self.hidden_size, self.kv_latent_dim, bias=False)
        # Latent -> per-head K_nope and V in ONE projection (split after).
        # In absorbed mode this weight is never applied to the cache; its
        # per-head slices act as W_UK (folded into the Q side) and W_UV
        # (folded into the output side).
        self.kv_b_proj = nn.Linear(
            self.kv_latent_dim,
            self.num_heads * (self.no_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Shared RoPE key (broadcast to all heads)
        self.k_rope_proj = nn.Linear(self.hidden_size, self.rope_head_dim, bias=False)

        # Output projection consumes per-head values of width v_head_dim
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.rope = RotaryEmbedding(
            self.rope_head_dim, config.max_position_embeddings,
            scaling=getattr(config.attention, "rope_scaling", None),
        )

        # v5.7: fp8 KV-cache storage (absorbed mode only, validated eagerly
        # in the config; the expanded mode raises below as a loud fallback).
        self.kv_cache_dtype = getattr(config.attention, "kv_cache_dtype", "auto")
        if self.kv_cache_dtype == "fp8":
            if not self.use_absorption:
                raise ValueError(
                    "kv_cache_dtype='fp8' is only implemented for the "
                    "absorbed (latent) cache; the expanded per-head cache "
                    "has no latent to quantize — set "
                    "attention.use_absorption=True or kv_cache_dtype='auto'"
                )
            try:
                torch.zeros(1).to(torch.float8_e4m3fn)
            except (TypeError, RuntimeError) as exc:
                raise NotImplementedError(
                    "kv_cache_dtype='fp8' requires a torch build with "
                    "float8_e4m3fn support"
                ) from exc

        # v5.8: DSA-style sparse top-k attention at decode (absorbed mode
        # only). None = dense attention (v5.7 behaviour).
        self.sparse_top_k = getattr(config.attention, "sparse_top_k", None)
        if self.sparse_top_k is not None:
            if not isinstance(self.sparse_top_k, int) or self.sparse_top_k <= 0:
                raise ValueError(
                    f"sparse_top_k must be a positive integer or None, got "
                    f"{self.sparse_top_k!r}"
                )
            if not self.use_absorption:
                raise ValueError(
                    "sparse_top_k is only implemented for the absorbed "
                    "(latent) cache — the top-k selection gathers shared "
                    "latents; the expanded per-head cache would need a "
                    "per-head selection and is not supported. Set "
                    "attention.use_absorption=True or sparse_top_k=None"
                )

        # v5.9: Gemma-2/3 & GLM-4.5 style attention logit soft-capping.
        # Validated eagerly in the config; re-checked here for hand-built
        # attention configs (same pattern as sparse_top_k above).
        self.logit_soft_cap = getattr(config.attention, "logit_soft_cap", None)
        if self.logit_soft_cap is not None:
            if not isinstance(self.logit_soft_cap, (int, float)) \
                    or self.logit_soft_cap <= 0:
                raise ValueError(
                    f"logit_soft_cap must be a positive number or None, got "
                    f"{self.logit_soft_cap!r}"
                )
            self.logit_soft_cap = float(self.logit_soft_cap)

        # v5.9: per-head QK-norm (GLM-4.5 / Qwen3 / Gemma style). The
        # per-head query parts (q_nope, q_rope) and the shared k_rope are
        # RMSNormed BEFORE RoPE. The nope-side K is deliberately NOT
        # re-normed per head: the shared latent c_kv already passes through
        # norm_kv (the DeepSeek-V3 design), and a per-head K_nope norm
        # cannot be folded into the absorbed W_UK matmul.
        self.qk_norm = bool(getattr(config.attention, "qk_norm", False))
        if self.qk_norm:
            self.norm_q_nope = RMSNorm(self.no_rope_head_dim,
                                       eps=config.rms_norm_eps)
            self.norm_q_rope = RMSNorm(self.rope_head_dim,
                                       eps=config.rms_norm_eps)
            self.norm_k_rope = RMSNorm(self.rope_head_dim,
                                       eps=config.rms_norm_eps)

        # v5.9: sliding-window attention with optional attention sinks
        # (StreamingLLM / Gemma-3 / Qwen3 hybrid style). Semantics: a query
        # at position p attends keys with position > p - sliding_window,
        # plus the first sliding_window_sink positions unconditionally.
        # The cache is NEVER modified; absorbed decode slices gathered
        # copies so the decode matmul is O(window) instead of O(L).
        self.sliding_window = getattr(config.attention, "sliding_window", None)
        self.sliding_window_sink = int(
            getattr(config.attention, "sliding_window_sink", 0))
        if self.sliding_window is not None:
            if not isinstance(self.sliding_window, int) \
                    or self.sliding_window <= 0:
                raise ValueError(
                    f"sliding_window must be a positive integer or None, "
                    f"got {self.sliding_window!r}"
                )
        if self.sliding_window_sink < 0:
            raise ValueError(
                f"sliding_window_sink must be non-negative, got "
                f"{self.sliding_window_sink}"
            )
        if self.sliding_window_sink > 0 and self.sliding_window is None:
            raise ValueError(
                "sliding_window_sink > 0 requires sliding_window to be set "
                "(attention sinks only have meaning inside a sliding window)"
            )

        self.norm_q = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.norm_kv = RMSNorm(self.kv_latent_dim, eps=config.rms_norm_eps)
        self.attention_dropout = config.attention.attention_dropout
        # Matches the SDPA default scale (1/sqrt(head_dim)) used by the
        # non-absorbed path, where the score matmul runs over head_dim dims.
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _resolve_positions(self, hidden_states, past_len, position_ids):
        """Default/expand position_ids. Returns (position_ids, custom)."""
        B, seq, _ = hidden_states.shape
        custom = position_ids is not None
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq, device=hidden_states.device
            ).unsqueeze(0).expand(B, seq)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, seq)
        return position_ids, custom

    def _project_new_tokens(self, hidden_states, position_ids):
        """Project only the NEW tokens (O(1) per decode step).

        Returns:
            q_nope: [B, H, seq, no_rope_head_dim]
            q_rope: [B, H, seq, rope_head_dim]   (post-RoPE)
            c_kv:   [B, seq, kv_latent_dim]      (post-norm latent)
            k_rope: [B, 1, seq, rope_head_dim]   (post-RoPE, shared)
        """
        B, seq, _ = hidden_states.shape

        # Q: compress then expand, split into no-RoPE / RoPE parts
        q = self.q_b_proj(self.norm_q(self.q_a_proj(hidden_states)))
        q = q.view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q_nope = q[..., : self.no_rope_head_dim]
        q_rope = q[..., self.no_rope_head_dim :]

        # KV: compress to the shared latent (post-norm; this is what the
        # absorbed cache stores directly).
        c_kv = self.norm_kv(self.kv_a_proj(hidden_states))  # [B, seq, d_c]

        # Shared RoPE key for new tokens
        k_rope = self.k_rope_proj(hidden_states)
        k_rope = k_rope.view(B, seq, 1, self.rope_head_dim).transpose(1, 2)

        # v5.9: per-head QK-norm BEFORE RoPE (rotation preserves the norm,
        # so normalizing pre-rotation keeps the rotated vectors normed).
        if self.qk_norm:
            q_nope = self.norm_q_nope(q_nope)
            q_rope = self.norm_q_rope(q_rope)
            k_rope = self.norm_k_rope(k_rope)

        # RoPE applied ONLY to the rope dims of q and the shared k_rope,
        # at the correct absolute positions. V is never rotated.
        cos, sin = self.rope(position_ids)
        q_rope = apply_rotary(q_rope, cos, sin)
        k_rope = apply_rotary(k_rope, cos, sin)
        return q_nope, q_rope, c_kv, k_rope

    def _w_uk_w_uv(self):
        """Split kv_b_proj.weight into per-head W_UK and W_UV (views).

        kv_b_proj maps latent -> concat_h[K_nope_h (d_nope) | V_h (d_v)],
        so weight.view(H, d_nope + d_v, d_c) slices cleanly:
            W_UK: [H, no_rope_head_dim, kv_latent_dim]
            W_UV: [H, v_head_dim,       kv_latent_dim]
        Views are cheap; recomputed per call so training updates are seen.
        """
        w = self.kv_b_proj.weight.view(
            self.num_heads, self.no_rope_head_dim + self.v_head_dim,
            self.kv_latent_dim,
        )
        return w[:, : self.no_rope_head_dim], w[:, self.no_rope_head_dim :]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _soft_cap(self, scores):
        """v5.9: Gemma-style logit soft-capping, cap*tanh(scores/cap).

        Applied AFTER the softmax scale and BEFORE the causal/padding
        mask, so masked positions still become exactly -inf afterwards.
        Bounds every attention logit to (-cap, cap).
        """
        if self.logit_soft_cap is None:
            return scores
        return self.logit_soft_cap * torch.tanh(scores / self.logit_soft_cap)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None):
        """
        Args:
            hidden_states: [B, seq, hidden] — only the NEW tokens when decoding.
            attention_mask: optional [B, kv_len] (1 = attend, 0 = pad). Must cover
                past + current positions; a [B, seq] mask for current tokens only
                is left-padded with ones for the cached prefix.
            past_key_value: optional cache tuple — (c_kv, k_rope) in absorbed
                mode, (k_nope, k_rope, v) otherwise. dim 2 = sequence length
                for every tensor in both modes.
            use_cache: return the updated cache tuple.
            position_ids: optional [B, seq] or [seq] absolute positions of the
                new tokens. Defaults to past_len .. past_len + seq - 1.
        Returns:
            (output [B, seq, hidden], present_key_value or None)
        """
        if self.use_absorption:
            return self._forward_absorbed(
                hidden_states, attention_mask, past_key_value, use_cache,
                position_ids,
            )
        return self._forward_expanded(
            hidden_states, attention_mask, past_key_value, use_cache,
            position_ids,
        )

    def _forward_expanded(self, hidden_states, attention_mask, past_key_value,
                          use_cache, position_ids):
        """Non-absorbed path (v5.1 behaviour): cache expanded per-head K/V."""
        B, seq, _ = hidden_states.shape
        if past_key_value is not None and len(past_key_value) != 3:
            raise ValueError(
                "expanded mode (use_absorption=False) expects a "
                "(k_nope, k_rope, v) cache tuple of 3 tensors, got "
                f"{len(past_key_value)}"
                + (
                    " — a 2-tuple is the absorbed (c_kv, k_rope) layout; the "
                    "cache format follows config.attention.use_absorption and "
                    "cannot be mixed"
                    if len(past_key_value) == 2 else ""
                )
            )
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        position_ids, custom_positions = self._resolve_positions(
            hidden_states, past_len, position_ids
        )

        q_nope, q_rope, c_kv, k_rope = self._project_new_tokens(
            hidden_states, position_ids
        )

        # Expand the latent ONCE per new token into K_nope and V.
        kv = self.kv_b_proj(c_kv)
        kv = kv.view(B, seq, self.num_heads, self.no_rope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)  # [B, H, seq, no_rope + v]
        k_nope = kv[..., : self.no_rope_head_dim].contiguous()
        v = kv[..., self.no_rope_head_dim :].contiguous()

        # Append to cache (cache holds expanded per-head tensors)
        if past_key_value is not None:
            past_k_nope, past_k_rope, past_v = past_key_value
            k_nope = torch.cat([past_k_nope, k_nope], dim=2)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k_nope, k_rope, v) if use_cache else None
        kv_len = k_nope.shape[2]

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat(
            [k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1
        )

        attn_mask = self._build_attn_mask(
            position_ids, kv_len, attention_mask, past_len, seq, B, custom_positions
        )
        if self.logit_soft_cap is None:
            attn_out = F.scaled_dot_product_attention(
                q_full, k_full, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attn_mask is None,
                attn_mask=attn_mask,
            )
        else:
            # v5.9: SDPA has no soft-capping hook, so the capped path runs
            # the score/softmax/weighted-sum matmuls manually (same math as
            # the absorbed path below, including the nan_to_num guard for
            # fully-masked rows).
            scores = torch.matmul(q_full, k_full.transpose(-2, -1))
            scores = self._soft_cap(scores * self.softmax_scale)
            if attn_mask is None:
                attn_mask = torch.ones(seq, kv_len, dtype=torch.bool,
                                       device=scores.device).tril(
                    diagonal=kv_len - seq)
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            probs = torch.nan_to_num(F.softmax(scores, dim=-1))
            if self.training and self.attention_dropout > 0:
                probs = F.dropout(probs, p=self.attention_dropout)
            attn_out = torch.matmul(probs, v)

        output = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, seq, -1))
        return output, present_key_value

    def _forward_absorbed(self, hidden_states, attention_mask, past_key_value,
                          use_cache, position_ids):
        """Absorbed path: cache stores (c_kv, k_rope); kv_b_proj is folded
        into the Q-side and output-side matmuls.

        Math (per head h, W_UK/W_UV = per-head slices of kv_b_proj.weight):
            K_nope = c_kv @ W_UK^T, V = c_kv @ W_UV^T   (definitions)
            score  = q_nope . K_nope^T + q_rope . k_rope^T
                   = (q_nope @ W_UK) . c_kv^T + q_rope . k_rope^T
            out    = P @ V = (P @ c_kv) @ W_UV^T
        so the cache only needs c_kv and k_rope.

        Prefill vs decode: the same absorbed kernel handles both (the score
        and weighted-sum matmuls are over the cached latents for any kv_len,
        and projections run on the new tokens only). An alternative prefill
        that runs the expanded path and then compresses back to latent is
        equally valid; the unified path is simpler and avoids a second code
        path to keep in sync. Decode is O(1) per step in projection compute
        (q_absorbed and W_UV are applied to the current token only).
        """
        B, seq, _ = hidden_states.shape
        if past_key_value is not None and len(past_key_value) != 2:
            raise ValueError(
                "absorbed mode expects a (c_kv, k_rope) cache tuple, got "
                f"{len(past_key_value)} tensors; the cache format follows "
                "config.attention.use_absorption and cannot be mixed"
            )
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        position_ids, custom_positions = self._resolve_positions(
            hidden_states, past_len, position_ids
        )

        q_nope, q_rope, c_kv, k_rope = self._project_new_tokens(
            hidden_states, position_ids
        )
        c_kv = c_kv.unsqueeze(1)  # [B, 1, seq, d_c] — shared across heads
        compute_dtype = hidden_states.dtype

        # Append to the latent cache (dim 2 = sequence length). An fp8-stored
        # past cache upcasts here; it is re-rounded to the E4M3 grid below
        # together with the new tokens, so cached decode sees exactly what
        # prefill stored (no double-quantization drift).
        if past_key_value is not None:
            past_c_kv, past_k_rope = past_key_value
            c_kv = torch.cat([past_c_kv.to(compute_dtype), c_kv], dim=2)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)

        stored_c_kv = c_kv
        if self.kv_cache_dtype == "fp8":
            # Saturating cast onto the E4M3 grid, scale-free (post-RMSNorm
            # latents are O(1); 3 mantissa bits -> <= 6.25% relative element
            # error). No sidecar scale tensor: cache consumers (MTP
            # rollback clone/slice, engine watermark cat) only need to
            # preserve dtype. Attention always runs over the SAME quantized
            # values that are stored (current tokens included).
            stored_c_kv = c_kv.detach().to(torch.float8_e4m3fn)
            # Straight-through estimator: the VALUE is the quantized
            # latent (bit-identical to the no-grad inference path) while
            # the GRADIENT flows back unchanged into kv_a_proj / norm_kv —
            # without it a grad-requiring forward stores a detached cache
            # and the KV projections silently stop training.
            c_kv = stored_c_kv.to(compute_dtype) + c_kv - c_kv.detach()

        present_key_value = (stored_c_kv, k_rope) if use_cache else None
        kv_len = c_kv.shape[2]

        w_uk, w_uv = self._w_uk_w_uv()  # [H, d_nope, d_c], [H, d_v, d_c]

        # q_absorbed[h] = q_nope[h] @ W_UK[h]  -> [B, H, seq, d_c]
        q_absorbed = torch.einsum("bhqd,hdc->bhqc", q_nope, w_uk)

        attn_mask = self._build_attn_mask(
            position_ids, kv_len, attention_mask, past_len, seq, B, custom_positions
        )

        # Sliding-window decode slice (v5.9): at DECODE the default
        # contiguous layout is enforced (custom position_ids raise in
        # _build_attn_mask), so the cached keys hold positions
        # 0..kv_len-1 and the query sits at kv_len-1. The window then
        # keeps exactly the first `sliding_window_sink` keys (attention
        # sinks) plus the last `sliding_window` keys. Only GATHERED
        # COPIES are sliced — the cache itself is never modified — and
        # the mask (already window-restricted by _build_attn_mask) is
        # sliced with the same indices. This makes the decode matmul
        # O(window + sink) instead of O(kv_len). When
        # sink + window >= kv_len every key is inside the window and
        # nothing is sliced (bit-identical to full attention).
        if self.sliding_window is not None and seq == 1:
            keep = self.sliding_window_sink + self.sliding_window
            if kv_len > keep:
                idx = torch.cat([
                    torch.arange(self.sliding_window_sink,
                                 device=c_kv.device),
                    torch.arange(kv_len - self.sliding_window, kv_len,
                                 device=c_kv.device),
                ])
                c_kv = c_kv.index_select(2, idx)
                k_rope = k_rope.index_select(2, idx)
                if attn_mask is not None:
                    attn_mask = attn_mask.index_select(-1, idx)
                kv_len = keep

        # DSA-style sparse top-k selection (v5.8): at DECODE (seq == 1) a
        # lightning-indexer-style score picks the top-`sparse_top_k` cached
        # tokens and attention runs only over them. The index score is the
        # head-mean of the TRUE score terms (a "free" indexer reusing the
        # absorbed projections — a real DSA indexer is a dedicated learned
        # scorer; see README). Selection is order-preserving (indices are
        # sorted), the cache tensors themselves are NOT modified (gathered
        # copies only), and the current token is always force-selected.
        # Prefill (seq > 1) stays dense by design — per-query top-k over
        # the full sequence costs O(L^2), defeating the point.
        if (self.sparse_top_k is not None and seq == 1
                and kv_len > self.sparse_top_k):
            index_scores = torch.matmul(
                q_absorbed.mean(dim=1, keepdim=True), c_kv.transpose(-2, -1)
            ) + torch.matmul(
                q_rope.mean(dim=1, keepdim=True), k_rope.transpose(-2, -1)
            )  # [B, 1, seq, kv_len] with seq == 1
            if attn_mask is not None:
                # Never select keys the causal/padding mask excludes.
                index_scores = index_scores.masked_fill(
                    ~attn_mask, float("-inf"))
            # The current (last) token must always attend to itself.
            index_scores[..., -1] = float("inf")
            top_idx = torch.topk(index_scores, self.sparse_top_k, dim=-1
                                 ).indices.sort(dim=-1).values  # [B,1,1,k]
            # gather along the sequence dim: index [B, 1, k, d]
            gidx = top_idx.squeeze(2).unsqueeze(-1)
            c_kv = torch.gather(
                c_kv, 2, gidx.expand(-1, -1, -1, c_kv.shape[-1]))
            k_rope = torch.gather(
                k_rope, 2, gidx.expand(-1, -1, -1, k_rope.shape[-1]))
            if attn_mask is not None:
                attn_mask = torch.gather(attn_mask, -1, top_idx)
            kv_len = self.sparse_top_k

        # Scores in latent space + the decoupled-RoPE term. c_kv / k_rope
        # have head dim 1 and broadcast over H.
        scores = torch.matmul(q_absorbed, c_kv.transpose(-2, -1))
        scores = scores + torch.matmul(q_rope, k_rope.transpose(-2, -1))
        scores = scores * self.softmax_scale  # [B, H, seq, kv_len]
        # v5.9: soft-cap after the scale, before the mask (masked
        # positions still become exactly -inf below).
        scores = self._soft_cap(scores)

        if attn_mask is None:
            # Fast path equivalent to SDPA is_causal=True: bottom-right
            # causal mask (here past_len == 0, so it is a plain tril).
            attn_mask = torch.ones(seq, kv_len, dtype=torch.bool,
                                   device=scores.device).tril(diagonal=kv_len - seq)
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        # nan_to_num: a fully-masked query row (e.g. a left-padding position
        # whose attention_mask is 0) softmaxes to NaN; those rows are dropped
        # by the caller anyway, but an unguarded NaN would flow into the
        # residual stream and poison real tokens in later layers via
        # 0 * NaN. SDPA returns 0 for such rows, so zeroing the weights here
        # keeps the absorbed path equivalent to the expanded one (C1).
        probs = torch.nan_to_num(F.softmax(scores, dim=-1))
        if self.training and self.attention_dropout > 0:
            probs = F.dropout(probs, p=self.attention_dropout)

        # Weighted sum in LATENT space (broadcast over heads), then one
        # W_UV application per new token. W_UV is deliberately kept separate
        # from o_proj: folding it in would give a single [H*d_c, hidden]
        # projection with d_c/d_v times more parameters than
        # (W_UV: H*d_v*d_c) + (o_proj: H*d_v*hidden) and would prevent
        # training-time weight updates from being reflected in a precomputed
        # fused matrix. The two-matmul form is what DeepSeek uses at decode.
        out_latent = torch.matmul(probs, c_kv)  # [B, H, seq, d_c]
        out = torch.einsum("bhqc,hvc->bqhv", out_latent, w_uv)
        output = self.o_proj(out.reshape(B, seq, self.num_heads * self.v_head_dim))
        return output, present_key_value

    def _build_attn_mask(self, position_ids, kv_len, attention_mask, past_len, seq, B, custom_positions):
        """Bottom-right-aligned causal mask, merged with the padding mask.

        Returns None when the fast path (SDPA is_causal=True) is equivalent:
        no past, no padding mask, and default contiguous positions.

        Causality compares key POSITIONS to query positions, not key cache
        indices (M-C1). Under the default layout (positions ==
        past_len .. past_len+seq-1) key index == key position and the two
        coincide. With custom ``position_ids``:
          - prefill (past_len == 0): the keys ARE the current tokens, so
            their positions are ``position_ids`` itself — comparing in
            position space stays correct even for non-contiguous /
            packed-sequence position layouts;
          - decode (past_len > 0): the positions of the cached keys are
            not recoverable from the cache (it stores tensors, not
            positions), so only the default monotonic layout is supported;
            any other custom position_ids raise ValueError instead of
            silently mis-masking.

        Packed sequences (B1): at prefill, custom position_ids may pack
        several documents into one row with per-document position restarts
        (e.g. [0,1,2,0,1,2]). A purely positional ``k_pos <= q_pos``
        comparison would let document B attend back into document A, so
        the causal mask additionally requires both tokens to belong to the
        same document. Document boundaries are derived from the reset
        points of position_ids: every position where
        ``pos[:, i] <= pos[:, i-1]`` starts a new segment.

        Sliding window (v5.9): when ``self.sliding_window`` is set, the
        mask additionally requires ``q_pos - k_pos < sliding_window``
        (position distance, so it composes with packed documents), unless
        ``k_pos < sliding_window_sink`` (StreamingLLM attention sinks are
        attendable from any position).
        """
        if attention_mask is None and past_len == 0 and not custom_positions \
                and self.sliding_window is None:
            return None

        device = position_ids.device
        same_doc = None
        if custom_positions:
            if past_len == 0:
                # Prefill: key positions are the positions of these tokens.
                k_pos = position_ids.view(B, 1, 1, seq)
                # B1: segment documents at position resets so attention
                # never crosses a packed document boundary.
                reset = torch.ones_like(position_ids, dtype=torch.bool)
                reset[:, 1:] = position_ids[:, 1:] <= position_ids[:, :-1]
                doc_id = reset.cumsum(dim=1)
                same_doc = doc_id.view(B, 1, seq, 1) == doc_id.view(B, 1, 1, seq)
            else:
                default = torch.arange(
                    past_len, past_len + seq, device=device
                ).unsqueeze(0).expand(B, seq)
                if not torch.equal(position_ids, default):
                    raise ValueError(
                        "custom position_ids are only supported at prefill "
                        "(no KV cache): during decode the positions of the "
                        "cached keys are unknown, so position_ids must equal "
                        f"the default arange({past_len}, {past_len + seq}); "
                        "got a non-default layout that would silently "
                        "mis-mask"
                    )
                k_pos = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
        else:
            # Default layout: cached key index i holds position i.
            k_pos = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
        q_pos = position_ids.view(B, 1, seq, 1)
        causal = k_pos <= q_pos  # [B, 1, seq, kv_len], True = attend
        if same_doc is not None:
            causal = causal & same_doc
        if self.sliding_window is not None:
            window_ok = (q_pos - k_pos) < self.sliding_window
            if self.sliding_window_sink > 0:
                window_ok = window_ok | (k_pos < self.sliding_window_sink)
            causal = causal & window_ok

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if attention_mask.shape[1] == seq and past_len > 0:
                # Mask covers current tokens only; cached prefix is all valid.
                prefix = torch.ones(B, past_len, dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([prefix, attention_mask], dim=1)
            if attention_mask.shape[1] != kv_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.shape[1]} matches neither "
                    f"kv_len {kv_len} nor current seq {seq}"
                )
            causal = causal & attention_mask[:, None, None, :].bool()
        return causal

    def get_kv_cache_size(self, seq_len):
        """Truthful per-token cache comparison (values, not bytes).

        Modes (this MLA instance reports ``mla`` for its ACTIVE mode):
          absorbed:  kv_latent_dim + rope_head_dim per token (latent cache).
          expanded:  H*(no_rope_head_dim + v_head_dim) + rope_head_dim.
        Baselines: GQA (num_key_value_heads) and MHA (num_attention_heads),
        both caching full per-head K and V.

        With weight absorption on, the MLA cache is dramatically smaller
        than MHA — e.g. the full config stores 576 values/token vs 24576
        for MHA (~97.7% reduction); the lite config stores 72 vs 256
        (~71.9% reduction).
        """
        expanded = seq_len * (
            self.num_heads * (self.no_rope_head_dim + self.v_head_dim)
            + self.rope_head_dim
        )
        latent = seq_len * (self.kv_latent_dim + self.rope_head_dim)
        mla = latent if self.use_absorption else expanded
        gqa = 2 * seq_len * self.num_kv_heads * self.head_dim
        mha = 2 * seq_len * self.num_heads * self.head_dim
        return {
            "mode": "absorbed" if self.use_absorption else "expanded",
            "mla": mla,                      # active mode, matches the real cache
            "mla_absorbed": latent,
            "mla_expanded": expanded,
            "gqa": gqa,
            "mha": mha,
            "reduction_vs_gqa_pct": (1 - mla / gqa) * 100,
            "reduction_vs_mha_pct": (1 - mla / mha) * 100,
            "absorbed_reduction_vs_mha_pct": (1 - latent / mha) * 100,
        }
