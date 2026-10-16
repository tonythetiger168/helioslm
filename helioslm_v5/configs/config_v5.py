"""HeliosLM v5.22 Configuration

Two sizes are supported via ``HeliosLMv5Config(size=...)``:
  - ``"lite"``: small CPU-friendly config for smoke tests (seconds per step).
  - ``"full"``: production-scale defaults.

v5.5 additions (Kimi-K3-aligned, all backward compatible on "lite"):
  - ``HybridAttentionConfig``: interleaved GatedDeltaAttention / MLA layers.
  - ``moe.latent_dim`` (LatentMoE), ``moe.balance_strategy`` ("heuristic" /
    "quantile"), ``moe.activation`` ("swiglu" / "situ") + ``situ_softcap``.
  - ``use_attention_residuals``: cross-layer accumulated attention residual.
v5.7: ``attention.rope_scaling`` ("linear"/"ntk"), ``kv_cache_dtype="fp8"``,
  ``use_hyper_connections``. v5.8: ``rope_scaling["type"]="yarn"``
  (NTK-by-parts), ``attention.sparse_top_k`` (DSA-style top-k decode).
v5.9: ``attention.logit_soft_cap`` (Gemma-style score capping),
  ``attention.qk_norm`` (per-head pre-RoPE QK-norm),
  ``attention.sliding_window`` + ``sliding_window_sink`` (windowed
  attention with StreamingLLM sinks), ``final_logit_soft_cap`` (Gemma-2
  style LM-head capping).

``None`` means "follow the size" on every size-dependent field: the v5.5+
switches (``hybrid.enabled`` / ``use_attention_residuals`` /
``moe.latent_dim`` / ``moe.balance_strategy`` / ``use_hyper_connections``)
and the architecture dims the size presets scale — ``vocab_size``,
``max_position_embeddings``, ``hidden_size``, ``num_hidden_layers``,
``intermediate_size``, the attention head/latent dims,
``moe.num_experts`` / ``moe.num_activated_experts`` /
``moe.device_group_size``, ``mtp.num_modules``, the hybrid
linear-attention dims and ``multimodal.enabled``. The "full" size fills
them with the production defaults, the "lite" size with the CPU smoke-test
preset (see ``HeliosLMv5Config._SIZE_PRESETS``), so the existing test
suite stays bit-identical. Explicit values always win: any non-``None``
value a caller passes is kept verbatim and validated exactly like the
"full" path — ``HeliosLMv5Config(size="lite", hidden_size=...)`` can no
longer be silently re-sized by the preset.

``__post_init__`` validates structural constraints (divisibility etc.) and
raises ``ValueError`` immediately on an invalid configuration instead of
letting silent truncation happen downstream.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class AttentionConfig:
    """MLA configuration.

    MLA decouples ``head_dim`` from ``hidden_size / num_attention_heads``:
      head_dim = no_rope_head_dim + rope_head_dim   (query/key per head)
      value dim per head = v_head_dim

    Cache layout depends on ``use_absorption``:
      - True (default, DeepSeek-style weight absorption): the cache stores
        the compressed latent ``c_kv`` plus the shared ``k_rope`` only;
        ``W_UK`` is folded into the Q side and ``W_UV`` into the output side
        at matmul time. Per-token cache cost is ``kv_latent_dim +
        rope_head_dim`` (vs. ``H*(no_rope+v_head_dim) + rope_head_dim``
        expanded).
      - False: the cache stores expanded per-head K_nope + V (v5.1
        behaviour), kept as a numerical reference and fallback.
    """
    # Size-dependent dims: None = "follow the size" (filled from
    # HeliosLMv5Config._SIZE_PRESETS in __post_init__; the values below are
    # the "full" production defaults, explicit values always win).
    num_attention_heads: Optional[int] = None  # full: 64 (was 96, but 4096 % 96 != 0, M4)
    num_key_value_heads: Optional[int] = None  # full: 8 (GQA baseline reference for cache-size stats)
    kv_latent_dim: Optional[int] = None        # full: 512 (d_c: KV compression latent dim)
    q_lora_rank: Optional[int] = None          # full: 1536 (d_c': Q compression latent dim)
    no_rope_head_dim: Optional[int] = None     # full: 128 (per-head q/k dims WITHOUT RoPE)
    rope_head_dim: Optional[int] = None        # full: 64 (per-head q/k dims WITH RoPE, shared k_rope)
    v_head_dim: Optional[int] = None           # full: 128 (per-head value dim, never rotated)
    attention_dropout: float = 0.0
    use_absorption: bool = True    # MLA weight absorption (latent KV cache)
    # RoPE position scaling (v5.7): None = vanilla RoPE; otherwise a dict
    # {"type": "linear" | "ntk" | "yarn", "factor": float >= 1}. All stretch
    # the effective context by ``factor`` so a checkpoint trained at L
    # positions can attend up to ~L*factor positions: "linear" scales all
    # frequencies down by factor (position p behaves like p/factor), "ntk"
    # (Neural Tangent Kernel / dynamic-NTK style) rescales only the RoPE
    # base, leaving the lowest frequencies nearly untouched. "yarn" (v5.8,
    # NTK-by-parts) keeps high frequencies, fully interpolates low
    # frequencies, and blends the band in between; optional keys
    # "original_max_position" (default = max_position_embeddings),
    # "beta_fast" (32), "beta_slow" (1), "attention_factor" (default
    # 0.1*ln(factor)+1, applied to cos/sin).
    rope_scaling: Optional[dict] = None
    # NoPE (v5.22, Kimi-K3 direction): when True, MLA layers skip RoPE
    # entirely — q_rope / k_rope stay unrotated and position_ids only
    # drive the causal mask and cache bookkeeping. Attention becomes
    # permutation-equivariant over the visible prefix (order enters only
    # through the causal mask), so position information must come from
    # elsewhere in the stack — in K3-class hybrids the recurrent
    # (GatedDeltaAttention) layers and higher-level structure carry it.
    # GDA layers are already position-free (v5.5). False = v5.4 behaviour
    # (bit-identical). Mutually exclusive with rope_scaling: NoPE wins and
    # rope_scaling is ignored (loud warning-free by design: the config
    # simply has no rotary to scale).
    nope: bool = False
    # v5.69 (roadmap U3, Llama 4): attention temperature for NoPE layers.
    # With nope, queries at position p are scaled by
    #   log(floor((p + 1) / nope_temp_floor_scale) + 1) * nope_temp_scale + 1
    # (exactly 1.0 for p + 1 < floor_scale), sharpening attention at long
    # range where a position-free softmax flattens. Llama 4 uses 8192 / 0.1.
    # False (default) = no scaling.
    # v5.70 (roadmap U4, Gemma 4), applied to GLOBAL MLA layers only (no
    # sliding window; every MLA layer when no window is configured):
    # global_rope_partial = p-RoPE fraction of RoPE frequency pairs kept
    # (the highest frequencies); the rest get inv_freq 0, i.e. an exact
    # identity rotation. 1.0 (default) = full RoPE, bit-identical. Gemma 4
    # uses 0.25. Not combinable with rope_scaling.
    global_rope_partial: float = 1.0
    # global_k_eq_v: values reuse the keys (V = K_nope per head), so
    # kv_b_proj only emits K_nope. Needs v_head_dim == no_rope_head_dim.
    # Halves kv_b_proj and the expanded-mode K/V cache share; the absorbed
    # cache stores the latent and does not shrink.
    global_k_eq_v: bool = False
    # v5.71 (roadmap U6, Gemma 3n/4 cross-layer KV sharing): the last N
    # MLA layers (counted over MLA layers only) compute no K/V of their
    # own; each reuses the full K/V of the most recent non-shared MLA
    # layer of the same type (global vs sliding window), including that
    # layer's kv_b_proj, so absorbed == expanded still holds. Their cache
    # entry aliases the source's tuple, so unique KV memory is exactly
    # (L_mla - N) / L_mla. 0 (default) = off, bit-identical. Not
    # combinable with sparse_top_k or kv_cache_dtype="fp8".
    kv_share_last_n: int = 0
    nope_attn_temperature: bool = False
    nope_temp_floor_scale: int = 8192
    nope_temp_scale: float = 0.1
    # Gated MLA (v5.57, Kimi-K3 direction): an elementwise sigmoid output
    # gate computed from the layer input, applied to the per-head values
    # right before o_proj:  o = o_proj(attn_out * sigmoid(o_gate_proj(x))).
    # Same placement in the expanded and absorbed paths, so the two stay
    # equivalent. False (default) = no o_gate_proj, bit-identical to v5.56.
    # Checkpoint caveat: a gated model has an extra o_gate_proj weight per
    # MLA layer, so loading an ungated checkpoint strictly fails loudly.
    output_gate: bool = False
    # KV-cache storage dtype (v5.7): "auto" keeps the compute dtype;
    # "fp8" stores the absorbed mode's c_kv cache in float8_e4m3fn
    # (saturating cast, scale-free — post-RMSNorm latents are O(1), and the
    # E4M3 mantissa grid gives <= 6.25% relative element error). Absorbed
    # mode only; non-absorbed configs raise ValueError at layer init.
    kv_cache_dtype: str = "auto"
    # DSA-style sparse top-k attention (v5.8, absorbed mode only): at
    # DECODE (one new token per step) a lightning-indexer-style score
    # selects the top-``sparse_top_k`` cached tokens and attention runs
    # only over them; prefill stays dense (documented simplification).
    # None = dense attention (v5.7 behaviour). The cache layout is
    # unchanged; when sparse_top_k >= kv_len the output is bit-identical
    # to dense attention.
    sparse_top_k: Optional[int] = None
    # Indexer choice for sparse_top_k (limitations round 2026-09-21):
    #   "head_mean": v5.8 "free" indexer — the head-mean of the true
    #     absorbed score terms. Default, bit-identical to v5.8.
    #   "learned": a dedicated LEARNED lightning indexer
    #     (attention/lightning_indexer.py) — H_idx low-dim heads score
    #     every cached token as sum_h w_h * relu(q_h . k_h); keys are
    #     derived from the cached latent c_kv by a learned projection so
    #     the (c_kv, k_rope) cache contract is untouched. Requires
    #     sparse_top_k. Top-k selection is discrete, so the indexer is
    #     NOT trained by the LM loss: train it with the DSA-style
    #     distillation helper MLA.indexer_distill_loss() (teacher = the
    #     detached head-mean true-score distribution).
    sparse_indexer: str = "head_mean"
    # Learned-indexer shape: few small heads keep all-L scoring cheap
    # relative to main attention (the "lightning" property).
    indexer_num_heads: int = 4
    indexer_head_dim: int = 32
    # IndexShare (B1-1): share ONE learned indexer across every
    # ``indexer_share_every`` MLA layers (GLM-5.2/5.3-style shared sparse
    # indexer) instead of one indexer per MLA layer. MLA layers are
    # numbered 0..M-1 in model order (hybrid interleaving included);
    # layer m shares the indexer of layer m - (m % indexer_share_every),
    # the group leader, which owns the registered parameters. 1 =
    # per-layer (default, bit-identical to the 2026-09-21 behaviour).
    # Indexer memory and per-step key-projection cost drop by
    # ~1 - 1/indexer_share_every; each layer still scores its OWN cache,
    # so selection semantics per layer are unchanged. Requires
    # sparse_indexer="learned" ("head_mean" has no parameters to share).
    indexer_share_every: int = 1
    # Sparse prefill (same round): when True and sparse_top_k is set,
    # PREFILL also selects top-k keys per query in index space (the v5.8
    # simplification kept prefill dense). Queries with fewer visible keys
    # than k degrade exactly to dense over the visible set; k >= kv_len is
    # bit-identical to dense prefill. Default False = v5.8 behaviour.
    sparse_prefill: bool = False
    # Attention logit soft-capping (v5.9, Gemma-2/3 & GLM-4.5 style):
    # when a positive float, attention scores are soft-capped as
    # ``cap * tanh(score / cap)`` after the softmax scale and before the
    # causal/padding mask, bounding every logit to (-cap, cap). This
    # suppresses the attention-logit blow-up observed in long training
    # runs. None = uncapped (v5.8 behaviour, bit-identical).
    logit_soft_cap: Optional[float] = None
    # Per-head QK-norm (v5.9, GLM-4.5 / Qwen3 / Gemma style): RMSNorm the
    # per-head query parts (q_nope over no_rope_head_dim, q_rope over
    # rope_head_dim) and the shared k_rope (over rope_head_dim) BEFORE
    # RoPE. Documented simplification: the nope-side K needs no extra
    # norm — the shared latent c_kv is already RMSNormed (norm_kv, the
    # DeepSeek-V3 design), and a per-head K_nope norm cannot be folded
    # into the absorbed W_UK matmul, so it is intentionally omitted.
    qk_norm: bool = False
    # Sliding-window attention (v5.9, StreamingLLM / Gemma-3 / Qwen3
    # hybrid style): when a positive int W, a query at position p attends
    # only keys with position > p - W (distance < W). In absorbed decode
    # the gathered cache copies are sliced to the window so the decode
    # matmul cost is O(W) instead of O(L); prefill and the expanded mode
    # apply the window through the attention mask. The cache itself is
    # never modified. ``sliding_window_sink`` > 0 additionally keeps the
    # first S positions attendable from anywhere (StreamingLLM attention
    # sinks); sinks require a sliding window. W >= sequence length is
    # bit-identical to full attention.
    sliding_window: Optional[int] = None
    sliding_window_sink: int = 0
    # Per-layer sliding/global interleave (v5.62, MiMo-V2-Flash style):
    # when an int N together with ``sliding_window``, MLA layer i keeps
    # FULL (global) attention iff ``i == 0 or i % N == N - 1`` and uses the
    # window otherwise (N=6 gives MiMo's 5 sliding : 1 global with a
    # global first layer). None (default) = the window applies to every
    # MLA layer, exactly as in v5.9.
    global_attention_every: Optional[int] = None
    # Learnable attention sink (v5.62, MiMo-V2-Flash / gpt-oss style): one
    # learnable logit per head joins every softmax as an extra "null" key
    # with no value, so a head can put probability mass nowhere instead of
    # being forced onto the few keys a small window leaves it. Off by
    # default (no parameter, bit-identical); when on, it starts at 0.
    learnable_sink: bool = False

    @property
    def head_dim(self) -> int:
        return self.no_rope_head_dim + self.rope_head_dim


@dataclass
class MoEConfig:
    # Size-dependent: None = "follow the size" (filled from
    # HeliosLMv5Config._SIZE_PRESETS in __post_init__; the values below are
    # the "full" production defaults, explicit values always win).
    num_experts: Optional[int] = None  # full: 256
    num_shared_experts: int = 1        # same for both sizes
    num_activated_experts: Optional[int] = None  # full: 8 (top-k)
    # None -> defaults to HeliosLMv5Config.intermediate_size (resolved in
    # HeliosLMv5Config.__post_init__)
    expert_hidden_size: Optional[int] = None
    # Number of devices across which experts are partitioned for
    # device-limited routing (experts_per_device = num_experts / device_group_size).
    device_group_size: Optional[int] = None  # full: 8; None = follow size
    # Fixed step size for the aux-free load-balancing bias update.
    bias_update_rate: float = 1e-3
    # LatentMoE (v5.5): when an int, routed experts operate in a shared
    # latent space: down_proj(hidden->latent) -> router/experts in latent ->
    # up_proj(latent->hidden). Shared experts stay full-width.
    # None -> size default (full: 1024, lite: full-width, i.e. v5.4);
    # a non-positive value explicitly selects full-width routed experts.
    latent_dim: Optional[int] = None
    # Aux-free balancing strategy for update_bias():
    #   "heuristic": v5.4 sign-of-deviation fixed-step update.
    #   "quantile":  bias tracks the (1 - target_frac) quantile of a sliding
    #                window of per-expert routing margins (simplified
    #                quantile estimator; see DeviceLimitedMoE).
    # None -> size default (full: "quantile", lite: "heuristic", i.e. v5.4).
    balance_strategy: Optional[str] = None
    # Expert MLP activation: "swiglu" (SiLU-gated GLU) or "situ"
    # (simplified K3-style SiTU-GLU: tanh soft-capped GLU with an RMSNorm
    # before the output projection).
    activation: str = "swiglu"
    situ_softcap: float = 10.0  # tanh soft-cap magnitude for "situ"
    # v5.68 (roadmap U5, gpt-oss clamped SwiGLU), "swiglu" experts only:
    # a = min(a, limit), b = clamp(b, -limit, limit),
    # out = w2(a * sigmoid(alpha * a) * (b + up_offset)).
    # gpt-oss: limit 7.0, alpha 1.702, up_offset 1.0. The defaults (None,
    # 1.0, 0.0) keep the original w2(silu(a) * b) code path bit for bit.
    swiglu_limit: Optional[float] = None
    swiglu_alpha: float = 1.0
    swiglu_up_offset: float = 0.0
    # v5.63 (roadmap Q3, Qwen3): optional load-balancing loss scope —
    # None (off), "micro" (f_i per forward / micro-batch) or "global"
    # (f_i over every forward since the last balance_loss() call, i.e. the
    # whole global batch). Read with model_v5.balance_loss(model).
    balance_loss_scope: Optional[str] = None
    # v5.65 (roadmap B2-3, GLM-5.3 first 3 layers / MiMo first block):
    # layers 0..first_k_dense-1 use a dense SwiGLU FFN of width
    # ``intermediate_size`` instead of the routed MoE. 0 = off (every
    # layer MoE, bit-identical construction and state dict).
    first_k_dense: int = 0


@dataclass
class HybridAttentionConfig:
    """Hybrid attention interleave (v5.5).

    When enabled, layer i uses MLA iff
    ``i == 0 or i % full_attention_every == full_attention_every - 1``
    and a GatedDeltaAttention (linear attention with a fixed-size recurrent
    state cache) otherwise.

    ``first_layer_full`` (v5.57): True (default) keeps layer 0 MLA as in
    v5.5. False drops the ``i == 0`` exception, giving the Kimi-K3 block
    order (linear, linear, linear, MLA for ``full_attention_every=4``).
    Past length is derived by scanning EVERY layer for a sequence cache
    (``linear_attention.past_seq_len``), so only "at least one MLA layer"
    is required; validation enforces it.
    """
    # None -> follow size (full: True, lite: False, i.e. v5.4 behaviour).
    enabled: Optional[bool] = None
    full_attention_every: int = 4
    # Size-dependent: None = "follow the size" (see MoEConfig).
    linear_num_heads: Optional[int] = None      # full: 32
    linear_head_dim: Optional[int] = None       # full: 128
    # v5.21: KDA-style per-channel decay gate. False keeps the v5.5 per-head
    # scalar gate (bit-identical). True makes the gate emit one sigmoid per
    # head per key channel ([Dk] vector per head), so state rows forget at
    # independent rates — the fine-grained eraser used by Kimi Linear /
    # GLM-5.3-Flash-class hybrid layers. Checkpoint caveat: enabling it on
    # an existing scalar-gate checkpoint fails loudly on the wider
    # g_proj/decay_bias shapes (by design; retrain or remap the gate).
    per_channel_decay: bool = False
    # v5.57: see the class docstring. True = v5.5 rule (bit-identical).
    first_layer_full: bool = True
    # v5.63 (roadmap Q1, Qwen3-Next): causal depthwise short conv (kernel
    # size) + SiLU over the linear-attention q/k/v projections. 0 = off.
    short_conv_kernel: int = 0


@dataclass
class MTPConfig:
    enabled: bool = True
    # Size-dependent: None = "follow the size" (full: 2, lite: 1).
    num_modules: Optional[int] = None
    # v5.67 (roadmap B2-5, MiMo-V2-Flash): "light" MTP blocks use
    # sliding-window causal attention (light_window tokens) in the training
    # path and a narrow dense FFN (light_ffn_size, None = hidden_size)
    # instead of intermediate_size. "default" = the original block.
    style: str = "default"
    light_window: int = 128
    light_ffn_size: Optional[int] = None


@dataclass
class PagedAttentionConfig:
    enabled: bool = True
    block_size: int = 16
    num_blocks: int = 10000


@dataclass
class MultimodalConfig:
    # Size-dependent: None = "follow the size" (full: True, lite: False —
    # the lite model skips the heavy vision/audio encoders).
    enabled: Optional[bool] = None
    vision_patch_size: int = 14
    vision_hidden_size: int = 1024
    vision_num_layers: int = 12
    vision_num_heads: int = 16
    vision_max_grid: int = 32  # max NaViT grid side (patches) per image
    audio_n_mels: int = 80
    audio_hidden_size: int = 512
    audio_num_layers: int = 4


@dataclass
class GRPOConfig:
    enabled: bool = True
    group_size: int = 8
    epsilon: float = 0.2
    kl_coef: float = 0.01
    lr: float = 1e-6
    # v5.64 (roadmap U1, DAPO / Olmo 3 OlmoRL). Every default reproduces
    # the original sequence-level GRPO update bit for bit.
    # "sequence": one ratio per response (summed log-probs), each sample's
    # loss weighted 1/N. "token": per-token ratios and clipping, the loss
    # averaged over every policy token in the batch (DAPO token-level).
    loss_agg: str = "sequence"
    # Clip-higher: upper clip bound 1 + clip_eps_high (None = epsilon).
    clip_eps_high: Optional[float] = None
    # Divide group advantages by the group std (False = Dr. GRPO / Olmo 3).
    normalize_adv_std: bool = True
    # Drop groups whose rewards are all equal (zero advantage) from the
    # loss and its normalization (DAPO dynamic sampling, filtering half).
    filter_zero_adv_groups: bool = False
    # Truncated importance sampling against the SAMPLER's log-probs when a
    # sample carries them ("sampler_logprob" / "sampler_token_logprobs"):
    # weight = min(exp(old - sampler), tis_cap). None = off.
    tis_cap: Optional[float] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        # group_size == 1 makes every group-normalized advantage exactly 0
        # (each sample is its own baseline), so the policy-gradient term
        # vanishes; grpo.py documents group_size >= 2 in practice.
        if self.group_size < 2:
            raise ValueError(
                f"grpo.group_size must be >= 2, got {self.group_size} "
                "(with group_size == 1 every advantage is exactly 0 and "
                "the policy-gradient loss vanishes)"
            )
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError(
                f"grpo.epsilon must be in (0, 1), got {self.epsilon}"
            )
        if self.kl_coef < 0:
            raise ValueError(
                f"grpo.kl_coef must be non-negative, got {self.kl_coef}"
            )
        if self.lr <= 0:
            raise ValueError(f"grpo.lr must be positive, got {self.lr}")
        if self.loss_agg not in ("sequence", "token"):
            raise ValueError(
                f"grpo.loss_agg must be 'sequence' or 'token', got "
                f"{self.loss_agg!r}")
        if self.clip_eps_high is not None and not (0.0 < self.clip_eps_high < 10.0):
            raise ValueError(
                f"grpo.clip_eps_high must be in (0, 10) or None, got "
                f"{self.clip_eps_high!r}")
        for name in ("normalize_adv_std", "filter_zero_adv_groups"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"grpo.{name} must be a bool")
        if self.tis_cap is not None and not self.tis_cap >= 1.0:
            raise ValueError(
                f"grpo.tis_cap must be >= 1 or None, got {self.tis_cap!r}")


class OptionConflictError(ValueError):
    """Raised when an OPTION_RULES rule fails; ``rule`` names the rule.
    A ValueError subclass, so existing ``except ValueError`` code and
    message checks are unaffected."""

    def __init__(self, rule, message):
        super().__init__(message)
        self.rule = rule


@dataclass(frozen=True)
class OptionRule:
    """One cross-option constraint (tech-debt Phase 2). ``kind`` is
    "requires" (when ``when(cfg)`` holds, ``other(cfg)`` must too) or
    "excludes" (``when(cfg)`` and ``other(cfg)`` may not both hold).
    ``message(cfg)`` is the ValueError text."""
    name: str
    kind: str
    when: Callable
    other: Callable
    message: Callable


# Every rule that couples two options lives here and nowhere else:
# HeliosLMv5Config._validate checks them after all per-field checks, and
# T110 (test_option_matrix.py) builds every pair of feature options and
# requires each rejection to come from this table. Add a rule here (not
# inline in _validate) when a new option cannot combine with another.
OPTION_RULES = (
    OptionRule(
        "global_attention_every_needs_window", "requires",
        lambda c: c.attention.global_attention_every is not None,
        lambda c: c.attention.sliding_window is not None,
        lambda c: "attention.global_attention_every requires "
                  "attention.sliding_window (it picks which layers keep "
                  "full attention instead of the window)"),
    OptionRule(
        "window_sink_needs_window", "requires",
        lambda c: c.attention.sliding_window_sink > 0,
        lambda c: c.attention.sliding_window is not None,
        lambda c: "attention.sliding_window_sink > 0 requires "
                  "attention.sliding_window to be set (attention sinks only "
                  "have meaning inside a sliding window)"),
    OptionRule(
        "p_rope_excludes_rope_scaling", "excludes",
        lambda c: c.attention.global_rope_partial < 1.0,
        lambda c: c.attention.rope_scaling is not None,
        lambda c: "attention.global_rope_partial < 1 cannot be combined with "
                  "rope_scaling (YaRN's mscale would rescale the identity dims)"),
    OptionRule(
        "k_eq_v_needs_equal_dims", "requires",
        lambda c: c.attention.global_k_eq_v,
        lambda c: c.attention.v_head_dim == c.attention.no_rope_head_dim,
        lambda c: f"attention.global_k_eq_v needs v_head_dim == no_rope_head_dim, "
                  f"got {c.attention.v_head_dim} vs {c.attention.no_rope_head_dim}"),
    OptionRule(
        "kv_share_excludes_sparse", "excludes",
        lambda c: c.attention.kv_share_last_n > 0,
        lambda c: c.attention.sparse_top_k is not None,
        lambda c: "attention.kv_share_last_n cannot be combined with sparse_top_k"),
    OptionRule(
        "kv_share_excludes_fp8_cache", "excludes",
        lambda c: c.attention.kv_share_last_n > 0,
        lambda c: c.attention.kv_cache_dtype == "fp8",
        lambda c: "attention.kv_share_last_n cannot be combined with "
                  "kv_cache_dtype='fp8'"),
    OptionRule(
        "nope_temperature_needs_nope", "requires",
        lambda c: c.attention.nope_attn_temperature,
        lambda c: c.attention.nope,
        lambda c: "attention.nope_attn_temperature requires attention.nope"),
    OptionRule(
        "sparse_needs_absorption", "requires",
        lambda c: c.attention.sparse_top_k is not None,
        lambda c: c.attention.use_absorption,
        lambda c: "attention.sparse_top_k requires use_absorption=True "
                  "(the top-k selection runs over the shared latent "
                  "cache; the expanded per-head cache is not supported)"),
    OptionRule(
        "learned_indexer_needs_sparse", "requires",
        lambda c: c.attention.sparse_indexer == "learned",
        lambda c: c.attention.sparse_top_k is not None,
        lambda c: "attention.sparse_indexer='learned' requires "
                  "attention.sparse_top_k to be set (the learned lightning "
                  "indexer only exists to pick the top-k set)"),
    OptionRule(
        "indexer_share_needs_learned", "requires",
        lambda c: c.attention.indexer_share_every > 1,
        lambda c: c.attention.sparse_indexer == "learned",
        lambda c: "attention.indexer_share_every > 1 requires "
                  "attention.sparse_indexer='learned' — the 'head_mean' "
                  "indexer has no parameters, so there is nothing to share"),
    OptionRule(
        "sparse_prefill_needs_sparse", "requires",
        lambda c: c.attention.sparse_prefill,
        lambda c: c.attention.sparse_top_k is not None,
        lambda c: "attention.sparse_prefill=True requires "
                  "attention.sparse_top_k to be set"),
    OptionRule(
        "fp8_cache_needs_absorption", "requires",
        lambda c: c.attention.kv_cache_dtype == "fp8",
        lambda c: c.attention.use_absorption,
        lambda c: "attention.kv_cache_dtype='fp8' requires use_absorption=True "
                  "(the fp8 cache stores the compressed latent c_kv; the "
                  "expanded per-head cache has no latent to quantize)"),
    OptionRule(
        "sinkhorn_needs_hyper_connections", "requires",
        lambda c: c.hyper_connection_sinkhorn,
        lambda c: c.use_hyper_connections,
        lambda c: "hyper_connection_sinkhorn requires use_hyper_connections"),
    OptionRule(
        # Both rewire cross-layer residual flow; combining them would
        # silently double-inject previous-layer outputs.
        "hyper_connections_exclude_attention_residuals", "excludes",
        lambda c: c.use_hyper_connections,
        lambda c: c.use_attention_residuals,
        lambda c: "use_hyper_connections and use_attention_residuals are "
                  "mutually exclusive (both rewire cross-layer residual flow); "
                  "enable exactly one"),
)


@dataclass
class HeliosLMv5Config:
    model_name: str = "HeliosLM-v5.49"
    size: str = "full"  # "full" (production defaults) or "lite" (CPU smoke tests)
    # Size-dependent architecture dims: None = "follow the size" (filled
    # from _SIZE_PRESETS in __post_init__; the values below are the "full"
    # production defaults, explicit values always win — see the module
    # docstring).
    vocab_size: Optional[int] = None  # full: 160000
    max_position_embeddings: Optional[int] = None  # full: 1048576
    hidden_size: Optional[int] = None  # full: 4096
    num_hidden_layers: Optional[int] = None  # full: 48
    # Default source for MoE expert hidden size and MTP feed-forward dim.
    intermediate_size: Optional[int] = None  # full: 14336
    rms_norm_eps: float = 1e-6
    eos_token_id: int = 2
    bos_token_id: int = 1
    pad_token_id: int = 0

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    hybrid_attention: HybridAttentionConfig = field(default_factory=HybridAttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    paged_attention: PagedAttentionConfig = field(default_factory=PagedAttentionConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Attention residuals (v5.5): per-layer learnable scalar gate injecting
    # the accumulated sum of previous layers' attention outputs into the
    # attention input. None -> follow size (full: True, lite: False = v5.4).
    use_attention_residuals: Optional[bool] = None

    # Hyper-Connections (v5.7, simplified mHC/HC): replace the single
    # residual stream with N virtual branches per token. Each sublayer reads
    # a static normalized mixing A (identity init, unit-norm columns) and
    # writes back through a learnable zero-init matrix B, so at init the
    # network is exactly a vanilla transformer (B=0 drops the sublayer
    # write). None -> follow size (default OFF for both sizes: it is a
    # training-time architecture switch; the DualPipe schedule and the
    # vLLM-style engine do not support the widened stream and raise loudly).
    use_hyper_connections: Optional[bool] = None
    hyper_connection_branches: int = 4
    # v5.66 (roadmap B2-4, GLM-5.3-Flash mHC): make the HC read matrix
    # learnable and doubly stochastic: A = (1 - a) I + a Sinkhorn(exp(tanh L))
    # with a in [0, 1] learnable (init 0), L learnable (init 0) and squashed
    # by tanh before Sinkhorn (guarantees convergence in 20 steps). At init
    # A == I exactly (bit-identical to the static HC); the Birkhoff
    # polytope is convex, so A stays doubly stochastic (to the Sinkhorn
    # tolerance). Requires use_hyper_connections.
    hyper_connection_sinkhorn: bool = False
    hyper_connection_sinkhorn_iters: int = 20

    # Final logit soft-capping (v5.9, Gemma-2 style): when a positive
    # float, the LM-head logits are soft-capped as
    # ``cap * tanh(logits / cap)`` inside ``HeliosLMv5.forward``, bounding
    # every logit to (-cap, cap) (Gemma-2 used 30.0). This stabilizes the
    # logits against rare activation spikes. None = uncapped (v5.8
    # behaviour, bit-identical).
    final_logit_soft_cap: Optional[float] = None
    # v5.63 (roadmap Q2, Qwen3-Next): zero-centered RMSNorm gains
    # (gain = 1 + w, w init 0) for the layer / final / MLA / QK norms.
    # Bit-identical at init; lets norm weights take weight decay toward 1.
    zero_centered_norm: bool = False

    # Size presets for the architecture fields declared Optional above. A
    # field still None at __post_init__ time ("follow the size") is filled
    # from this table; an explicit non-None value always wins and reaches
    # _validate unchanged. Fields with one value across both sizes (e.g.
    # moe.num_shared_experts) keep plain defaults and are not listed.
    _SIZE_PRESETS = {
        "full": {
            "vocab_size": 160000,
            "max_position_embeddings": 1048576,
            "hidden_size": 4096,
            "num_hidden_layers": 48,
            "intermediate_size": 14336,
            "attention.num_attention_heads": 64,
            "attention.num_key_value_heads": 8,
            "attention.kv_latent_dim": 512,
            "attention.q_lora_rank": 1536,
            "attention.no_rope_head_dim": 128,
            "attention.rope_head_dim": 64,
            "attention.v_head_dim": 128,
            "moe.num_experts": 256,
            "moe.num_activated_experts": 8,
            "moe.device_group_size": 8,
            "mtp.num_modules": 2,
            "hybrid_attention.linear_num_heads": 32,
            "hybrid_attention.linear_head_dim": 128,
            "multimodal.enabled": True,
        },
        "lite": {
            "vocab_size": 1024,
            "max_position_embeddings": 2048,
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "attention.num_attention_heads": 4,
            "attention.num_key_value_heads": 4,
            "attention.kv_latent_dim": 64,
            "attention.q_lora_rank": 128,
            "attention.no_rope_head_dim": 24,
            "attention.rope_head_dim": 8,  # head_dim = 24 + 8 = 32
            "attention.v_head_dim": 32,
            "moe.num_experts": 8,
            "moe.num_activated_experts": 2,
            "moe.device_group_size": 2,
            "mtp.num_modules": 1,
            "hybrid_attention.linear_num_heads": 4,
            "hybrid_attention.linear_head_dim": 32,
            "multimodal.enabled": False,  # skip heavy vision/audio encoders
        },
        # v5.33 "mid": the scale-validation rung (BPE vocab). Purpose
        # (recorded): separate toy artifacts from scale-invariant
        # findings -- acceptance oracles in tests/test_v533_mid.py.
        "mid": {
            "vocab_size": 32768,      # pairs with HeliosBPE (v5.33 P0)
            "max_position_embeddings": 8192,
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "intermediate_size": 2048,
            "attention.num_attention_heads": 16,
            "attention.num_key_value_heads": 4,
            "attention.kv_latent_dim": 256,
            "attention.q_lora_rank": 512,
            "attention.no_rope_head_dim": 64,
            "attention.rope_head_dim": 32,   # head_dim = 64 + 32 = 96
            "attention.v_head_dim": 96,
            "moe.num_experts": 4,
            "moe.num_activated_experts": 2,
            "moe.device_group_size": 4,
            "mtp.num_modules": 1,
            "hybrid_attention.linear_num_heads": 8,
            "hybrid_attention.linear_head_dim": 64,
            "multimodal.enabled": False,
        },
    }

    def __post_init__(self):
        if self.size not in ("full", "lite", "mid"):
            raise ValueError(
                f"unknown size {self.size!r}; expected 'full', 'mid' or 'lite'")
        self._apply_size_defaults()
        self._resolve_defaults()
        self._validate()

    def _apply_size_defaults(self):
        """Fill size-dependent fields left at None ("follow the size").

        Only fields the caller did NOT set (still None) are replaced by the
        size preset; explicit values always win and flow into _validate
        unchanged (see the module docstring and _SIZE_PRESETS).
        """
        for dotted, value in self._SIZE_PRESETS[self.size].items():
            obj = self
            for part in dotted.split(".")[:-1]:
                obj = getattr(obj, part)
            name = dotted.rsplit(".", 1)[-1]
            if getattr(obj, name) is None:
                setattr(obj, name, value)

    def _resolve_defaults(self):
        if self.moe.expert_hidden_size is None:
            self.moe.expert_hidden_size = self.intermediate_size
        # v5.5 size-dependent defaults (None = follow size; explicit values win).
        if self.hybrid_attention.enabled is None:
            self.hybrid_attention.enabled = self.size == "full"
        if self.use_attention_residuals is None:
            self.use_attention_residuals = self.size == "full"
        if self.moe.latent_dim is None:
            # LatentMoE on by default for the full size; the lite size keeps
            # full-width routed experts (v5.4 behaviour). A non-positive
            # value explicitly requests full-width experts.
            self.moe.latent_dim = 1024 if self.size == "full" else None
        if self.moe.balance_strategy is None:
            # Quantile balancing on by default for the full size; the lite
            # size keeps the v5.4 heuristic update.
            self.moe.balance_strategy = "quantile" if self.size == "full" else "heuristic"
        if self.use_hyper_connections is None:
            # v5.7: opt-in for both sizes (training-time architecture switch).
            self.use_hyper_connections = False
        if self.moe.latent_dim is not None and self.moe.latent_dim <= 0:
            self.moe.latent_dim = None  # explicit full-width opt-out

    def _validate(self):
        a, m = self.attention, self.moe
        # Uniform positivity checks first (clear per-field messages), then
        # the structural constraints below.
        positive_ints = {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "attention.num_attention_heads": a.num_attention_heads,
            "attention.num_key_value_heads": a.num_key_value_heads,
            "moe.num_shared_experts": m.num_shared_experts,
            "moe.expert_hidden_size": m.expert_hidden_size,
        }
        for name, value in positive_ints.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if m.bias_update_rate <= 0:
            raise ValueError(
                f"moe.bias_update_rate must be positive, got {m.bias_update_rate}"
            )
        # MTP's TransformerEncoderLayer requires embed_dim % nhead == 0; MLA
        # itself is decoupled from hidden_size via explicit head dims.
        if self.hidden_size % a.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({a.num_attention_heads})"
            )
        if a.rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim ({a.rope_head_dim}) must be even for RoPE")
        if a.no_rope_head_dim <= 0 or a.rope_head_dim <= 0 or a.v_head_dim <= 0:
            raise ValueError("no_rope_head_dim/rope_head_dim/v_head_dim must be positive")
        if a.kv_latent_dim <= 0 or a.q_lora_rank <= 0:
            raise ValueError("kv_latent_dim and q_lora_rank must be positive")
        if m.num_experts <= 0 or m.device_group_size <= 0:
            raise ValueError("num_experts and device_group_size must be positive")
        if m.num_experts % m.device_group_size != 0:
            raise ValueError(
                f"num_experts ({m.num_experts}) must be divisible by "
                f"device_group_size ({m.device_group_size})"
            )
        if not (1 <= m.num_activated_experts <= m.num_experts):
            raise ValueError(
                f"num_activated_experts ({m.num_activated_experts}) must be in "
                f"[1, num_experts={m.num_experts}]"
            )
        # Dropout is a probability.
        if not (0.0 <= a.attention_dropout <= 1.0):
            raise ValueError(
                f"attention.attention_dropout must be in [0, 1], got "
                f"{a.attention_dropout}"
            )
        # MTP needs at least one module when enabled.
        if self.mtp.num_modules <= 0:
            raise ValueError(
                f"mtp.num_modules must be a positive integer, got "
                f"{self.mtp.num_modules}"
            )
        # GRPO training hyper-parameters (validated eagerly in GRPOConfig
        # too; re-checked here for configs mutated after construction).
        self.grpo._validate()
        # v5.5: hybrid interleave / LatentMoE / balancing / activation.
        h = self.hybrid_attention
        if h.full_attention_every < 2:
            raise ValueError(
                f"hybrid_attention.full_attention_every must be >= 2 "
                f"(a value of 1 would make every layer MLA), got "
                f"{h.full_attention_every}"
            )
        k = h.short_conv_kernel
        if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k == 1:
            raise ValueError(
                f"hybrid_attention.short_conv_kernel must be 0 (off) or an "
                f"int >= 2, got {k!r}")
        if not isinstance(h.first_layer_full, bool):
            raise ValueError(
                f"hybrid_attention.first_layer_full must be a bool, got "
                f"{h.first_layer_full!r}")
        if (h.enabled and not h.first_layer_full
                and self.num_hidden_layers < h.full_attention_every):
            raise ValueError(
                f"hybrid_attention.first_layer_full=False with "
                f"num_hidden_layers={self.num_hidden_layers} < "
                f"full_attention_every={h.full_attention_every} leaves no "
                f"MLA layer: the cache would carry no sequence axis to "
                f"derive the past length from")
        ge = self.attention.global_attention_every
        if ge is not None:
            if not isinstance(ge, int) or isinstance(ge, bool) or ge < 2:
                raise ValueError(
                    f"attention.global_attention_every must be an int >= 2 "
                    f"or None, got {ge!r}")
        if not isinstance(self.attention.learnable_sink, bool):
            raise ValueError(
                f"attention.learnable_sink must be a bool, got "
                f"{self.attention.learnable_sink!r}")
        k = self.moe.first_k_dense
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(
                f"moe.first_k_dense must be an int >= 0, got {k!r}")
        if k > self.num_hidden_layers:
            raise ValueError(
                f"moe.first_k_dense ({k}) exceeds num_hidden_layers "
                f"({self.num_hidden_layers})")
        if self.mtp.style not in ("default", "light"):
            raise ValueError(
                f"mtp.style must be 'default' or 'light', got {self.mtp.style!r}")
        w = self.mtp.light_window
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            raise ValueError(f"mtp.light_window must be an int >= 1, got {w!r}")
        f = self.mtp.light_ffn_size
        if f is not None and (isinstance(f, bool) or not isinstance(f, int) or f < 1):
            raise ValueError(
                f"mtp.light_ffn_size must be a positive int or None, got {f!r}")
        if self.moe.balance_loss_scope not in (None, "micro", "global"):
            raise ValueError(
                f"moe.balance_loss_scope must be None, 'micro' or 'global', "
                f"got {self.moe.balance_loss_scope!r}")
        if not isinstance(self.zero_centered_norm, bool):
            raise ValueError(
                f"zero_centered_norm must be a bool, got "
                f"{self.zero_centered_norm!r}")
        a_ = self.attention
        rp = a_.global_rope_partial
        if isinstance(rp, bool) or not isinstance(rp, (int, float)) \
                or not 0.0 <= rp <= 1.0:
            raise ValueError(
                f"attention.global_rope_partial must be in [0, 1], got {rp!r}")
        if not isinstance(a_.global_k_eq_v, bool):
            raise ValueError("attention.global_k_eq_v must be a bool")
        ks = a_.kv_share_last_n
        if isinstance(ks, bool) or not isinstance(ks, int) or ks < 0:
            raise ValueError(
                f"attention.kv_share_last_n must be an int >= 0, got {ks!r}")
        if not isinstance(a_.nope_attn_temperature, bool):
            raise ValueError("attention.nope_attn_temperature must be a bool")
        fs = a_.nope_temp_floor_scale
        if isinstance(fs, bool) or not isinstance(fs, int) or fs < 1:
            raise ValueError(
                f"attention.nope_temp_floor_scale must be an int >= 1, got {fs!r}")
        ts = a_.nope_temp_scale
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts < 0:
            raise ValueError(
                f"attention.nope_temp_scale must be a number >= 0, got {ts!r}")
        if not isinstance(self.attention.output_gate, bool):
            raise ValueError(
                f"attention.output_gate must be a bool, got "
                f"{self.attention.output_gate!r}")
        if h.linear_num_heads <= 0 or h.linear_head_dim <= 0:
            raise ValueError(
                f"hybrid_attention.linear_num_heads/linear_head_dim must be "
                f"positive, got {h.linear_num_heads}/{h.linear_head_dim}"
            )
        # m9: no latent_dim<=0 check here — _resolve_defaults already
        # normalizes a non-positive latent_dim to None (the documented
        # explicit full-width opt-out), so by validation time latent_dim is
        # either None or a positive int.
        if m.balance_strategy not in ("heuristic", "quantile"):
            raise ValueError(
                f"moe.balance_strategy must be 'heuristic' or 'quantile', "
                f"got {m.balance_strategy!r}"
            )
        if m.activation not in ("swiglu", "situ"):
            raise ValueError(
                f"moe.activation must be 'swiglu' or 'situ', got "
                f"{m.activation!r}"
            )
        if m.swiglu_limit is not None and not (
                isinstance(m.swiglu_limit, (int, float))
                and not isinstance(m.swiglu_limit, bool) and m.swiglu_limit > 0):
            raise ValueError(
                f"moe.swiglu_limit must be a positive number or None, got "
                f"{m.swiglu_limit!r}")
        if isinstance(m.swiglu_alpha, bool) or not isinstance(
                m.swiglu_alpha, (int, float)) or m.swiglu_alpha <= 0:
            raise ValueError(
                f"moe.swiglu_alpha must be a positive number, got "
                f"{m.swiglu_alpha!r}")
        if isinstance(m.swiglu_up_offset, bool) or not isinstance(
                m.swiglu_up_offset, (int, float)):
            raise ValueError(
                f"moe.swiglu_up_offset must be a number, got "
                f"{m.swiglu_up_offset!r}")
        if m.situ_softcap <= 0:
            raise ValueError(
                f"moe.situ_softcap must be positive, got {m.situ_softcap}"
            )
        # v5.7/v5.8: RoPE scaling, KV-cache dtype, sparse top-k.
        rs = a.rope_scaling
        if rs is not None:
            if not isinstance(rs, dict):
                raise ValueError(
                    f"attention.rope_scaling must be a dict like "
                    f"{{'type': 'linear'|'ntk'|'yarn', 'factor': f}}, got "
                    f"{rs!r}"
                )
            if rs.get("type") not in ("linear", "ntk", "yarn"):
                raise ValueError(
                    f"attention.rope_scaling['type'] must be 'linear', "
                    f"'ntk' or 'yarn', got {rs.get('type')!r}"
                )
            factor = rs.get("factor")
            if factor is None or not (isinstance(factor, (int, float))) \
                    or factor < 1.0:
                raise ValueError(
                    f"attention.rope_scaling['factor'] must be a number "
                    f">= 1.0, got {factor!r}"
                )
        if a.sparse_top_k is not None:
            if not isinstance(a.sparse_top_k, int) or a.sparse_top_k <= 0:
                raise ValueError(
                    f"attention.sparse_top_k must be a positive integer or "
                    f"None, got {a.sparse_top_k!r}"
                )
        if a.sparse_indexer not in ("head_mean", "learned"):
            raise ValueError(
                f"attention.sparse_indexer must be 'head_mean' or "
                f"'learned', got {a.sparse_indexer!r}"
            )
        for _name in ("indexer_num_heads", "indexer_head_dim",
                      "indexer_share_every"):
            _value = getattr(a, _name)
            if not isinstance(_value, int) or _value <= 0:
                raise ValueError(
                    f"attention.{_name} must be a positive integer, got "
                    f"{_value!r}"
                )
        if not isinstance(a.sparse_prefill, bool):
            raise ValueError(
                f"attention.sparse_prefill must be a bool, got "
                f"{a.sparse_prefill!r}"
            )
        if a.kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(
                f"attention.kv_cache_dtype must be 'auto' or 'fp8', got "
                f"{a.kv_cache_dtype!r}"
            )
        # v5.9: soft-capping, QK-norm, sliding window.
        if a.logit_soft_cap is not None:
            if not isinstance(a.logit_soft_cap, (int, float)) \
                    or a.logit_soft_cap <= 0:
                raise ValueError(
                    f"attention.logit_soft_cap must be a positive number or "
                    f"None, got {a.logit_soft_cap!r}"
                )
        if not isinstance(a.qk_norm, bool):
            raise ValueError(
                f"attention.qk_norm must be a bool, got {a.qk_norm!r}"
            )
        if a.sliding_window is not None:
            if not isinstance(a.sliding_window, int) or a.sliding_window <= 0:
                raise ValueError(
                    f"attention.sliding_window must be a positive integer or "
                    f"None, got {a.sliding_window!r}"
                )
        if not isinstance(a.sliding_window_sink, int) or a.sliding_window_sink < 0:
            raise ValueError(
                f"attention.sliding_window_sink must be a non-negative "
                f"integer, got {a.sliding_window_sink!r}"
            )
        # Hyper-Connections (v5.7): branch count and mutual exclusion with
        # the v5.5 attention-residual accumulator (both rewire how sublayer
        # outputs flow across the layer stack; combining them is untested
        # and would silently double-inject previous-layer outputs).
        if self.hyper_connection_branches < 2:
            raise ValueError(
                f"hyper_connection_branches must be >= 2, got "
                f"{self.hyper_connection_branches}"
            )
        if not isinstance(self.hyper_connection_sinkhorn, bool):
            raise ValueError("hyper_connection_sinkhorn must be a bool")
        it = self.hyper_connection_sinkhorn_iters
        if isinstance(it, bool) or not isinstance(it, int) or it < 1:
            raise ValueError(
                f"hyper_connection_sinkhorn_iters must be an int >= 1, "
                f"got {it!r}")
        # Vision transformer heads must tile the hidden width exactly.
        vision_hidden = getattr(self.multimodal, "vision_hidden_size", None)
        vision_heads = getattr(self.multimodal, "vision_num_heads", None)
        if vision_hidden is not None and vision_heads:
            if vision_hidden % vision_heads != 0:
                raise ValueError(
                    f"multimodal.vision_hidden_size ({vision_hidden}) must be "
                    f"divisible by multimodal.vision_num_heads ({vision_heads})"
                )
        # Paged-attention block pool must be non-empty.
        pa = self.paged_attention
        if pa.block_size <= 0 or pa.num_blocks <= 0:
            raise ValueError(
                f"paged_attention.block_size and paged_attention.num_blocks "
                f"must be positive, got block_size={pa.block_size}, "
                f"num_blocks={pa.num_blocks}"
            )
        # Special token ids must address real embedding rows.
        for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
            tid = getattr(self, name)
            if not (0 <= tid < self.vocab_size):
                raise ValueError(
                    f"{name} ({tid}) must be in [0, vocab_size="
                    f"{self.vocab_size})"
                )
        # v5.9: final logit soft-capping.
        if self.final_logit_soft_cap is not None:
            if not isinstance(self.final_logit_soft_cap, (int, float)) \
                    or self.final_logit_soft_cap <= 0:
                raise ValueError(
                    f"final_logit_soft_cap must be a positive number or "
                    f"None, got {self.final_logit_soft_cap!r}"
                )
        # Cross-option constraints: one table (OPTION_RULES), checked after
        # every per-field check so the predicates see well-typed values.
        for rule in OPTION_RULES:
            on, other = rule.when(self), rule.other(self)
            if on and (other if rule.kind == "excludes" else not other):
                raise OptionConflictError(rule.name, rule.message(self))


def k3_block_config(size="lite", num_hidden_layers=None):
    """Kimi-K3-style block preset (v5.57, roadmap Phase B).

    Sets the four K3 ingredients this stack implements (Kimi K3
    technical report, arXiv 2607.24653):

    - 3:1 linear/global interleave in K3 order: KDA, KDA, KDA, MLA
      (``full_attention_every=4``, ``first_layer_full=False``);
    - KDA = GatedDeltaAttention with the per-channel decay gate
      (``per_channel_decay=True``);
    - Gated MLA (``attention.output_gate=True``, ``MLA.o_gate_proj``);
    - Attention Residuals (``use_attention_residuals=True``) and
      LatentMoE (``moe.latent_dim``; the full size keeps its 1024
      default, smaller sizes use ``hidden_size // 2``).

    Not reproduced: K3's "Stable" LatentMoE variant and its exact
    hyper-parameters, FP4 training, and scale. ``num_hidden_layers``
    defaults to the size preset, except "lite" (2 layers) is raised to 4
    so one full K3 block exists. Everything else follows ``size``;
    mutate the returned config and call ``_validate()`` to adjust.
    """
    if size == "lite" and num_hidden_layers is None:
        num_hidden_layers = 4
    cfg = HeliosLMv5Config(
        size=size,
        num_hidden_layers=num_hidden_layers,
        use_attention_residuals=True,
        attention=AttentionConfig(output_gate=True),
        hybrid_attention=HybridAttentionConfig(
            enabled=True, full_attention_every=4, first_layer_full=False,
            per_channel_decay=True),
    )
    if cfg.moe.latent_dim is None:
        cfg.moe.latent_dim = cfg.hidden_size // 2
        cfg._validate()
    return cfg


def mimo_hybrid_config(size="lite", window=128, num_hidden_layers=None,
                       learnable_sink=True):
    """MiMo-V2-Flash-style hybrid attention preset (v5.62, roadmap B2).

    - every MLA layer gets a ``window``-token sliding window except the
      global ones: layer i is global iff ``i == 0 or i % 6 == 5`` (5
      sliding : 1 global, global first layer; MiMo-V2-Flash uses W=128);
    - learnable per-head attention sink logits (``learnable_sink``).

    Not reproduced: MiMo's dense-FFN first block (roadmap B2-3), its MTP
    head (B2-5), and scale. ``num_hidden_layers`` defaults to the size
    preset, except "lite" (2 layers) is raised to 6 so one full
    sliding/global pattern exists. The linear-attention hybrid stays
    off: every layer is MLA, as in MiMo.
    """
    if size == "lite" and num_hidden_layers is None:
        num_hidden_layers = 6
    return HeliosLMv5Config(
        size=size,
        num_hidden_layers=num_hidden_layers,
        attention=AttentionConfig(sliding_window=window,
                                  global_attention_every=6,
                                  learnable_sink=learnable_sink),
        hybrid_attention=HybridAttentionConfig(enabled=False),
    )


def qwen_next_config(size="lite", num_hidden_layers=None):
    """Qwen3-Next-style preset (v5.63, roadmap Q4).

    - 3:1 linear/global interleave in Qwen3-Next order: GDA, GDA, GDA,
      gated attention (``full_attention_every=4``, ``first_layer_full=
      False``), with the causal short conv on the GDA q/k/v (Q1,
      ``short_conv_kernel=4``);
    - gated attention (``attention.output_gate=True``) with QK-norm;
    - zero-centered RMSNorm gains (Q2, ``zero_centered_norm=True``);
    - ultra-sparse MoE with one shared expert: "full" uses 512 routed
      experts, top-10 (Qwen3-Next-80B-A3B's ratio, ~2% active), expert
      width ``hidden_size // 4``; smaller sizes use 32 experts, top-2;
    - global-batch load-balancing loss (Q3,
      ``moe.balance_loss_scope="global"``) on top of the bias balancing;
    - MTP stays on (the size default).

    Not reproduced: Qwen3-Next's exact widths, its partial RoPE on the
    attention layers (MLA already rotates only ``rope_head_dim`` of each
    head), training recipe and scale. "lite" (2 layers) is raised to 4 so
    one full block exists. Mutate the result and call ``_validate()`` to
    adjust.
    """
    if size == "lite" and num_hidden_layers is None:
        num_hidden_layers = 4
    full = size == "full"
    cfg = HeliosLMv5Config(
        size=size,
        num_hidden_layers=num_hidden_layers,
        zero_centered_norm=True,
        attention=AttentionConfig(output_gate=True, qk_norm=True),
        hybrid_attention=HybridAttentionConfig(
            enabled=True, full_attention_every=4, first_layer_full=False,
            short_conv_kernel=4),
        moe=MoEConfig(num_experts=512 if full else 32,
                      num_activated_experts=10 if full else 2,
                      num_shared_experts=1,
                      balance_loss_scope="global"),
    )
    cfg.moe.expert_hidden_size = cfg.hidden_size // 4
    cfg._validate()
    return cfg
