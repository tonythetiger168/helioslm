"""HeliosLM v5.19 Configuration

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

``None`` on ``hybrid.enabled`` / ``use_attention_residuals`` /
``moe.latent_dim`` means "follow the size default": the "full" size turns
hybrid / residuals on and sets latent_dim=1024; the "lite" size keeps the
v5.4 behaviour (all off / full-width experts) so the existing test suite is
bit-identical. Explicit True/False/int values always win.

``__post_init__`` validates structural constraints (divisibility etc.) and
raises ``ValueError`` immediately on an invalid configuration instead of
letting silent truncation happen downstream.
"""
from dataclasses import dataclass, field
from typing import Optional


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
    num_attention_heads: int = 64  # was 96, but 4096 % 96 != 0 (M4)
    num_key_value_heads: int = 8   # GQA baseline reference for cache-size stats
    kv_latent_dim: int = 512       # d_c: KV compression latent dim
    q_lora_rank: int = 1536        # d_c': Q compression latent dim
    no_rope_head_dim: int = 128    # per-head q/k dims WITHOUT RoPE
    rope_head_dim: int = 64        # per-head q/k dims WITH RoPE (shared k_rope)
    v_head_dim: int = 128          # per-head value dim (never rotated)
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

    @property
    def head_dim(self) -> int:
        return self.no_rope_head_dim + self.rope_head_dim


@dataclass
class MoEConfig:
    num_experts: int = 256
    num_shared_experts: int = 1
    num_activated_experts: int = 8  # top-k
    # None -> defaults to HeliosLMv5Config.intermediate_size (resolved in
    # HeliosLMv5Config.__post_init__)
    expert_hidden_size: Optional[int] = None
    # Number of devices across which experts are partitioned for
    # device-limited routing (experts_per_device = num_experts / device_group_size).
    device_group_size: int = 8
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


@dataclass
class HybridAttentionConfig:
    """Hybrid attention interleave (v5.5).

    When enabled, layer i uses MLA iff
    ``i == 0 or i % full_attention_every == full_attention_every - 1``
    and a GatedDeltaAttention (linear attention with a fixed-size recurrent
    state cache) otherwise. Layer 0 is ALWAYS MLA, so
    ``past_key_values[0]`` keeps the dim-2-sequence layout that
    ``HeliosLMv5.forward`` uses to derive past length.
    """
    # None -> follow size (full: True, lite: False, i.e. v5.4 behaviour).
    enabled: Optional[bool] = None
    full_attention_every: int = 4
    linear_num_heads: int = 32
    linear_head_dim: int = 128


@dataclass
class MTPConfig:
    enabled: bool = True
    num_modules: int = 2


@dataclass
class PagedAttentionConfig:
    enabled: bool = True
    block_size: int = 16
    num_blocks: int = 10000


@dataclass
class MultimodalConfig:
    enabled: bool = True
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


@dataclass
class HeliosLMv5Config:
    model_name: str = "HeliosLM-v5.19"
    size: str = "full"  # "full" (production defaults) or "lite" (CPU smoke tests)
    vocab_size: int = 160000
    max_position_embeddings: int = 1048576
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    # Default source for MoE expert hidden size and MTP feed-forward dim.
    intermediate_size: int = 14336
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

    # Final logit soft-capping (v5.9, Gemma-2 style): when a positive
    # float, the LM-head logits are soft-capped as
    # ``cap * tanh(logits / cap)`` inside ``HeliosLMv5.forward``, bounding
    # every logit to (-cap, cap) (Gemma-2 used 30.0). This stabilizes the
    # logits against rare activation spikes. None = uncapped (v5.8
    # behaviour, bit-identical).
    final_logit_soft_cap: Optional[float] = None

    def __post_init__(self):
        if self.size not in ("full", "lite"):
            raise ValueError(f"unknown size {self.size!r}; expected 'full' or 'lite'")
        if self.size == "lite":
            self._apply_lite_overrides()
        self._resolve_defaults()
        self._validate()

    def _apply_lite_overrides(self):
        """Small CPU-runnable config: forward/backward in seconds."""
        self.vocab_size = 1024
        self.max_position_embeddings = 2048
        self.hidden_size = 256
        self.num_hidden_layers = 2
        self.intermediate_size = 512

        self.attention.num_attention_heads = 4
        self.attention.num_key_value_heads = 4
        self.attention.kv_latent_dim = 64
        self.attention.q_lora_rank = 128
        self.attention.no_rope_head_dim = 24
        self.attention.rope_head_dim = 8  # head_dim = 24 + 8 = 32
        self.attention.v_head_dim = 32

        self.moe.num_experts = 8
        self.moe.num_shared_experts = 1
        self.moe.num_activated_experts = 2
        self.moe.expert_hidden_size = None  # -> intermediate_size
        self.moe.device_group_size = 2

        # Small hybrid linear-attention dims (used only when a test enables
        # hybrid_attention explicitly).
        self.hybrid_attention.linear_num_heads = 4
        self.hybrid_attention.linear_head_dim = 32

        self.mtp.num_modules = 1
        # Keep the lite model small: skip building heavy vision/audio encoders.
        self.multimodal.enabled = False

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
        # v5.5: hybrid interleave / LatentMoE / balancing / activation.
        h = self.hybrid_attention
        if h.full_attention_every < 2:
            raise ValueError(
                f"hybrid_attention.full_attention_every must be >= 2 "
                f"(layer 0 is always MLA), got {h.full_attention_every}"
            )
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
            if not a.use_absorption:
                raise ValueError(
                    "attention.sparse_top_k requires use_absorption=True "
                    "(the top-k selection runs over the shared latent "
                    "cache; the expanded per-head cache is not supported)"
                )
        if a.kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError(
                f"attention.kv_cache_dtype must be 'auto' or 'fp8', got "
                f"{a.kv_cache_dtype!r}"
            )
        if a.kv_cache_dtype == "fp8" and not a.use_absorption:
            raise ValueError(
                "attention.kv_cache_dtype='fp8' requires use_absorption=True "
                "(the fp8 cache stores the compressed latent c_kv; the "
                "expanded per-head cache has no latent to quantize)"
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
        if a.sliding_window_sink > 0 and a.sliding_window is None:
            raise ValueError(
                "attention.sliding_window_sink > 0 requires "
                "attention.sliding_window to be set (attention sinks only "
                "have meaning inside a sliding window)"
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
        if self.use_hyper_connections and self.use_attention_residuals:
            raise ValueError(
                "use_hyper_connections and use_attention_residuals are "
                "mutually exclusive (both rewire cross-layer residual flow); "
                "enable exactly one"
            )
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
