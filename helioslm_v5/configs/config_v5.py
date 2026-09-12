"""HeliosLM v5.4 Configuration

Two sizes are supported via ``HeliosLMv5Config(size=...)``:
  - ``"lite"``: small CPU-friendly config for smoke tests (seconds per step).
  - ``"full"``: production-scale defaults.

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
    model_name: str = "HeliosLM-v5.4"
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
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    paged_attention: PagedAttentionConfig = field(default_factory=PagedAttentionConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

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

        self.mtp.num_modules = 1
        # Keep the lite model small: skip building heavy vision/audio encoders.
        self.multimodal.enabled = False

    def _resolve_defaults(self):
        if self.moe.expert_hidden_size is None:
            self.moe.expert_hidden_size = self.intermediate_size

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
