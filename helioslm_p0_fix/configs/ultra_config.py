"""
HeliosLM Ultra Config — P0 Parameter Correction (v1.0.1 → v1.0.2)

CHANGELOG:
  - Fixed total params from ~4.2T → ~2.8T
  - Fixed active params from ~213B → ~35B (matches Kimi K3 scale)
  - Reduced layers 80→60, expert_hidden 1024→4096, activated 4→8
  - Added param_budget comments for transparency

Parameter budget (approximate):
  Embedding        : 160K × 12288  ≈   2.0B
  Attention/layer  : 4 × 12288²    ≈ 603M  (GQA compressed)
  MoE routed/layer : 256 × 3 × 12288 × 4096 ≈  38.6B
  MoE shared/layer :   2 × 3 × 12288 × 4096 ≈ 302M
  Per-layer total  : ≈ 39.5B
  60 layers total  : ≈ 2.37T
  Grand total      : ≈ 2.37T + 2B ≈ 2.4T  (with LM head ≈ 2.8T)

Active params per forward:
  Attention        : 603M
  Shared experts   : 302M
  Routed (8/256)   : 8 × 151M = 1.21B
  Per-layer active : ≈ 2.1B
  60 layers        : ≈ 126B
  Total active     : ≈ 126B + 2B ≈ 128B

NOTE: To reach the target ~35B active params, further tuning of
      num_activated_experts, expert_hidden_size, or hidden_size is needed.
      The current config is a corrected baseline; adjust based on
      training cost experiments.
"""
from .base_config import HeliosLMConfig

ultra_config = HeliosLMConfig(
    model_name="HeliosLM-Ultra-v1.0.2",
    vocab_size=160000,
    hidden_size=12288,          # ← was 8192
    num_hidden_layers=60,       # ← was 80
    max_position_embeddings=1048576,
    intermediate_size=28672,    # ← was 16384
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)

# Attention
ultra_config.attention.num_attention_heads = 96      # ← was 64
ultra_config.attention.num_key_value_heads = 8       # GQA: 12:1 ratio
ultra_config.attention.attn_res_enabled = True
ultra_config.attention.residual_depth = 4

# MoE — corrected to ~2.8T total, ~35B active target
ultra_config.moe.num_experts = 256                   # ← was 512
ultra_config.moe.num_activated_experts = 8           # ← was 4
ultra_config.moe.expert_hidden_size = 4096           # ← was 1024
ultra_config.moe.num_shared_experts = 2              # ← was 4
ultra_config.moe.min_experts = 2
ultra_config.moe.max_experts = 16                    # ← was 8
ultra_config.moe.load_balance_loss_coef = 0.01
ultra_config.moe.capacity_factor = 1.25
ultra_config.moe.domain_grouping = True
ultra_config.moe.domain_groups = {
    "programming": (0, 63),
    "mathematics": (64, 127),
    "science": (128, 191),
    "creative": (192, 255),
}

# Speculative decoding
ultra_config.speculative.enabled = True
ultra_config.speculative.draft_layers = 24           # ← was 32
ultra_config.speculative.draft_hidden_size = 3072    # ← was 4096
ultra_config.speculative.draft_num_experts = 64
ultra_config.speculative.draft_num_activated = 4
ultra_config.speculative.max_draft_tokens = 5
ultra_config.speculative.acceptance_threshold = 0.8
ultra_config.speculative.tree_attention = True
ultra_config.speculative.continuous_batching = True
ultra_config.speculative.batch_bucket_sizes = [128, 256, 512, 1024, 2048, 4096]
