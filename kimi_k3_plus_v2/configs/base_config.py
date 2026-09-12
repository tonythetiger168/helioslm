"""
Kimi K3+ Unified Base Configuration
一个架构，四个尺寸
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class AttentionConfig:
    type: str = "KimiDeltaAttention"
    kda_ratio: str = "3:1"
    use_nope: bool = True
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    attn_res_enabled: bool = True
    residual_depth: int = 4
    use_gated_mla: bool = True
    attention_dropout: float = 0.0


@dataclass
class MoEConfig:
    type: str = "StableLatentMoE"
    num_experts: int = 896
    num_shared_experts: int = 4
    num_activated_experts: int = 16
    expert_hidden_size: int = 3584
    dynamic_sparsity_enabled: bool = True
    min_experts: int = 8
    max_experts: int = 16
    load_balance_loss_coef: float = 0.01
    top_k: int = 2
    capacity_factor: float = 1.25
    activation: str = "SiTU_GLU"


@dataclass
class SpeculativeDecodingConfig:
    enabled: bool = True
    draft_model_size: str = "30B"
    draft_layers: int = 32
    draft_hidden_size: int = 4096
    draft_num_experts: int = 64
    draft_num_activated: int = 4
    max_draft_tokens: int = 5
    acceptance_threshold: float = 0.8
    tree_attention: bool = True


@dataclass
class QuantizationConfig:
    cloud_weight_bits: int = 4
    cloud_activation_bits: int = 8
    cloud_format: str = "MXFP"
    edge_weight_bits: int = 4
    edge_activation_bits: int = 4
    edge_format: str = "AWQ"
    mobile_weight_bits: int = 3
    mobile_activation_bits: int = 4
    mobile_format: str = "GPTQ"
    kv_cache_compression: bool = True
    kv_cache_bits: int = 4


@dataclass
class MultimodalConfig:
    enabled: bool = True
    vision_encoder: str = "ViT-Giant"
    vision_image_size: int = 448
    vision_patch_size: int = 14
    vision_hidden_size: int = 1536
    video_temporal_resolution: int = 32
    video_frame_sample_rate: int = 2
    video_motion_aware: bool = True
    audio_sampling_rate: int = 16000
    audio_n_mels: int = 128
    audio_enable_emotion: bool = True
    audio_enable_speaker_id: bool = True
    cross_modal_temperature: float = 0.07


@dataclass
class AgenticConfig:
    enabled: bool = True
    mcp_enabled: bool = True
    max_tools_per_turn: int = 8
    tool_timeout_seconds: int = 30
    long_term_memory_enabled: bool = True
    memory_max_entries: int = 100000
    memory_compression_ratio: float = 0.1
    self_reflection_enabled: bool = True
    max_reflection_depth: int = 3
    task_planning_horizon: str = "7_days"
    checkpoint_interval: str = "1_hour"


@dataclass
class KimiK3PlusBaseConfig:
    """统一架构基础配置"""
    model_name: str = "Kimi-K3-Plus"
    version: str = "2.0.0"
    vocab_size: int = 160000
    max_position_embeddings: int = 1048576
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    intermediate_size: int = 28672
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    speculative: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
