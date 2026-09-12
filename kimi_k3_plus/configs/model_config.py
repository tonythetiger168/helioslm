"""
Kimi K3+ Model Configuration
基于 Kimi K3 框架的增强配置
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class AttentionConfig:
    """注意力机制配置"""
    type: str = "KimiDeltaAttention"
    kda_ratio: str = "3:1"
    use_nope: bool = True
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
    attn_res_enabled: bool = True
    residual_depth: int = 4
    use_gated_mla: bool = True


@dataclass
class MoEConfig:
    """混合专家模型配置"""
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
class MultimodalConfig:
    """多模态配置"""
    enabled: bool = True
    modalities: List[str] = field(default_factory=lambda: ["text", "image", "video", "audio"])


@dataclass
class KimiK3PlusConfig:
    """Kimi K3+ 主配置"""
    model_name: str = "Kimi-K3-Plus"
    version: str = "1.0.0"
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
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)


default_config = KimiK3PlusConfig()
