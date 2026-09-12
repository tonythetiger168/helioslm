"""Kimi K3+ v3.0 Config - P0+P1+P2+P3+P4"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
    domain_grouping: bool = True
    domain_groups: Dict[str, tuple] = field(default_factory=lambda: {
        "programming": (0, 223), "mathematics": (224, 447),
        "science": (448, 671), "creative": (672, 895),
    })

@dataclass
class SpeculativeConfig:
    enabled: bool = True
    draft_model_size: str = "30B"
    draft_layers: int = 32
    draft_hidden_size: int = 4096
    draft_num_experts: int = 64
    draft_num_activated: int = 4
    max_draft_tokens: int = 5
    acceptance_threshold: float = 0.8
    tree_attention: bool = True
    continuous_batching: bool = True
    batch_bucket_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])

@dataclass
class RAGConfig:
    enabled: bool = True
    retrieval_k: int = 5
    confidence_threshold: float = 0.7
    force_retrieval: bool = True
    multi_model_verification: bool = True
    fact_check_db: str = "faiss"
    uncertainty_phrase: str = "[不確定]"

@dataclass
class CoTConfig:
    enabled: bool = True
    format: str = "structured_xml"
    force_steps: bool = True
    max_reasoning_tokens: int = 4096
    tool_use_enabled: bool = True
    tools: List[str] = field(default_factory=lambda: ["wolfram", "calculator", "python"])

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
    code_execution_enabled: bool = True
    auto_retry_on_fail: bool = True
    max_retries: int = 3

@dataclass
class ReasoningBudgetConfig:
    enabled: bool = True
    max_reasoning_ratio: float = 0.5
    adaptive_depth: bool = True
    simple_task_budget: int = 512
    medium_task_budget: int = 2048
    hard_task_budget: int = 4096
    speculative_reasoning: bool = True
    parallel_subtasks: bool = True

@dataclass
class MultimodalConfig:
    enabled: bool = True
    vision_patch_size: int = 14
    vision_channels: int = 3
    vision_hidden_size: int = 1024
    vision_num_layers: int = 12
    vision_num_heads: int = 16
    vision_mlp_dim: int = 4096
    vision_image_size: int = 224
    audio_n_mels: int = 80
    audio_hidden_size: int = 512
    audio_num_layers: int = 6
    audio_num_heads: int = 8
    audio_mlp_dim: int = 2048
    cross_modal_heads: int = 16

@dataclass
class AdaptiveReasoningV2Config:
    enabled: bool = True
    mcts_simulations: int = 64
    mcts_c_puct: float = 1.414
    num_reasoning_levels: int = 5

@dataclass
class OnlineLearningConfig:
    enabled: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    consolidation_threshold: int = 100
    memory_compression_ratio: float = 0.1

@dataclass
class SafetyConfig:
    enabled: bool = True
    constitutional_rules: list = None
    harm_threshold: float = 0.5

@dataclass
class CompressionConfig:
    enabled: bool = True
    quant_bits: int = 4
    kv_compression_ratio: float = 0.8
    num_sink_tokens: int = 4
    recent_window: int = 256
    target_context_length: int = 4194304
    yarn_beta: float = 0.25
    yarn_alpha: float = 0.1
    target_device: str = "cuda"

@dataclass
class KimiK3PlusConfig:
    model_name: str = "Kimi-K3-Plus-v4"
    version: str = "4.0.0"
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
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    cot: CoTConfig = field(default_factory=CoTConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    reasoning_budget: ReasoningBudgetConfig = field(default_factory=ReasoningBudgetConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    adaptive_reasoning_v2: AdaptiveReasoningV2Config = field(default_factory=AdaptiveReasoningV2Config)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
