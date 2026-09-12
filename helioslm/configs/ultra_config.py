from .base_config import HeliosLMConfig

ultra_config = HeliosLMConfig(
    model_name="HeliosLM-Ultra-v1",
    hidden_size=8192,
    num_hidden_layers=80,
    max_position_embeddings=1048576,
    intermediate_size=16384,
)
ultra_config.moe.num_experts = 512
ultra_config.moe.num_activated_experts = 4
ultra_config.moe.expert_hidden_size = 1024
ultra_config.moe.num_shared_experts = 4
ultra_config.moe.min_experts = 2
ultra_config.moe.max_experts = 8
ultra_config.attention.num_attention_heads = 64
ultra_config.attention.num_key_value_heads = 8
