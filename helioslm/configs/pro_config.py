from .base_config import HeliosLMConfig

pro_config = HeliosLMConfig(
    model_name="HeliosLM-Pro-v1",
    hidden_size=8192,
    num_hidden_layers=64,
    max_position_embeddings=1048576,
    intermediate_size=24576,
)
pro_config.moe.num_experts = 128
pro_config.moe.num_activated_experts = 6
pro_config.moe.expert_hidden_size = 6144
pro_config.moe.num_shared_experts = 2
pro_config.attention.num_attention_heads = 64
pro_config.attention.num_key_value_heads = 8
