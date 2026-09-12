from .base_config import HeliosLMConfig

lite_config = HeliosLMConfig(
    model_name="HeliosLM-Lite-v1",
    hidden_size=4096,
    num_hidden_layers=48,
    max_position_embeddings=524288,
    intermediate_size=12288,
)
lite_config.moe.num_experts = 64
lite_config.moe.num_activated_experts = 4
lite_config.moe.expert_hidden_size = 4096
lite_config.moe.num_shared_experts = 1
lite_config.attention.num_attention_heads = 32
lite_config.attention.num_key_value_heads = 8
lite_config.speculative.enabled = True
