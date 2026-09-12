from .base_config import HeliosLMConfig

nano_config = HeliosLMConfig(
    model_name="HeliosLM-Nano-v1",
    hidden_size=2048,
    num_hidden_layers=32,
    max_position_embeddings=262144,
    intermediate_size=6144,
)
nano_config.moe.num_experts = 32
nano_config.moe.num_activated_experts = 4
nano_config.moe.expert_hidden_size = 2048
nano_config.moe.num_shared_experts = 1
nano_config.attention.num_attention_heads = 32
nano_config.attention.num_key_value_heads = 8
nano_config.speculative.enabled = False
