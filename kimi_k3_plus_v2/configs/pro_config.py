from .base_config import KimiK3PlusBaseConfig
pro_config = KimiK3PlusBaseConfig(
    model_name="Kimi-K3-Plus-Pro",
    max_position_embeddings=1048576,
    hidden_size=5120,
    num_hidden_layers=64,
    intermediate_size=20480,
)
pro_config.moe.num_experts = 256
pro_config.moe.num_activated_experts = 8
pro_config.moe.expert_hidden_size = 2560
pro_config.attention.num_attention_heads = 64
