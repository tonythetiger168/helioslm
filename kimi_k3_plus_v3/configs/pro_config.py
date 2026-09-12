from .base_config import KimiK3PlusConfig
pro_config = KimiK3PlusConfig(
    model_name="Kimi-K3-Plus-Pro-v3",
    hidden_size=5120, num_hidden_layers=64,
    max_position_embeddings=1048576,
)
pro_config.moe.num_experts = 256
pro_config.moe.num_activated_experts = 8
