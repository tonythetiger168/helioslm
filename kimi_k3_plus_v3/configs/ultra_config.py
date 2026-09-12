from .base_config import KimiK3PlusConfig
ultra_config = KimiK3PlusConfig(
    model_name="Kimi-K3-Plus-Ultra-v3",
    hidden_size=7168, num_hidden_layers=93,
    max_position_embeddings=2097152,
)
ultra_config.moe.num_experts = 896
ultra_config.moe.num_activated_experts = 16
