from .base_config import KimiK3PlusBaseConfig
ultra_config = KimiK3PlusBaseConfig(
    model_name="Kimi-K3-Plus-Ultra",
    max_position_embeddings=2097152,
    hidden_size=7168,
    num_hidden_layers=93,
    intermediate_size=28672,
)
ultra_config.moe.num_experts = 896
ultra_config.moe.num_activated_experts = 16
