from .base_config import KimiK3PlusConfig
lite_config = KimiK3PlusConfig(
    model_name="Kimi-K3-Plus-Lite-v3",
    hidden_size=4096, num_hidden_layers=48,
    max_position_embeddings=524288,
)
lite_config.moe.num_experts = 128
lite_config.moe.num_activated_experts = 4
