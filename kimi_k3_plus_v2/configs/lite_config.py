from .base_config import KimiK3PlusBaseConfig
lite_config = KimiK3PlusBaseConfig(
    model_name="Kimi-K3-Plus-Lite",
    vocab_size=128000,
    max_position_embeddings=524288,
    hidden_size=4096,
    num_hidden_layers=48,
    intermediate_size=16384,
)
lite_config.moe.num_experts = 128
lite_config.moe.num_activated_experts = 4
lite_config.moe.expert_hidden_size = 2048
lite_config.attention.num_attention_heads = 32
