from .base_config import KimiK3PlusBaseConfig
nano_config = KimiK3PlusBaseConfig(
    model_name="Kimi-K3-Plus-Nano",
    vocab_size=128000,
    max_position_embeddings=131072,
    hidden_size=3072,
    num_hidden_layers=32,
    intermediate_size=12288,
)
nano_config.moe.num_experts = 1
nano_config.moe.num_activated_experts = 1
nano_config.attention.num_attention_heads = 24
nano_config.speculative.enabled = False
