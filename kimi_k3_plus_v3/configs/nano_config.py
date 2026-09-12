from .base_config import KimiK3PlusConfig
nano_config = KimiK3PlusConfig(
    model_name="Kimi-K3-Plus-Nano-v3",
    hidden_size=3072, num_hidden_layers=32,
    max_position_embeddings=131072,
)
nano_config.moe.num_experts = 1
nano_config.moe.num_activated_experts = 1
nano_config.speculative.enabled = False
nano_config.rag.enabled = False
