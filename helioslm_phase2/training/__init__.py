"""HeliosLM Phase 2 — Training Infrastructure

Distributed pre-training with DeepSpeed ZeRO-3 + Megatron-LM integration.
"""

from .deepspeed_trainer import DeepSpeedTrainer
from .checkpoint_manager import CheckpointManager
from .lr_scheduler import CosineDecayScheduler, WarmupStableDecayScheduler
from .long_context import LongContextTrainer, YaRNConfig
from .utils import get_model_size, estimate_memory, setup_distributed

__all__ = [
    "DeepSpeedTrainer",
    "CheckpointManager",
    "CosineDecayScheduler",
    "WarmupStableDecayScheduler",
    "LongContextTrainer",
    "YaRNConfig",
    "get_model_size",
    "estimate_memory",
    "setup_distributed",
]
