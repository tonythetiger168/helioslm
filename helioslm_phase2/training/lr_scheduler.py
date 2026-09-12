"""Learning rate schedulers for long pre-training runs."""
import math
from typing import Optional
import torch
from torch.optim import Optimizer


class CosineDecayScheduler:
    """Cosine decay with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1

        for i, base_lr in enumerate(self.base_lrs):
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(1.0, progress)
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

            self.optimizer.param_groups[i]["lr"] = lr

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


class WarmupStableDecayScheduler:
    """
    Warmup → Stable → Decay scheduler.

    Used by many large models (e.g., LLaMA, GPT-4):
      - Warmup: linear increase
      - Stable: constant LR for most of training
      - Decay: cosine decay at the end
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0
        self.total_steps = warmup_steps + stable_steps + decay_steps

    def step(self):
        self.current_step += 1

        for i, base_lr in enumerate(self.base_lrs):
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step / self.warmup_steps)
            elif self.current_step < self.warmup_steps + self.stable_steps:
                lr = base_lr
            else:
                progress = (self.current_step - self.warmup_steps - self.stable_steps) / self.decay_steps
                progress = min(1.0, progress)
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

            self.optimizer.param_groups[i]["lr"] = lr

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
