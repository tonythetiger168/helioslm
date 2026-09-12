"""DeepSpeed distributed trainer for 256x H100 pre-training."""
import os
import time
import json
from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DeepSpeedTrainer:
    """
    Production DeepSpeed trainer supporting:
      - ZeRO-1/2/3 with offload
      - Tensor Parallelism (TP)
      - Pipeline Parallelism (PP)
      - Expert Parallelism (EP) for MoE
      - Gradient accumulation
      - Mixed precision (BF16/FP8)
      - Checkpoint save/resume
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        deepspeed_config_path: str,
        output_dir: str = "checkpoints",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        os.makedirs(output_dir, exist_ok=True)

        # Initialize DeepSpeed
        self._init_deepspeed(deepspeed_config_path)

        self.global_step = 0
        self.epoch = 0

    def _init_deepspeed(self, config_path: str):
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed

            with open(config_path) as f:
                ds_config = json.load(f)

            # Create optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-4),
                weight_decay=self.config.get("weight_decay", 0.1),
                betas=(0.9, 0.95),
            )

            # Create LR scheduler
            from .lr_scheduler import CosineDecayScheduler
            total_steps = self.config.get("max_steps", 100000)
            warmup_steps = self.config.get("warmup_steps", 2000)
            lr_scheduler = CosineDecayScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=0.1,
            )

            # Initialize DeepSpeed engine
            self.engine, self.optimizer, self.lr_scheduler, _ = deepspeed.initialize(
                model=self.model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=ds_config,
            )

            self.deepspeed_available = True

        except ImportError:
            print("⚠️  DeepSpeed not available, falling back to DDP")
            self.deepspeed_available = False
            self.engine = self.model
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-4),
                weight_decay=self.config.get("weight_decay", 0.1),
            )
            self.lr_scheduler = None

    def train_step(self, batch: Dict) -> Dict:
        """Single training step with gradient accumulation."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)

        if self.deepspeed_available:
            # DeepSpeed handles backward, gradient accumulation, and optimizer step
            loss = self.engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            if isinstance(loss, dict):
                main_loss = loss["loss"]
                aux_loss = loss.get("aux_loss", 0.0)
                total_loss = main_loss + aux_loss
            else:
                total_loss = loss

            self.engine.backward(total_loss)
            self.engine.step()
        else:
            # Standard PyTorch path
            self.optimizer.zero_grad()
            outputs = self.engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.engine.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            total_loss = loss

        self.global_step += 1

        return {
            "loss": total_loss.item() if hasattr(total_loss, "item") else float(total_loss),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.global_step,
        }

    def save_checkpoint(self, tag: Optional[str] = None):
        """Save checkpoint."""
        tag = tag or f"step_{self.global_step}"

        if self.deepspeed_available:
            self.engine.save_checkpoint(self.output_dir, tag=tag)
        else:
            path = os.path.join(self.output_dir, f"checkpoint_{tag}.pt")
            torch.save({
                "model_state_dict": self.engine.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
            }, path)

        print(f"💾 Checkpoint saved: {tag}")

    def load_checkpoint(self, tag: Optional[str] = None):
        """Load checkpoint."""
        if self.deepspeed_available:
            _, client_state = self.engine.load_checkpoint(self.output_dir, tag=tag)
            self.global_step = client_state.get("global_step", 0)
            self.epoch = client_state.get("epoch", 0)
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint_")]
            if not checkpoints:
                print("⚠️  No checkpoint found")
                return
            latest = sorted(checkpoints)[-1]
            path = os.path.join(self.output_dir, latest)
            checkpoint = torch.load(path, map_location="cpu")
            self.engine.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.global_step = checkpoint.get("global_step", 0)
            self.epoch = checkpoint.get("epoch", 0)

        print(f"📂 Checkpoint loaded: step {self.global_step}")

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 10):
        """Train for one epoch."""
        self.engine.train()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.engine.local_rank if hasattr(self.engine, "local_rank") else 0) 
                     for k, v in batch.items()}

            metrics = self.train_step(batch)

            if batch_idx % log_interval == 0:
                print(f"Step {metrics['step']} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}")

            # Periodic checkpointing
            if self.global_step % self.config.get("checkpoint_interval", 1000) == 0:
                self.save_checkpoint()

        self.epoch += 1
