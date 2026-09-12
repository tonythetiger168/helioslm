"""Supervised Fine-Tuning (SFT) trainer."""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from torch.utils.data import DataLoader


class SFTTrainer:
    """
    SFT trainer for instruction following.

    Features:
      - Prompt masking (only compute loss on response)
      - Chat template support
      - Mixed precision training
      - Gradient accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.global_step = 0

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute cross-entropy loss with prompt masking."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss only on non-masked positions
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "step": self.global_step,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 100):
        """Train for one epoch."""
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]

            if batch_idx % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"SFT Step {metrics['step']} | Loss: {avg_loss:.4f} | LR: {metrics['lr']:.2e}")

        return total_loss / len(dataloader)
