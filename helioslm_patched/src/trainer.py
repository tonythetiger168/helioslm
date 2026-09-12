"""Trainer - P0+P1+P2+P3+P4 (Production-Ready)"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class PretrainingTrainer:
    """Pretraining trainer with gradient accumulation and mixed precision."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.5e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )
        self.scaler = GradScaler()
        self.grad_accum_steps = 4

    def train(self, dataloader, num_steps):
        self.model.train()
        for step in range(num_steps):
            batch = next(iter(dataloader))
            ctx = self._get_ctx(step, num_steps)

            input_ids = batch["input_ids"][:, :ctx].to(self.device)
            labels = batch["labels"][:, :ctx].to(self.device)

            with autocast(dtype=torch.bfloat16):
                logits, aux_loss, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                ) + 0.01 * aux_loss

            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Step {step}/{num_steps} Loss:{loss.item():.4f} Ctx:{ctx}")

    def _get_ctx(self, step, total):
        """Progressive context length training."""
        if step < total * 0.3:
            return 32768
        elif step < total * 0.6:
            return 131072
        return 1048576


class SFTTrainer:
    """Supervised fine-tuning trainer."""

    def __init__(self, model, config, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def train(self, dataloader, num_epochs=3):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"SFT Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, aux_loss, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
