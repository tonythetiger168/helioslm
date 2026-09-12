"""
四阶段训练器 + 知识蒸馏 (Nano)
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class PretrainingTrainer:
    """Phase 1: 预训练"""
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.5e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        self.scaler = GradScaler()

    def train(self, dataloader, num_steps):
        self.model.train()
        for step in range(num_steps):
            batch = next(iter(dataloader))
            ctx = self._get_context_length(step, num_steps)
            input_ids = batch['input_ids'][:, :ctx].to(self.device)
            labels = batch['labels'][:, :ctx].to(self.device)

            with autocast(dtype=torch.bfloat16):
                logits, aux_loss = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss + 0.01 * aux_loss

            self.scaler.scale(loss).backward()
            if (step + 1) % 4 == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Step {step}/{num_steps} | Loss: {loss.item():.4f} | Ctx: {ctx}")

    def _get_context_length(self, step, total):
        if step < total * 0.3: return 32768
        elif step < total * 0.6: return 131072
        return 1048576


class DistillationTrainer:
    """Nano 知识蒸馏训练器"""
    def __init__(self, teacher, student, config, device, alpha=0.5):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.device = device
        self.alpha = alpha  # 蒸馏损失权重
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)
        self.teacher.eval()

    def train(self, dataloader, num_epochs=3):
        self.student.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Distill Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.no_grad():
                    teacher_logits, _ = self.teacher(input_ids)

                student_logits, _ = self.student(input_ids)

                # 硬标签损失
                ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)

                # KL 蒸馏损失
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits / 2.0, dim=-1),
                    F.softmax(teacher_logits / 2.0, dim=-1),
                    reduction="batchmean"
                ) * (2.0 ** 2)

                loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")


class SFTTrainer:
    """Phase 3: 监督微调"""
    def __init__(self, model, config, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def train(self, dataloader, num_epochs=3):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"SFT Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, _ = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")
