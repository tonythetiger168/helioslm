"""Direct Preference Optimization (DPO) trainer.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
           (Rafailov et al., 2023)

DPO avoids training a separate reward model by directly optimizing the policy
using preference data (prompt, chosen, rejected).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import DataLoader


class DPOTrainer:
    """
    DPO trainer for alignment without explicit reward modeling.

    Loss: -log sigmoid(beta * (log pi(chosen)/pi_ref(chosen) - log pi(rejected)/pi_ref(rejected)))
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.global_step = 0

    def _compute_log_probs(self, model: nn.Module, input_ids: torch.Tensor, 
                           attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of labels under the model."""
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather log probs for actual tokens
            token_log_probs = torch.gather(
                log_probs, dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask and average
            masked_log_probs = token_log_probs * shift_mask
            seq_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)

            return seq_log_probs

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute DPO loss."""
        # Chosen
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch.get("chosen_attention_mask", torch.ones_like(chosen_input_ids)).to(self.device)
        chosen_labels = batch.get("chosen_labels", chosen_input_ids).to(self.device)

        # Rejected
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch.get("rejected_attention_mask", torch.ones_like(rejected_input_ids)).to(self.device)
        rejected_labels = batch.get("rejected_labels", rejected_input_ids).to(self.device)

        # Policy log probs
        policy_chosen_logps = self._compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = self._compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        # Reference log probs
        with torch.no_grad():
            ref_chosen_logps = self._compute_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self._compute_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )

        # DPO loss
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps

        logits = self.beta * (policy_ratio - ref_ratio)

        # Label smoothing
        if self.label_smoothing > 0:
            loss = -F.logsigmoid(logits) * (1 - self.label_smoothing) - F.logsigmoid(-logits) * self.label_smoothing
        else:
            loss = -F.logsigmoid(logits)

        return loss.mean()

    def train_step(self, batch: Dict) -> Dict:
        """Single DPO training step."""
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
                print(f"DPO Step {metrics['step']} | Loss: {avg_loss:.4f} | LR: {metrics['lr']:.2e}")

        return total_loss / len(dataloader)
