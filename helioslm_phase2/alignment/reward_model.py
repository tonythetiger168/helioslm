"""Reward model for RLHF (optional, if not using DPO)."""
import torch
import torch.nn as nn
from typing import Dict


class RewardModel(nn.Module):
    """
    Bradley-Terry reward model for preference learning.

    Architecture: Shared transformer backbone + scalar reward head.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """Return reward score for input sequence."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["logits"]  # Use last hidden or dedicated output

        # Use last token representation
        last_hidden = hidden_states[:, -1, :]
        reward = self.reward_head(last_hidden)

        return {"reward": reward.squeeze(-1)}
