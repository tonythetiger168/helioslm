"""Cross-depth attention feasibility reference, not a reproduction of Google DCA."""
import math
import torch
from torch import nn


class CrossDepthAttention(nn.Module):
    """Token-local attention over earlier layer states, with no temporal mixing."""

    def __init__(self, width: int):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.scale = math.sqrt(width)

    def forward(self, current, history):
        if current.ndim != 3 or not history:
            raise ValueError("expected [batch, time, width] and nonempty history")
        if any(x.shape != current.shape for x in history):
            raise ValueError("history shape mismatch")
        states = torch.stack(tuple(history), dim=-2)
        q = self.query(current).unsqueeze(-2)
        scores = (q * self.key(states)).sum(-1) / self.scale
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * self.value(states)).sum(-2)
