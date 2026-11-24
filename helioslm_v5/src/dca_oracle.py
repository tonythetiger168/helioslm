"""Experimental cross-layer attention baseline, not a faithful DCA reproduction."""
import math
import torch
from torch import nn


class CrossLayerAttention(nn.Module):
    """Attend across depth at each token, never across sequence positions."""

    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, current, history):
        if not history or any(h.shape != current.shape for h in history):
            raise ValueError("nonempty history with matching shapes required")
        layers = torch.stack(tuple(history), dim=-2)
        q = self.query(current).unsqueeze(-2)
        k = self.key(layers)
        v = self.value(layers)
        scores = (q * k).sum(-1) / math.sqrt(current.shape[-1])
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * v).sum(-2)
