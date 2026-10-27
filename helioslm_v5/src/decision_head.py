"""HeliosLM v5.41 — LM-backed DecisionHead (local decision engine direction).

Post-merge new direction (2026-10-06 handoff): HeliosLM becomes a locally
deployable, verifiable decision engine. This head is the LM-facing piece:
it consumes the model's OWN hidden states and answers three typed decision
questions in one parallel pass — no token-by-token generation.

Three outputs (the Jev-typed-decision convention already used by the
agent-line head):
  - ``noul``:  sigmoid -> P(yes), a CONTINUOUS probability (v5.36
    convention: uncertainty is a value near 0.5, not a class).
  - ``choice``: softmax over ``num_choices`` routing options.
  - ``score``: unbounded linear output for continuous scores.

Deviation from the handoff sketch, recorded honestly: the sketch named the
attribute ``noul`` and used ``num_choices=255``; we keep both, but the
forward pass works on ARBITRARY leading batch dims ([H], [B, H], [B, T, H])
and loud-errors on a hidden-size mismatch instead of silently broadcasting.

Relationship to ``helioslm_v5/agent/decision_head.py``: that one (v5.32) is
self-contained at toy scale — it owns a char-embedding encoder and a
per-question head registry, and it trains against outcome-verified records.
This module is the thin LM-side readout: the encoder is the LM itself, the
question set is the fixed three-way split, and training losses live with
the caller (CE/BCE/MSE + the RLCD Brier term can be composed outside).
Both are kept deliberately; which one answers depends on where the hidden
state comes from.
"""
import torch
import torch.nn as nn


class DecisionHead(nn.Module):
    """Three-way typed decision readout over LM hidden states.

    Args:
        hidden_size: last dim of the hidden states this head reads.
        num_choices: size of the ``choice`` softmax. Must be >= 2 —
            a 1-way softmax is a constant 1.0 and carries no information,
            so it fails loudly instead of pretending to route.

    Forward:
        hidden: [..., hidden_size] -> {
            "noul":   [...,]            in (0, 1)      — P(yes)
            "choice": [..., num_choices] simplex       — routing dist
            "score":  [...,]            unbounded      — continuous
        }
    """

    def __init__(self, hidden_size: int, num_choices: int = 255):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"DecisionHead: hidden_size must be a positive int, "
                f"got {hidden_size!r}")
        if not isinstance(num_choices, int) or num_choices < 2:
            raise ValueError(
                f"DecisionHead: num_choices must be an int >= 2 (a "
                f"{num_choices}-way softmax carries no signal), got "
                f"{num_choices!r}")
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        self.noul = nn.Linear(hidden_size, 1)
        self.choice = nn.Linear(hidden_size, num_choices)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> dict:
        if hidden.dim() < 1 or hidden.size(-1) != self.hidden_size:
            raise ValueError(
                f"DecisionHead: expected hidden [..., "
                f"{self.hidden_size}], got shape "
                f"{tuple(hidden.shape)} — last dim mismatch is a "
                f"wiring bug, failing loudly")
        return {
            "noul": torch.sigmoid(self.noul(hidden)).squeeze(-1),
            "choice": torch.softmax(self.choice(hidden).float(), dim=-1),
            "score": self.score(hidden).squeeze(-1),
        }
