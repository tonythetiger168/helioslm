"""Checkpoint merging (v5.124, LB7 of the 2026-10-23 leaderboard review).

K2 Horizon averages checkpoints and merges specialist models after
training; MiniCPM5 trains 16 RL experts and folds them back into one
model. Merging is free at train time (no GPU hours), so it is the cheap
half of LB7 (the other half, the top-k-union on-policy distillation, is
``mopd.topk_union_reverse_kl``).

All methods take state dicts with identical keys and shapes; integer and
bool tensors (step counters, masks) are copied from the base (or the
first model), never averaged.

  linear            sum_i w_i * theta_i, w renormalised to sum 1: model
                    soups / checkpoint averaging over the WSD decay tail;
  task_arithmetic   base + scale * sum_i w_i * (theta_i - base);
  ties              TIES-Merging (Yadav et al., NeurIPS 2023): trim each
                    task vector to its top ``density`` share of entries
                    by magnitude, elect a sign per entry from the summed
                    trimmed vectors, average only the entries that agree
                    with it, then base + scale * merged.
"""
from typing import Dict, List, Optional, Sequence

import torch

METHODS = ("linear", "task_arithmetic", "ties")

State = Dict[str, torch.Tensor]


def _check(states: Sequence[State], base: Optional[State]):
    if not states:
        raise ValueError("need at least one state dict")
    ref = base if base is not None else states[0]
    for s in states:
        if set(s) != set(ref):
            raise ValueError("state dicts have different keys")
        for k, v in s.items():
            if v.shape != ref[k].shape:
                raise ValueError(f"{k}: shape {tuple(v.shape)} vs "
                                 f"{tuple(ref[k].shape)}")
    return ref


def _weights(weights, n):
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    if len(w) != n:
        raise ValueError(f"{len(w)} weights for {n} models")
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("weights must be >= 0 with a positive sum")
    return w


def _mergeable(t):
    return torch.is_floating_point(t)


def ties_merge(deltas: List[torch.Tensor], weights: Sequence[float],
               density: float) -> torch.Tensor:
    """TIES on one tensor's task vectors (same shape)."""
    trimmed = []
    for d in deltas:
        d = d.float()
        n = d.numel()
        keep = max(1, int(round(density * n)))
        if keep < n:
            thresh = d.abs().flatten().kthvalue(n - keep + 1).values
            d = torch.where(d.abs() >= thresh, d, torch.zeros_like(d))
        trimmed.append(d)
    stack = torch.stack([w * d for w, d in zip(weights, trimmed)])
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign) & (stack != 0)
    num = (stack * agree).sum(0)
    den = torch.stack([w * a for w, a in zip(
        weights, agree.float().unbind(0))]).sum(0)
    return torch.where(den > 0, num / den.clamp(min=1e-12),
                       torch.zeros_like(num))


def merge_states(states: Sequence[State], method: str = "linear",
                 weights: Optional[Sequence[float]] = None,
                 base: Optional[State] = None, scale: float = 1.0,
                 density: float = 0.2) -> State:
    """Merge ``states`` into one state dict (see the module docstring)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method != "linear" and base is None:
        raise ValueError(f"{method} needs a base model")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    ref = _check(states, base)
    w = _weights(weights, len(states))
    out = {}
    for k, r in ref.items():
        if not _mergeable(r):
            out[k] = r.clone()
            continue
        if method == "linear":
            z = sum(w)
            m = sum((wi / z) * s[k].float() for wi, s in zip(w, states))
        elif method == "task_arithmetic":
            m = base[k].float() + scale * sum(
                wi * (s[k].float() - base[k].float())
                for wi, s in zip(w, states))
        else:
            m = base[k].float() + scale * ties_merge(
                [s[k].float() - base[k].float() for s in states], w, density)
        out[k] = m.to(r.dtype)
    return out


def load_state(path) -> State:
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["model"] if isinstance(st, dict) and "model" in st else st
