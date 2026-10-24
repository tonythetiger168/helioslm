"""Thinking-budget control (v5.73, roadmap U2; Nemotron 3 Nano budget
control, gpt-oss reasoning effort).

Two halves:

- Inference: ``ThinkingBudget`` is a per-row state machine used by
  ``EvastarV5.generate(thinking_budget=...)``. It tracks whether a row is
  inside a thinking span (opened by ``think_open_id``, closed by
  ``think_close_id``) and how many thinking tokens it has generated. Once a
  row has spent ``budget`` thinking tokens, the next token is forced to
  the close marker (every other logit is masked to -inf), so a row never
  generates more than ``budget`` tokens between the markers. A span opened
  in the prompt counts (its prompt tokens do not). After a row's span has
  closed, the open marker is masked unless ``allow_reopen``.
- Training: ``truncate_thinking`` cuts the thinking span of a training
  sequence to ``b`` tokens and closes it right there, keeping the answer,
  so the model learns to answer from a partial trace (Nemotron trains on
  such truncated traces; without them a forced close at inference lands
  the model in a state it never saw). ``random_truncation`` draws ``b``
  per example.

``EFFORT_BUDGETS`` maps gpt-oss-style reasoning effort names to fractions
of a caller-chosen maximum budget.
"""
import random

import torch

EFFORT_BUDGETS = {"low": 0.25, "medium": 0.5, "high": 1.0}


def effort_to_budget(effort, max_budget):
    """``"low" | "medium" | "high"`` -> an int budget out of ``max_budget``."""
    if effort not in EFFORT_BUDGETS:
        raise ValueError(f"effort must be one of {sorted(EFFORT_BUDGETS)}, got {effort!r}")
    if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget < 0:
        raise ValueError(f"max_budget must be an int >= 0, got {max_budget!r}")
    return int(round(EFFORT_BUDGETS[effort] * max_budget))


class ThinkingBudget:
    """Force the thinking span closed after ``budget`` tokens (per row)."""

    def __init__(self, think_open_id, think_close_id, budget, allow_reopen=False):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(f"budget must be an int >= 0, got {budget!r}")
        if think_open_id == think_close_id:
            raise ValueError("think_open_id and think_close_id must differ")
        self.open_id = int(think_open_id)
        self.close_id = int(think_close_id)
        self.budget = budget
        self.allow_reopen = bool(allow_reopen)
        self.inside = self.used = self.closed = None

    def start(self, input_ids):
        """Initialise from the prompt: a row is inside a span when its last
        open marker comes after its last close marker."""
        B, L = input_ids.shape
        pos = torch.arange(1, L + 1, device=input_ids.device).expand(B, L)
        last_open = torch.where(input_ids == self.open_id, pos, 0).max(1).values
        last_close = torch.where(input_ids == self.close_id, pos, 0).max(1).values
        self.inside = last_open > last_close
        self.closed = last_close > 0
        self.used = torch.zeros(B, dtype=torch.long, device=input_ids.device)
        return self

    def process(self, logits):
        """Mask ``logits`` [B, V] in place of a copy; call before sampling."""
        logits = logits.clone()
        force = self.inside & (self.used >= self.budget)
        if force.any():
            close_logit = torch.zeros_like(logits[force][:, 0])
            logits[force] = float("-inf")
            logits[force, self.close_id] = close_logit
        if not self.allow_reopen:
            block = self.closed & ~self.inside
            if block.any():
                logits[block, self.open_id] = float("-inf")
        # A close marker outside a span is meaningless; never emit one.
        outside = ~self.inside
        if outside.any():
            logits[outside, self.close_id] = float("-inf")
        return logits

    def update(self, next_token, active=None):
        """Advance the state with the sampled tokens [B] or [B, 1]; rows
        with ``active`` False (finished / frozen) are left unchanged."""
        tok = next_token.reshape(-1)
        if active is None:
            active = torch.ones_like(self.inside)
        closing = active & self.inside & (tok == self.close_id)
        thinking = active & self.inside & ~closing
        opening = active & ~self.inside & (tok == self.open_id)
        self.used = self.used + thinking.long()
        self.inside = (self.inside & ~closing) | opening
        self.closed = self.closed | closing
        return self


def thinking_spans(ids, think_open_id, think_close_id):
    """[(start, end)] index pairs of each open..close span in a 1-D id list
    (``end`` is the close marker's index, or None when unclosed)."""
    spans, start = [], None
    for i, t in enumerate(ids):
        if t == think_open_id and start is None:
            start = i
        elif t == think_close_id and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, None))
    return spans


def truncate_thinking(ids, think_open_id, think_close_id, b):
    """Keep the first ``b`` thinking tokens of the first closed span and
    close it there; everything after the original close marker (the answer)
    is kept. Sequences without a closed span are returned unchanged."""
    ids = list(ids)
    spans = [s for s in thinking_spans(ids, think_open_id, think_close_id)
             if s[1] is not None]
    if not spans or b < 0:
        return ids
    start, end = spans[0]
    keep = min(b, end - start - 1)
    return ids[:start + 1 + keep] + ids[end:]


def random_truncation(ids, think_open_id, think_close_id, p, rng=None):
    """With probability ``p``, truncate the first span to a uniformly drawn
    length in [0, full length]; otherwise return ``ids`` unchanged."""
    rng = rng or random
    if rng.random() >= p:
        return list(ids)
    spans = [s for s in thinking_spans(ids, think_open_id, think_close_id)
             if s[1] is not None]
    if not spans:
        return list(ids)
    start, end = spans[0]
    return truncate_thinking(ids, think_open_id, think_close_id,
                             rng.randint(0, end - start - 1))
