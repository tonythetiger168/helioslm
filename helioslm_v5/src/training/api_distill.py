"""Distillation from an API teacher (v5.82).

A large teacher is usually only reachable through an API, which returns
one of two things, and never the teacher's full distribution in the
student's own tokens:

  text      sampled responses only -> sequence-level KD: fine-tune on the
            teacher's answers, optionally keeping only answers a verifier
            accepts (rejection sampling; `collect_sequences`);
  top-k     the k most likely next tokens with log-probs, in the TEACHER's
            tokenizer (OpenAI-style ``top_logprobs``) -> sparse logit KD.

Top-k targets must be moved to the student's tokens. Here the student
is character-level and a teacher token is a string of one or more
characters, so the alignment is exact in principle:
``first_char_targets`` sums the probability of every top-k token by its
first character (the teacher's marginal over the next character), and
``next_char_targets`` takes the teacher's distribution over the following
character given the one the student is conditioned on. Only the top k
are known, so both distributions are renormalised over the covered
support; with k equal to the teacher vocabulary they are exact, which
T121 checks.
"""
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

TopK = List[Tuple[str, float]]          # [(teacher token string, logprob)]


def first_char_targets(topk: TopK) -> Dict[str, float]:
    """Next-character distribution implied by the top-k teacher tokens,
    renormalised over the characters they cover."""
    mass = defaultdict(float)
    for tok, lp in topk:
        if tok:
            mass[tok[0]] += math.exp(lp)
    z = sum(mass.values())
    if z <= 0:
        raise ValueError("top-k list has no probability mass")
    return {c: m / z for c, m in mass.items()}


def next_char_targets(topk: TopK, prefix: str) -> Optional[Dict[str, float]]:
    """Distribution of the character after ``prefix`` (a proper prefix of
    the teacher token being emitted), from the top-k tokens that start with
    it; None when no top-k token starts with ``prefix``."""
    mass = defaultdict(float)
    n = len(prefix)
    for tok, lp in topk:
        if len(tok) > n and tok.startswith(prefix):
            mass[tok[n]] += math.exp(lp)
    z = sum(mass.values())
    if z <= 0:
        return None
    return {c: m / z for c, m in mass.items()}


def sparse_kd_loss(logits: torch.Tensor, targets: Sequence[Dict[int, float]]):
    """Mean over positions of -sum_c q(c) log p_student(c). ``logits`` is
    [N, V]; ``targets[i]`` maps student token id -> teacher probability.
    Equals cross-entropy to a one-hot target when q is one-hot."""
    if logits.shape[0] != len(targets):
        raise ValueError("one target distribution per logit row")
    logp = F.log_softmax(logits.float(), dim=-1)
    q = torch.zeros_like(logp)
    for i, t in enumerate(targets):
        for tok, p in t.items():
            q[i, tok] = p
    return -(q * logp).sum(-1).mean()


def collect_sequences(prompts: Sequence, sample: Callable,
                      verify: Optional[Callable] = None):
    """Sequence-level KD data: one teacher sample per prompt; with a
    verifier, rejected samples are dropped. Returns (pairs, stats)."""
    pairs, accepted = [], 0
    for p in prompts:
        r = sample(p)
        ok = True if verify is None else bool(verify(p, r))
        accepted += ok
        if ok:
            pairs.append((p, r))
    n = len(prompts)
    return pairs, {"queries": n, "kept": len(pairs),
                   "accept_rate": round(accepted / n, 4) if n else 0.0}
