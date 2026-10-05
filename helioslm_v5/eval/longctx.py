"""Long-context retrieval eval — passkey-by-loglikelihood harness (Phase 2.2).

Measures passkey retrieval WITHOUT generation: the task prompt embeds a
key token at a controlled position; the model scores N candidate
continuations and picks argmax log-prob. Loglikelihood scoring keeps the
harness CPU-runnable and deterministic (no sampling), matching the repo's
constructed-ground-truth audit philosophy.

Honesty constraints (the point of this module):
  - An UNTRAINED model scores near chance — the harness measures the
    harness on such models. Real capability claims require a trained
    checkpoint; the runner records the model/config identity with every
    result so numbers never float free of their provenance.
  - Beyond ``max_position_embeddings`` the caller must pass an explicit
    ``rope_scaling`` (linear/ntk/yarn); the harness fails loudly instead of
    silently extrapolating rotary phases.
  - 32K–256K lengths are supported by the interface but are the caller's
    compute budget (mid/full preset, GPU); this module never claims a
    context number it did not run.
"""
import random
from typing import Dict, List, Optional

import torch

KEY_MARKER = 3     # task-space token: "the magic word is ..."
QUERY_MARKER = 4   # task-space token: "what is the magic word?"


def make_task(length: int, key_id: int, pos_frac: float, seed: int,
              filler_lo: int = 50, filler_hi: int = 900) -> List[int]:
    """Token-id-space passkey task: [filler ..., KEY, key, filler ...,
    QUERY]. ``key_id`` must live OUTSIDE the filler range so distractor
    candidates can be guaranteed absent."""
    if not (filler_lo <= filler_hi):
        raise ValueError("empty filler range")
    if filler_lo <= key_id <= filler_hi:
        raise ValueError("key_id inside filler range — distractors could "
                         "collide with the filler, breaking the task")
    if not (0.0 <= pos_frac <= 1.0):
        raise ValueError("pos_frac must be in [0, 1]")
    rng = random.Random(seed)
    ids = [rng.randint(filler_lo, filler_hi) for _ in range(length - 3)]
    pos = min(int(length * pos_frac), length - 3)
    ids.insert(pos, KEY_MARKER)
    ids.insert(pos + 1, key_id)
    ids.append(QUERY_MARKER)
    assert len(ids) == length
    return ids


def make_candidates(key_id: int, n: int, filler_lo: int = 50,
                    filler_hi: int = 900, seed: int = 0) -> List[int]:
    """Distractors are drawn from OUTSIDE the filler range, so none can
    appear in the prompt by construction."""
    pool = [i for i in range(10, filler_lo)] + \
           [i for i in range(filler_hi + 1, 1000)]
    rng = random.Random(seed)
    ds = rng.sample(pool, n - 1)
    cands = ds + [key_id]
    rng.shuffle(cands)
    return cands


def candidate_logprob(model, ids: List[int], cand_id: int) -> float:
    """log P(cand_id | ids) under the model's next-token distribution."""
    with torch.no_grad():
        out = model(torch.tensor([ids]))
        logits = out[0] if isinstance(out, tuple) else out
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return float(logp[cand_id])


def run_passkey_eval(model, lengths=(512, 1024, 2048), trials=4,
                     n_candidates=8, pos_fracs=(0.1, 0.5, 0.9),
                     rope_scaling: Optional[dict] = None,
                     model_tag: str = "unknown") -> Dict:
    """Passkey accuracy per length. Records provenance with the numbers."""
    cfg_max = getattr(model, "config", None)
    max_pos = getattr(cfg_max, "max_position_embeddings", None) if \
        cfg_max is not None else None
    results = {}
    for L in lengths:
        if max_pos is not None and L > max_pos and rope_scaling is None:
            raise ValueError(
                f"length {L} exceeds max_position_embeddings {max_pos} "
                f"without rope_scaling — refusing to silently extrapolate "
                f"rotary phases (pass rope_scaling, e.g. linear with "
                f"factor {L / max_pos:.2f})")
        correct = 0
        total = 0
        for t in range(trials):
            for pf in pos_fracs:
                key = 10 + ((t * 7 + int(pf * 100)) % 30)   # 10..39
                ids = make_task(L, key, pf, seed=t * 1000 + int(pf * 10))
                cands = make_candidates(key, n_candidates, seed=t)
                scores = {c: candidate_logprob(model, ids, c)
                          for c in cands}
                pick = max(scores, key=scores.get)
                correct += int(pick == key)
                total += 1
        results[L] = {"accuracy": correct / total, "n": total}
    return {"model": model_tag,
            "max_position_embeddings": max_pos,
            "rope_scaling": rope_scaling,
            "n_candidates": n_candidates,
            "lengths": results}
