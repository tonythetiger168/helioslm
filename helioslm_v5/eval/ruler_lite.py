"""RULER-lite: synthetic long-context tasks in token-id space (OS7).

Three tasks after RULER (Hsieh et al., 2024), small enough to train a
probe model on, so attention layouts can be compared on the same data at
the same size:

- ``niah_single``: one needle ``[NEEDLE k v]`` in filler; query ``k``.
- ``niah_multikey``: ``n_pairs`` needles with distinct keys; query one.
- ``vartrack``: a chain ``[NEEDLE k1 v]``, ``[NEEDLE k2 ARROW k1]``, ...
  spread in order through the filler, plus one distractor chain; query
  the last variable of the real chain (answer ``v``, ``hops`` lookups).

Every sequence ends ``[QUERY k]`` and the answer is the token predicted
at the last position, always a value token, so accuracy is exact match
on one token (chance 1 / N_VALUES). Needle positions are uniform over the
sequence, so longer evaluation lengths mean longer distances. Data is
fully determined by the torch.Generator passed in.

Vocabulary (``VOCAB`` = 128): 0 pad, 1 NEEDLE, 2 QUERY, 3 ARROW,
keys 4..35, values 36..67, filler 68..127.
"""
from typing import Tuple

import torch

NEEDLE, QUERY, ARROW = 1, 2, 3
KEY0, N_KEYS = 4, 32
VAL0, N_VALUES = 36, 32
FILL0, VOCAB = 68, 128
TASKS = ("niah_single", "niah_multikey", "vartrack")


def _randint(gen, lo, hi):
    return int(torch.randint(lo, hi, (1,), generator=gen))


def _place(length, chunks, gen, ordered=False):
    """Filler of ``length`` tokens with ``chunks`` (token lists) inserted
    at uniform, non-overlapping positions, then ``[QUERY k]`` appended by
    the caller. ``ordered`` keeps the chunks in the given order."""
    body = length - 2
    need = sum(len(c) for c in chunks)
    if need > body:
        raise ValueError(f"length {length} too short for {need} needle tokens")
    fill = torch.randint(FILL0, VOCAB, (body - need,), generator=gen).tolist()
    slots = sorted(torch.randperm(len(fill) + 1, generator=gen)[:len(chunks)]
                   .tolist())
    if not ordered:
        perm = torch.randperm(len(chunks), generator=gen).tolist()
        chunks = [chunks[i] for i in perm]
    out, prev = [], 0
    for s, c in zip(slots, chunks):
        out += fill[prev:s] + c
        prev = s
    return out + fill[prev:]


def sample(task, length, gen, n_pairs=4, hops=3) -> Tuple[list, int]:
    """One sequence (list of ids, ending ``[QUERY k]``) and its answer."""
    keys = (torch.randperm(N_KEYS, generator=gen) + KEY0).tolist()
    vals = (torch.randint(0, N_VALUES, (n_pairs + 2,), generator=gen)
            + VAL0).tolist()
    if task == "niah_single":
        k, v = keys[0], vals[0]
        ids = _place(length, [[NEEDLE, k, v]], gen)
    elif task == "niah_multikey":
        pairs = list(zip(keys[:n_pairs], vals[:n_pairs]))
        q = _randint(gen, 0, n_pairs)
        k, v = pairs[q]
        ids = _place(length, [[NEEDLE, a, b] for a, b in pairs], gen)
    elif task == "vartrack":
        chain = keys[:hops]
        decoy = keys[hops:2 * hops]
        v, dv = vals[0], vals[1]
        real = [[NEEDLE, chain[0], v]] + [[NEEDLE, chain[i], ARROW, chain[i - 1]]
                                          for i in range(1, hops)]
        fake = [[NEEDLE, decoy[0], dv]] + [[NEEDLE, decoy[i], ARROW, decoy[i - 1]]
                                           for i in range(1, hops)]
        # interleave the two chains, each kept in order
        merged, ri, fi = [], 0, 0
        while ri < hops or fi < hops:
            if fi >= hops or (ri < hops and _randint(gen, 0, 2) == 0):
                merged.append(real[ri])
                ri += 1
            else:
                merged.append(fake[fi])
                fi += 1
        k = chain[-1]
        ids = _place(length, merged, gen, ordered=True)
    else:
        raise ValueError(f"unknown task {task!r}; tasks: {TASKS}")
    ids = ids + [QUERY, k]
    assert len(ids) == length
    return ids, v


def make_batch(task, batch, length, gen, **kw):
    """(ids [B, length], answers [B]); the answer is predicted at the last
    position, i.e. from ``logits[:, -1]``."""
    rows, ans = zip(*(sample(task, length, gen, **kw) for _ in range(batch)))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(ans)


@torch.no_grad()
def accuracy(model, task, length, n_batches, batch, seed=0, **kw):
    """Exact-match accuracy of ``model(ids)[0][:, -1].argmax()``."""
    gen = torch.Generator().manual_seed(seed)
    was = model.training
    model.eval()
    hit = tot = 0
    for _ in range(n_batches):
        ids, ans = make_batch(task, batch, length, gen, **kw)
        pred = model(ids.to(next(model.parameters()).device))[0][:, -1].argmax(-1)
        hit += (pred.cpu() == ans).sum().item()
        tot += ans.numel()
    model.train(was)
    return hit / tot
