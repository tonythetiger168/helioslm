"""Repo-scale long-context probe (v5.74, roadmap AC4).

Task: a synthetic "repository" is a sequence of files. Every file is

  FILE  c filler tokens  DEF name EQ value  c filler tokens

with a distinct ``name`` per file (drawn from a pool larger than the file
count) and a random ``value``. After the last file come ``n_queries``
look-ups ``USE name value`` for names defined somewhere earlier; the model
must produce ``value`` after ``USE name`` (chance 1 / n_values). Lengths
grow only by adding files, so the distance from a query back to its
definition grows with the repository size.

Architectures (``RepoProbeLM``): the same tiny pre-norm stack for every
arm; only the attention module differs:

  dense        causal attention over the whole prefix;
  all_swa      every layer windowed (the structural negative control);
  hybrid       windowed layers with a global layer every
               ``global_every`` layers (B2's sliding / global interleave);
  csa_mean     CompressedSparseAttention (chunks of ``ratio`` mean-pooled,
               top-``topk`` selected) — DeepSeek-V4 order;
  csa_learned  the same with the learned compressor (v5.59).

No position embedding (NoPE) in any arm, so evaluating longer
repositories adds no unseen positions; instead every position also sees
its previous token (a fixed token shift at the input), which lets one
attention layer read "the token after the matching name" (the induction
pattern a definition look-up needs) in every arm alike.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from evastar.src.attention.kv_compress import CompressedSparseAttention

PAD, FILE, DEF, USE, EQ = 0, 1, 2, 3, 4
NAME0, N_NAMES = 16, 32
VAL0, N_VALUES = 48, 16
CODE0, N_CODE = 64, 64
VOCAB = CODE0 + N_CODE
ARMS = ("dense", "all_swa", "hybrid", "csa_mean", "csa_learned")


def make_repo(n_files, n_queries, filler, rng):
    """One sequence. Returns (ids, query_value_positions, answers)."""
    if n_files > N_NAMES:
        raise ValueError(f"n_files must be <= {N_NAMES}, got {n_files}")
    names = rng.sample(range(N_NAMES), n_files)
    values = [rng.randrange(N_VALUES) for _ in names]
    ids = []
    for nm, v in zip(names, values):
        ids.append(FILE)
        ids += [CODE0 + rng.randrange(N_CODE) for _ in range(filler)]
        ids += [DEF, NAME0 + nm, EQ, VAL0 + v]
        ids += [CODE0 + rng.randrange(N_CODE) for _ in range(filler)]
    asked = rng.sample(range(n_files), min(n_queries, n_files))
    pos, ans = [], []
    for i in asked:
        ids += [USE, NAME0 + names[i]]
        pos.append(len(ids))          # the value is predicted at this index
        ids.append(VAL0 + values[i])
        ans.append(VAL0 + values[i])
    return ids, pos, ans


def make_batch(batch, n_files, n_queries, filler, rng):
    seqs = [make_repo(n_files, n_queries, filler, rng) for _ in range(batch)]
    ids = torch.tensor([s[0] for s in seqs])
    pos = torch.tensor([s[1] for s in seqs])
    ans = torch.tensor([s[2] for s in seqs])
    return ids, pos, ans


class WindowedAttention(nn.Module):
    """Causal multi-head attention; ``window`` = None is dense."""

    def __init__(self, dim, n_heads, window=None):
        super().__init__()
        self.h, self.dh, self.window = n_heads, dim // n_heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        i = torch.arange(L, device=x.device)
        mask = i[None, :] <= i[:, None]
        if self.window is not None:
            mask = mask & (i[:, None] - i[None, :] < self.window)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


def _attention(arm, layer, n_layers, dim, heads, window, global_every,
               ratio, topk):
    if arm == "dense":
        return WindowedAttention(dim, heads)
    if arm == "all_swa":
        return WindowedAttention(dim, heads, window)
    if arm == "hybrid":
        is_global = layer % global_every == global_every - 1
        return WindowedAttention(dim, heads, None if is_global else window)
    if arm in ("csa_mean", "csa_learned"):
        return CompressedSparseAttention(
            dim, heads, ratio=ratio, topk=topk,
            compressor="learned" if arm == "csa_learned" else "mean")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


class RepoProbeLM(nn.Module):
    def __init__(self, arm, dim=128, heads=4, n_layers=4, window=16,
                 global_every=2, ratio=4, topk=8):
        super().__init__()
        self.arm = arm
        self.emb = nn.Embedding(VOCAB, dim)
        self.prev_emb = nn.Embedding(VOCAB, dim)      # token shift
        self.attn = nn.ModuleList([
            _attention(arm, i, n_layers, dim, heads, window, global_every,
                       ratio, topk) for i in range(n_layers)])
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                          nn.Linear(4 * dim, dim)) for _ in range(n_layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, VOCAB)

    def forward(self, ids):
        prev = F.pad(ids[:, :-1], (1, 0), value=PAD)
        x = self.emb(ids) + self.prev_emb(prev)
        for attn, mlp, n1, n2 in zip(self.attn, self.mlp, self.n1, self.n2):
            x = x + attn(n1(x))
            x = x + mlp(n2(x))
        return self.head(self.norm(x))


def query_loss_and_acc(model, ids, pos, ans):
    """Cross-entropy and accuracy on the value positions only (the logits
    at index p - 1 predict the token at p)."""
    logits = model(ids)
    lg = logits.gather(1, (pos - 1).unsqueeze(-1).expand(-1, -1, VOCAB))
    loss = F.cross_entropy(lg.reshape(-1, VOCAB), ans.reshape(-1))
    acc = (lg.argmax(-1) == ans).float().mean().item()
    return loss, acc


def max_query_distance(n_files, filler):
    """Length of the file section (all files, queries excluded): an upper
    bound on how far back a query's definition can sit, not counting the
    up to 3 * n_queries query tokens in between."""
    file_len = 2 * filler + 5
    return n_files * file_len


def seed_rng(seed):
    return random.Random(seed)
