"""Certified top-delta sparse decode (v5.21).

Reference implementation of FFD-style sparse attention decode ("Faster
Than Flash", ICML 2026, arXiv:2609.00097) with a runtime CERTIFICATE —
the piece the paper leaves empirical.

The paper's Eq. 2 gives a per-token guarantee: with threshold
s >= m̃ - delta, every dropped token has attention weight
<= exp(-(delta - g)) relative to the true max, where g = m̃ - m_true.
This module turns that into an end-to-end, checkable output bound:

    retained includes the argmax block, so total softmax mass Z >= 1.
    dropped_mass <= n_dropped * exp(-(delta - g))
    => dropped_fraction <= r / (1 + r),  r = n_dropped * exp(-(delta - g))
    => |out_sparse - out_dense|_inf <= 2 * v_max * r / (1 + r)      (cert)

The certificate is evaluated EVERY step, and `certified_ok` tells you
whether the run actually deserved it:

    g > 0  (pseudo-max OVERestimated the max) degrades the guarantee to
    delta_true = delta - g. certified mode uses m_true (g == 0); pseudo
    mode reports the measured g so the caller can compensate.

    g < 0  (pseudo-max UNDERestimated) is the safe direction: the
    threshold drops lower, selection is a superset, delta_true > delta.

MLA note: with weight absorption, attention logits are LINEAR in the
latent cache (k_j = W_UK c_j), so the top-delta criterion carries over to
MLA latent caches verbatim — pass precomputed scores via ``scores=``.
The V-side bound (v_j = W_UV c_j) additionally absorbs ||W_UV||_2 into
v_max.

This is the first extension of the HeliosLM gate spectrum: cache/draft/
tier gates are bitwise invariances; sparsity SHOULD change the math, so
the gate becomes a certificate instead.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

__all__ = ["CertReport", "certified_sparse_attention"]


@dataclass
class CertReport:
    kept_blocks: int
    dropped_tokens: int
    n_tokens: int
    delta: float
    g: float                    # m̃ - m_true (pseudo-max gap; 0 if certified)
    delta_true: float           # delta - g (the guarantee that actually holds)
    dropped_mass_bound: float   # r = n_dropped * exp(-delta_true)
    output_bound: float         # 2 * v_max * r / (1 + r)
    actual_error: float         # |out_sparse - out_dense|_inf (measured)
    bound_holds: bool           # actual_error <= output_bound
    certified_ok: bool          # delta_true >= 0 (guarantee not degraded)
    notes: List[str] = field(default_factory=list)


@torch.no_grad()
def certified_sparse_attention(
    q: torch.Tensor,               # [H, D] single query position
    k: torch.Tensor,               # [L, H, D] cache
    v: torch.Tensor,               # [L, H, Dv]
    delta: float = 5.0,
    block_size: int = 16,
    n_sink: int = 1,
    local_window: int = 16,
    certified: bool = True,        # use true max (g == 0) vs pseudo-max
    scores: Optional[torch.Tensor] = None,   # precomputed [H, L] (MLA latent)
) -> tuple:
    """Sparse decode with a per-step certificate. Returns (out, report).

    Retained set: sink blocks + local blocks + the argmax block + every
    block whose max score satisfies s >= m̃ - delta.
    """
    H, D = q.shape
    L = k.shape[0]
    if scores is None:
        s = torch.einsum("hd,lhd->hl", q.float(), k.float()) / math.sqrt(D)
    else:
        s = scores.float()                    # [H, L], MLA-latent path
    m_true = s.max(dim=1).values              # [H]

    if certified:
        m_tilde = m_true.clone()
    else:
        # pseudo-max: sinks + local context only (no global sync)
        n_local = min(local_window, L)
        m_tilde = torch.maximum(
            s[:, :n_sink].amax(dim=1),
            s[:, L - n_local:].amax(dim=1),
        )
    g = float((m_tilde - m_true).max())

    # block max scores
    n_blocks = (L + block_size - 1) // block_size
    pad = n_blocks * block_size - L
    s_pad = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    block_max = s_pad.view(H, n_blocks, block_size).amax(dim=2)   # [H, B]

    # union over heads: keep a block if ANY head wants it
    keep_block = block_max.ge(
        (m_tilde - delta).unsqueeze(1)).any(dim=0)                # [B]
    # always retain: sinks, local window, argmax block
    keep_block[: max(1, (n_sink + block_size - 1) // block_size)] = True
    n_local_blocks = max(1, (local_window + block_size - 1) // block_size)
    keep_block[-n_local_blocks:] = True
    argmax_block = (s.argmax(dim=1) // block_size)
    keep_block[argmax_block] = True

    keep_mask = torch.zeros(L, dtype=torch.bool)
    for b in range(n_blocks):
        if keep_block[b]:
            keep_mask[b * block_size: min((b + 1) * block_size, L)] = True

    dropped = int((~keep_mask).sum())
    delta_true = delta - g
    r = dropped * math.exp(-max(delta_true, 0.0))
    v_max = float(v.abs().max())
    output_bound = 2.0 * v_max * r / (1.0 + r) if r > 0 else 0.0

    out_sparse = _masked_softmax_attn(s, v, keep_mask)

    # dense reference (the certificate is checked against it every step)
    out_dense = _masked_softmax_attn(s, v, torch.ones(L, dtype=torch.bool))
    actual = float((out_sparse - out_dense).abs().max())

    notes: List[str] = []
    if g > 0 and not certified:
        notes.append(f"pseudo-max overestimates by g={g:.4f}; effective "
                     f"delta degraded to {delta_true:.4f}")
    if delta_true < 0:
        notes.append("delta_true < 0: guarantee void; raise delta or use "
                     "certified=True")
    return out_sparse, CertReport(
        kept_blocks=int(keep_block.sum()), dropped_tokens=dropped,
        n_tokens=L, delta=delta, g=round(g, 6),
        delta_true=round(delta_true, 6),
        dropped_mass_bound=round(r, 8),
        output_bound=round(output_bound, 8),
        actual_error=round(actual, 8),
        bound_holds=actual <= output_bound + 1e-12,
        certified_ok=delta_true >= 0,
        notes=notes,
    )


def _masked_softmax_attn(s: torch.Tensor, v: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """softmax over the masked token set, per head. Returns [H, Dv]."""
    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
    w = torch.softmax(s, dim=1)                     # [H, L]
    return torch.einsum("hl,lhd->hd", w, v.float())
