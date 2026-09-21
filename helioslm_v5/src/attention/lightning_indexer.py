"""Learned Lightning Indexer for DSA-style sparse top-k attention.

Limitations round (2026-09-21): replaces the v5.8 "free" indexer (head-mean
of the true score terms) with a dedicated LEARNED scorer, in the spirit of
DeepSeek-V3.2's lightning indexer: a small number of index heads score every
cached token against the current query in a low-dimensional space, and the
top-``sparse_top_k`` scorers are the only tokens the (expensive) main
attention runs over.

Score for query token t and cached token j:

    score(t, j) = sum_h  w_h * relu( q_h(t) . k_h(j) )

with per-head learned scalar weights ``w_h`` (the head-weights are the
documented simplification of DeepSeek's per-token head-weight projection —
one scalar per head instead of a hidden->heads projection of the query
token).

Key-design constraint: the MLA cache contract (see mla.py module docstring)
is a 2-tuple ``(c_kv, k_rope)`` with dim 2 = sequence length, relied upon by
MTP rollback and the paged engine. This indexer therefore derives its key
side from the ALREADY-CACHED latent ``c_kv`` via a learned projection
instead of caching a third tensor:

    q_h(t) = W_q @ hidden_t          (current tokens only)
    k_h(j) = W_k @ c_kv[j]           (from the latent cache, on the fly)

Documented deviations from DeepSeek-V3.2's indexer:

1. Keys come from the compressed latent ``c_kv``, not from the token hidden
   states — the cache layout is untouched, so MTP clone/slice rollback,
   engine watermarking, and fp8 cache storage keep working unmodified.
   When the cache is fp8-stored the indexer sees exactly the same quantized
   latents the attention path sees (no side channel).
2. Per-head scalar weights instead of per-token head weights (above).
3. Index keys are RE-DERIVED from the latent cache at every decode step
   (an O(L * d_c * H_idx * d_idx) projection per step). A production
   implementation would cache index keys alongside the latent cache; that
   requires extending the cache contract and is deliberately out of scope.
   This is a compute-side simplification only — selection semantics are
   unaffected.

The module has no softmax and no causal logic of its own: masking,
force-selection of the current token, and order-preserving gathering stay in
MLA where the v5.8 head-mean indexer had them.
"""
import torch
import torch.nn as nn


class LearnedLightningIndexer(nn.Module):
    """Low-dimensional learned scorer for sparse top-k key selection.

    Args:
        hidden_size: model hidden width (query side input).
        kv_latent_dim: MLA latent width (key side input, from the cache).
        num_heads: number of index heads (H_idx). Few and small by design —
            the whole point of a lightning indexer is that scoring all L
            cached tokens stays cheap relative to main attention.
        head_dim: per-head index width (d_idx).
    """

    def __init__(self, hidden_size, kv_latent_dim, num_heads, head_dim):
        super().__init__()
        for name, value in (("num_heads", num_heads), ("head_dim", head_dim)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"indexer {name} must be a positive integer, got "
                    f"{value!r}"
                )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_latent_dim, num_heads * head_dim,
                                bias=False)
        # Per-head learned weights, init 1.0 (uniform head mixing at init,
        # i.e. the sum-of-ReLUs scorer; training can re-weight or switch
        # heads off via w_h -> 0).
        self.head_weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, hidden_states, c_kv):
        """Score current query tokens against cached latents.

        Args:
            hidden_states: [B, seq, hidden_size] — the CURRENT tokens only
                (the same tensor MLA projects; never cached tokens).
            c_kv: [B, 1, kv_len, kv_latent_dim] or [B, kv_len, kv_latent_dim]
                — the latent cache in its COMPUTE dtype (with fp8 storage
                this is the upcast of the stored grid values, exactly what
                the attention score matmul consumes).

        Returns:
            [B, 1, seq, kv_len] index scores (higher = more attendable).
            No masking is applied here; the caller masks with -inf before
            top-k.
        """
        if c_kv.dim() == 4:
            if c_kv.shape[1] != 1:
                raise ValueError(
                    f"indexer expects a shared latent cache [B, 1, L, d_c], "
                    f"got shape {tuple(c_kv.shape)}"
                )
            c_kv = c_kv.squeeze(1)
        if c_kv.dim() != 3:
            raise ValueError(
                f"indexer expects c_kv as [B, 1, L, d_c] or [B, L, d_c], "
                f"got shape {tuple(c_kv.shape)}"
            )
        B, seq, _ = hidden_states.shape
        kv_len = c_kv.shape[1]
        q = self.q_proj(hidden_states).view(B, seq, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        k = self.k_proj(c_kv).view(B, kv_len, self.num_heads,
                                   self.head_dim).transpose(1, 2)
        # [B, H_idx, seq, kv_len]; relu as in the DSA lightning indexer.
        scores = torch.relu(torch.matmul(q, k.transpose(-2, -1)))
        scores = (scores * self.head_weights.view(1, -1, 1, 1)
                  ).sum(dim=1, keepdim=True)
        return scores
