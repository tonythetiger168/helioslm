"""Learned-sparse top-k self-consistency probe (v5.43; v5.44 adds the
all-positions ``prefill_stability`` companion).

A learned lightning indexer (DSA-style sparse attention) makes a hard
discrete choice — which ``top_k`` cached tokens the expensive main
attention runs over. A silently UNSTABLE selection is an anti-shortcut
failure mode: the policy gradient or a downstream audit can be riding on
top-k sets that reshuffle under arbitrary perturbation. This probe
quantifies that stability directly:

  - run the indexer on the query with zero noise -> the reference top-k set
  - run it ``n_trials`` more times with Gaussian noise (std ``noise_std``)
    injected on the QUERY side -> perturbed top-k sets
  - report mean pairwise Jaccard agreement across all runs

Honest scope notes:
  - Noise is applied to the query-side hidden states only. The latent
    cache side (``c_kv``) is left untouched — in decode the cache is
    shared and frozen, so query-side jitter is the perturbation that
    matters at inference time. A training-time probe that also perturbs
    cached latents would need the caller's own cache copy.
  - The probe measures MECHANICAL stability of the selection function,
    not selection QUALITY. A stable-but-wrong indexer scores 1.0 here;
    quality is the indexer training loss's job.
  - ``topk_self_consistency`` reports agreement on the LAST query
    position only (the decode step), batch 0. Use
    ``prefill_stability`` for the all-positions prefill view: it
    scores every query position separately and averages the per-
    position Jaccard agreements.
"""
import torch
import torch.nn as nn


def _validate_probe_inputs(indexer, hidden_states, top_k, n_trials,
                           noise_std):
    """Shared loud-input guards for the stability probes."""
    if not isinstance(indexer, nn.Module):
        raise ValueError("indexer must be an nn.Module with the "
                         "forward(hidden_states, c_kv) contract")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2 (reference + at least "
                         f"one perturbation), got {n_trials}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}")
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, seq, hidden], got "
                         f"shape {tuple(hidden_states.shape)}")


def _pairwise_jaccards(sets: list) -> list:
    jaccards = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            jaccards.append(inter / union if union else 1.0)
    return jaccards


def _mean_pairwise_jaccard(sets: list) -> float:
    jaccards = _pairwise_jaccards(sets)
    return sum(jaccards) / len(jaccards)


def topk_self_consistency(indexer: nn.Module,
                          hidden_states: torch.Tensor,
                          c_kv: torch.Tensor,
                          top_k: int,
                          n_trials: int = 8,
                          noise_std: float = 0.05,
                          seed: int = 0) -> dict:
    """Mean pairwise Jaccard agreement of the top-k selected token sets
    under repeated query-side Gaussian perturbation.

    Args:
        indexer: a LearnedLightningIndexer (any module with the
            ``forward(hidden_states, c_kv) -> [B, 1, seq, kv_len]``
            contract).
        hidden_states: [B, seq, hidden] current-token queries.
        c_kv: latent cache, [B, 1, kv_len, d_c] or [B, kv_len, d_c].
        top_k: how many cached tokens the selection keeps.
        n_trials: number of noisy repetitions (>= 2; the zero-noise
            reference run always happens first).
        noise_std: Gaussian std on the query side. 0.0 gives the
            degenerate perfectly-consistent probe (a useful oracle).
        seed: seeds the noise generator -> fully deterministic.

    Returns:
        {"mean_jaccard", "min_jaccard", "top_k", "n_trials",
         "noise_std", "reference_set", "trial_sets"} with sets as sorted
        lists of ints.

    Loud errors: non-module indexer, non-positive top_k, top_k larger
    than the cache, fewer than 2 trials, negative noise_std, or an
    unexpected indexer output shape.
    """
    _validate_probe_inputs(indexer, hidden_states, top_k, n_trials,
                           noise_std)

    with torch.no_grad():
        ref_out = indexer(hidden_states, c_kv)
        if ref_out.dim() != 4 or ref_out.shape[2] != hidden_states.shape[1]:
            raise ValueError(
                f"indexer must return [B, 1, seq, kv_len] with seq == "
                f"{hidden_states.shape[1]}, got {tuple(ref_out.shape)}")
        ref_scores = ref_out[0, 0, -1]  # last query position, batch 0
        kv_len = ref_scores.shape[-1]
        if top_k > kv_len:
            raise ValueError(
                f"top_k ({top_k}) exceeds the cache length ({kv_len})")

        reference = set(torch.topk(ref_scores, top_k).indices.tolist())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        trial_sets = []
        for _ in range(n_trials):
            noisy = hidden_states + torch.randn(
                hidden_states.shape, generator=generator) * noise_std
            scores = indexer(noisy, c_kv)[0, 0, -1]
            trial_sets.append(
                set(torch.topk(scores, top_k).indices.tolist()))

    all_sets = [reference] + trial_sets
    jaccards = _pairwise_jaccards(all_sets)
    return {
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "min_jaccard": min(jaccards),
        "top_k": top_k,
        "n_trials": n_trials,
        "noise_std": noise_std,
        "reference_set": sorted(reference),
        "trial_sets": [sorted(s) for s in trial_sets],
    }


def prefill_stability(indexer: nn.Module,
                      hidden_states: torch.Tensor,
                      c_kv: torch.Tensor,
                      top_k: int,
                      n_trials: int = 8,
                      noise_std: float = 0.05,
                      seed: int = 0) -> dict:
    """Mean pairwise Jaccard agreement of the top-k selected token sets,
    scored independently at EVERY query position (batch 0).

    Where ``topk_self_consistency`` audits the decode step (last query
    position), this probe audits the prefill pass: for each query
    position p it builds a reference top-k set over the cache and
    ``n_trials`` perturbed sets (Gaussian noise, std ``noise_std``, on
    the query-side hidden states), then scores per-position agreement.
    The overall score is the mean of the per-position means — every
    position weights equally regardless of how often it is attended.

    Simplification vs. the real prefill pass: the cache ``c_kv`` is
    shared across positions and never perturbed, and positions are
    scored against the SAME cache rather than the causal prefix each
    position actually attends. For a DSA-style indexer whose selection
    is a pure function of (query, cache) this matches the deployed
    call; probes that need causal-prefix caches should slice the cache
    per position before calling.

    Args:
        indexer: module with ``forward(hidden_states, c_kv) ->
            [B, 1, seq, kv_len]``.
        hidden_states: [B, seq, hidden] current-token queries.
        c_kv: latent cache, [B, 1, kv_len, d_c] or [B, kv_len, d_c].
        top_k: how many cached tokens each position keeps.
        n_trials: number of noisy repetitions (>= 2).
        noise_std: Gaussian std on the query side. 0.0 gives a
            perfectly-consistent probe (useful oracle).
        seed: seeds the noise generator -> fully deterministic.

    Returns:
        {"mean_jaccard", "per_position", "top_k", "n_trials",
         "noise_std"} where "per_position" is a list of floats, one per
         query position (same length as the query sequence), and
         "mean_jaccard" is their mean.

    Loud errors: same guards as ``topk_self_consistency`` (non-module
    indexer, non-positive top_k, top_k larger than the cache, fewer
    than 2 trials, negative noise_std, unexpected output shape).
    """
    _validate_probe_inputs(indexer, hidden_states, top_k, n_trials,
                           noise_std)

    with torch.no_grad():
        ref_out = indexer(hidden_states, c_kv)
        if ref_out.dim() != 4 or ref_out.shape[2] != hidden_states.shape[1]:
            raise ValueError(
                f"indexer must return [B, 1, seq, kv_len] with seq == "
                f"{hidden_states.shape[1]}, got {tuple(ref_out.shape)}")
        ref_scores = ref_out[0, 0]  # [seq, kv_len], batch 0
        kv_len = ref_scores.shape[-1]
        if top_k > kv_len:
            raise ValueError(
                f"top_k ({top_k}) exceeds the cache length ({kv_len})")

        reference = [set(torch.topk(ref_scores[p], top_k).indices.tolist())
                     for p in range(ref_scores.shape[0])]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        per_position = [[set(s) for s in reference] for _ in range(n_trials)]
        for t in range(n_trials):
            noisy = hidden_states + torch.randn(
                hidden_states.shape, generator=generator) * noise_std
            scores = indexer(noisy, c_kv)[0, 0]  # [seq, kv_len]
            for p in range(scores.shape[0]):
                per_position[t][p] = set(
                    torch.topk(scores[p], top_k).indices.tolist())

    pos_means = []
    for p in range(ref_scores.shape[0]):
        sets_at_p = [reference[p]] + [per_position[t][p]
                                      for t in range(n_trials)]
        pos_means.append(_mean_pairwise_jaccard(sets_at_p))
    return {
        "mean_jaccard": sum(pos_means) / len(pos_means),
        "per_position": pos_means,
        "top_k": top_k,
        "n_trials": n_trials,
        "noise_std": noise_std,
    }
