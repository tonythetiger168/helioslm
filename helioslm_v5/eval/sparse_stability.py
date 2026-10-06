"""Learned-sparse top-k self-consistency probe (v5.43).

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
  - Agreement is computed on the LAST query position only (the decode
    step), batch 0. Prefill stability across all positions is a
    deliberate out-of-scope extension.
"""
import torch
import torch.nn as nn


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
    jaccards = []
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            inter = len(all_sets[i] & all_sets[j])
            union = len(all_sets[i] | all_sets[j])
            jaccards.append(inter / union if union else 1.0)
    return {
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "min_jaccard": min(jaccards),
        "top_k": top_k,
        "n_trials": n_trials,
        "noise_std": noise_std,
        "reference_set": sorted(reference),
        "trial_sets": [sorted(s) for s in trial_sets],
    }
