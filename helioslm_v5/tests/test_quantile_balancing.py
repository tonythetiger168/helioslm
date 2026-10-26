"""T116 — Stable LatentMoE study: Kimi K3 Quantile Balancing, v5.77.

Pins:

1. histogram_quantile matches torch.quantile on the raw samples within one
   bin width;
2. repeated updates on one fixed batch: the first solve halves the
   largest imbalance and eight get within 15% of K / E (the cutoffs move
   with the biases, so the solve is a fixed-point iteration), ending
   better than the v5.5 rule (last 512 margins) while the sign rule is
   still above 40%;
3. the bias is mean-centred, and centring never changes the routing
   (adding a constant to every bias leaves top-k unchanged);
4. two "ranks" whose histograms are summed give exactly the bias of one
   process that saw all tokens, while the v5.5 rule (averaged per-rank
   quantiles) differs when the ranks see different data (by little here:
   ~0.006, above the histogram resolution);
5. eval-mode and no-grad forwards record nothing; K == E skips the
   update; other strategies' state dicts carry no new buffers; config
   validation;
6. the committed A/B artifact: Quantile Balancing is the fastest from a
   skewed start and after a drift, never drifts its bias, and beats the
   v5.5 rule when an update sees more than 512 tokens; SMEBU is lower in
   steady state (momentum averages out per-batch sampling noise).

Run from repo root: python3 helioslm_v5/tests/test_quantile_balancing.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config, MoEConfig  # noqa: E402
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402

_ART = _REPO / "benchmarks" / "quantile_balancing_ab_2026-10-21.json"
E, K = 16, 2


def _moe(strategy="quantile_balancing", k=K, bins=1000, latent=32):
    cfg = HeliosLMv5Config(size="lite", moe=MoEConfig(
        num_experts=E, num_activated_experts=k, device_group_size=1,
        balance_strategy=strategy, quantile_bins=bins, latent_dim=latent))
    torch.manual_seed(0)
    m = DeviceLimitedMoE(cfg).train()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg.hidden_size


def _skewed(hidden, n, seed=1, skew=3.0):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(hidden, generator=g)
    return torch.randn(1, n, hidden, generator=g) + d / d.norm() * skew


def _selection_counts(m, x):
    flat = x.reshape(-1, x.shape[-1])
    if m.down_proj is not None:
        flat = m.down_proj(flat)
    idx, _ = m._route(flat)
    return torch.bincount(idx.flatten(), minlength=m.num_experts)


def test_histogram_quantile():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 5000, generator=g, dtype=torch.float64) * 0.3 + 0.2
    lo, hi, nb = -1.5, 2.0, 700
    idx = ((x - lo) / (hi - lo) * nb).floor().clamp(0, nb - 1).long()
    hist = torch.zeros(4, nb, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones_like(x))
    for q in (0.05, 0.125, 0.5, 0.9):
        got = DeviceLimitedMoE.histogram_quantile(hist, lo, hi, q)
        ref = torch.quantile(x, q, dim=1)
        assert (got - ref).abs().max() <= (hi - lo) / nb, (q, got, ref)
    print("PASS test_histogram_quantile")


def test_repeated_updates_balance_a_fixed_batch():
    x = None
    devs = {}
    target = 4096 * K / E                        # 512 per expert
    for strategy in ("quantile_balancing", "quantile", "heuristic"):
        m, hidden = _moe(strategy)
        x = _skewed(hidden, 4096) if x is None else x
        d = []
        for _ in range(8):
            with torch.no_grad():
                d.append((_selection_counts(m, x) - target).abs().max().item())
            m(x)                                # training forward: record
            m.update_bias()
        with torch.no_grad():
            d.append((_selection_counts(m, x) - target).abs().max().item())
        devs[strategy] = d
    qb = devs["quantile_balancing"]
    assert qb[0] > 0.5 * target
    # the first solve halves the imbalance (the cutoffs move with the
    # biases, so it is a fixed-point iteration, not one exact step) ...
    assert qb[1] < 0.6 * qb[0], qb
    # ... and 8 solves get within 15% while the sign rule is still > 40%
    assert qb[-1] < 0.15 * target, qb
    assert devs["heuristic"][-1] > 0.4 * target, devs["heuristic"]
    # every token vs the last 512 margins: QB ends better than v5.5
    assert qb[-1] < devs["quantile"][-1], devs
    print("PASS test_repeated_updates_balance_a_fixed_batch "
          + json.dumps({k: [int(v) for v in d] for k, d in devs.items()}))


def test_centred_and_ranking_invariant():
    m, hidden = _moe()
    x = _skewed(hidden, 2048)
    m(x)
    m.update_bias()
    assert abs(m.route_bias.mean().item()) < 1e-6
    with torch.no_grad():
        a = _selection_counts(m, x)
        m.route_bias.add_(0.37)
        b = _selection_counts(m, x)
        m.route_bias.sub_(0.37)
    assert torch.equal(a, b)
    print("PASS test_centred_and_ranking_invariant")


def test_summed_histograms_are_global():
    m, hidden = _moe()
    x1 = _skewed(hidden, 2048, seed=1)
    x2 = _skewed(hidden, 2048, seed=7, skew=0.0)  # an unskewed rank
    m(torch.cat([x1, x2], 1))
    whole = m.margin_hist.clone()
    m.margin_hist.zero_()
    m(x1)
    h1 = m.margin_hist.clone()
    m.margin_hist.zero_()
    m(x2)
    h2 = m.margin_hist.clone()
    assert torch.equal(h1 + h2, whole)
    lo, hi = m.hist_range.tolist()
    f = K / E
    q_sum = DeviceLimitedMoE.histogram_quantile(h1 + h2, lo, hi, f)
    q_avg = (DeviceLimitedMoE.histogram_quantile(h1, lo, hi, f)
             + DeviceLimitedMoE.histogram_quantile(h2, lo, hi, f)) / 2
    assert torch.equal(q_sum, DeviceLimitedMoE.histogram_quantile(
        whole, lo, hi, f))
    # averaging per-rank quantiles (the v5.5 distributed rule) is not the
    # global quantile when the ranks see different data; the error is
    # small here (~0.006 with one skewed and one unskewed rank: sigmoid
    # scores compress the margins) but above the histogram resolution
    width = (hi - lo) / m.quantile_bins
    assert (q_avg - q_sum).abs().max() > 2 * width, (q_avg - q_sum).abs().max()
    print("PASS test_summed_histograms_are_global "
          f"(rank-average error {(q_avg - q_sum).abs().max():.3f})")


def test_guards_and_state():
    m, hidden = _moe()
    x = _skewed(hidden, 256)
    m.eval()
    m(x)
    assert m.margin_hist.sum() == 0
    m.train()
    with torch.no_grad():
        m(x)
    assert m.margin_hist.sum() == 0
    full, _ = _moe(k=E)
    full(x)
    b = full.route_bias.clone()
    full.update_bias()
    assert torch.equal(full.route_bias, b)
    for s in ("heuristic", "quantile", "smebu"):
        keys = set(_moe(s)[0].state_dict())
        assert not {"margin_hist", "hist_range"} & keys, s
    for bad in (dict(balance_strategy="qb"), dict(quantile_bins=1),
                dict(quantile_bins=True)):
        try:
            HeliosLMv5Config(size="lite", moe=MoEConfig(**bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("PASS test_guards_and_state")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = {(r["scenario"], r["strategy"]): r for r in art["results"]}
    for sc in ("stationary", "drift", "micro"):
        qb = res[(sc, "quantile_balancing")]
        # fastest from the skewed start in every scenario
        for other in ("heuristic", "smebu", "quantile"):
            assert qb["maxvio_first20_mean"] \
                < res[(sc, other)]["maxvio_first20_mean"], (sc, other)
        # centred: no bias drift; the uncentred v5.5 rule drifts
        assert qb["bias_abs_mean_end"] < 1e-5
        assert res[(sc, "quantile")]["bias_abs_mean_end"] > 0.1
        # every token beats the last 512 margins once there are more
        # tokens per update than the v5.5 window holds
    assert res[("micro", "quantile_balancing")]["maxvio_tail_mean"] \
        < res[("micro", "quantile")]["maxvio_tail_mean"]
    # fastest recovery after the skew direction changes
    d = res[("drift", "quantile_balancing")]
    for other in ("heuristic", "smebu"):
        assert d["maxvio_after_drift20_mean"] \
            < res[("drift", other)]["maxvio_after_drift20_mean"], other
    # honest negative: in steady state SMEBU's momentum averages out the
    # per-batch sampling noise that a per-step solve inherits
    for sc in ("stationary", "micro"):
        assert res[(sc, "smebu")]["maxvio_tail_mean"] \
            < res[(sc, "quantile_balancing")]["maxvio_tail_mean"], sc
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact")


if __name__ == "__main__":
    test_histogram_quantile()
    test_repeated_updates_balance_a_fixed_batch()
    test_centred_and_ranking_invariant()
    test_summed_histograms_are_global()
    test_guards_and_state()
    test_ab_artifact()
