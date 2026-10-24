"""HeliosLM v5 test cases — sparse-indexer stability probes.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import _pass


# ----------------------------------------------------------------------
# v5.43 (2/3): learned-sparse top-k self-consistency — zero noise is the
# perfect-stability oracle, full-selection (top_k == kv_len) stays 1.0
# under any noise, and loud errors guard the probe's own inputs
# ----------------------------------------------------------------------
def test_topk_self_consistency():
    from evastar.eval.sparse_stability import topk_self_consistency
    from evastar.src.attention.lightning_indexer import \
        LearnedLightningIndexer

    torch.manual_seed(0)
    indexer = LearnedLightningIndexer(hidden_size=32, kv_latent_dim=16,
                                      num_heads=4, head_dim=8).eval()
    h = torch.randn(1, 3, 32)
    c = torch.randn(1, 1, 20, 16)

    # Zero noise -> identical selections -> Jaccard exactly 1.0.
    r0 = topk_self_consistency(indexer, h, c, top_k=5, n_trials=4,
                               noise_std=0.0)
    assert r0["mean_jaccard"] == 1.0 and r0["min_jaccard"] == 1.0

    # Selecting EVERYTHING is stable under arbitrary noise (degenerate but
    # a real boundary: full selection cannot reshuffle).
    r_full = topk_self_consistency(indexer, h, c, top_k=20, n_trials=4,
                                   noise_std=0.5, seed=3)
    assert r_full["mean_jaccard"] == 1.0

    # Real noise on a random indexer: agreement stays in [0, 1] and is
    # deterministic across calls with the same seed.
    r1 = topk_self_consistency(indexer, h, c, top_k=5, n_trials=6,
                               noise_std=0.3, seed=7)
    r2 = topk_self_consistency(indexer, h, c, top_k=5, n_trials=6,
                               noise_std=0.3, seed=7)
    assert 0.0 <= r1["min_jaccard"] <= r1["mean_jaccard"] <= 1.0
    assert r1["mean_jaccard"] == r2["mean_jaccard"], "same seed must repeat"
    assert r1["mean_jaccard"] < 1.0, \
        "noise_std=0.3 on a random indexer should reshuffle some top-k sets"

    # Loud errors on bad probe inputs.
    for bad in (lambda: topk_self_consistency(indexer, h, c, top_k=0),
                lambda: topk_self_consistency(indexer, h, c, top_k=21),
                lambda: topk_self_consistency(indexer, h, c, top_k=5,
                                              n_trials=1),
                lambda: topk_self_consistency(indexer, h, c, top_k=5,
                                              noise_std=-0.1)):
        try:
            bad()
            assert False, "bad probe input must fail loudly"
        except ValueError:
            pass
    _pass("test_topk_self_consistency",
          f"noise 0.0 -> 1.0; noise 0.3 (seed 7) mean Jaccard "
          f"{r1['mean_jaccard']:.3f}; loud-input guards ok")


# ----------------------------------------------------------------------
# v5.44 (1/3): prefill top-k stability — the decode-step probe extended to
# EVERY query position; zero noise and full selection are the same two
# oracles, per-position scores are reported and averaged equally
# ----------------------------------------------------------------------
def test_prefill_stability():
    from evastar.eval.sparse_stability import prefill_stability
    from evastar.src.attention.lightning_indexer import \
        LearnedLightningIndexer

    torch.manual_seed(0)
    indexer = LearnedLightningIndexer(hidden_size=32, kv_latent_dim=16,
                                      num_heads=4, head_dim=8).eval()
    h = torch.randn(1, 3, 32)
    c = torch.randn(1, 1, 20, 16)

    # Zero noise -> every position perfectly consistent.
    r0 = prefill_stability(indexer, h, c, top_k=5, n_trials=4,
                           noise_std=0.0)
    assert r0["mean_jaccard"] == 1.0
    assert len(r0["per_position"]) == 3, "one score per query position"
    assert all(v == 1.0 for v in r0["per_position"])

    # Full selection (top_k == kv_len) is stable under arbitrary noise.
    r_full = prefill_stability(indexer, h, c, top_k=20, n_trials=4,
                               noise_std=0.5, seed=3)
    assert r_full["mean_jaccard"] == 1.0

    # Real noise: scores bounded, deterministic under the same seed, and
    # the overall score is the mean of the per-position means.
    r1 = prefill_stability(indexer, h, c, top_k=5, n_trials=6,
                           noise_std=0.3, seed=7)
    r2 = prefill_stability(indexer, h, c, top_k=5, n_trials=6,
                           noise_std=0.3, seed=7)
    assert 0.0 <= r1["mean_jaccard"] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in r1["per_position"])
    assert abs(r1["mean_jaccard"]
               - sum(r1["per_position"]) / len(r1["per_position"])) < 1e-12
    assert r1["mean_jaccard"] == r2["mean_jaccard"], "same seed must repeat"

    # Loud errors on bad probe inputs (same guards as the decode probe).
    for bad in (lambda: prefill_stability(indexer, h, c, top_k=0),
                lambda: prefill_stability(indexer, h, c, top_k=21),
                lambda: prefill_stability(indexer, h, c, top_k=5,
                                          n_trials=1),
                lambda: prefill_stability(indexer, h, c, top_k=5,
                                          noise_std=-0.1)):
        try:
            bad()
            assert False, "bad probe input must fail loudly"
        except ValueError:
            pass
    _pass("test_prefill_stability",
          f"noise 0.0 -> 1.0 over {len(r0['per_position'])} positions; "
          f"noise 0.3 (seed 7) mean {r1['mean_jaccard']:.3f}; loud guards ok")


# ----------------------------------------------------------------------
# v5.47 (3/3): indexer distillation stability floor — a loud quality
# gate that refuses to silently ship an indexer whose prefill top-k
# selection reshuffles past the caller's floor
# ----------------------------------------------------------------------
def test_indexer_stability_min_gate():
    from examples.train_indexer_distill import distill_indexer, stability_gate

    # Pass: layers at/above the floor (boundary inclusive).
    stability_gate({0: {"mean_jaccard": 0.95}, 1: {"mean_jaccard": 0.5}},
                   0.5)

    # Loud: below floor, boundary exclusive on the low side, empty stats,
    # malformed layer stats, floor outside [0, 1].
    for bad in (lambda: stability_gate({0: {"mean_jaccard": 0.4}}, 0.5),
                lambda: stability_gate({0: {"mean_jaccard": 0.49999}},
                                       0.5),
                lambda: stability_gate({}, 0.5),
                lambda: stability_gate({0: {}}, 0.5),
                lambda: stability_gate({0: {"mean_jaccard": 0.9}}, 1.5),
                lambda: stability_gate({0: {"mean_jaccard": 0.9}}, -0.1)):
        try:
            bad()
            assert False, "bad stability gate input must fail loudly"
        except ValueError:
            pass

    # The refusal names every offending layer with its measured value.
    try:
        stability_gate({0: {"mean_jaccard": 0.4}, 3: {"mean_jaccard": 0.2}},
                       0.5)
        assert False, "below-floor layers must fail loudly"
    except ValueError as e:
        assert "layer 0" in str(e) and "layer 3" in str(e), e

    # Wiring: stability_min without stability_probe is a loud error, not
    # a silent no-op (a gate on a measurement never taken is theatre).
    from evastar.src.model_v5 import EvastarV5
    cfg = EvastarV5Config(size="lite")
    cfg.attention.sparse_indexer = "learned"
    cfg.attention.sparse_top_k = 4
    torch.manual_seed(0)
    model = EvastarV5(cfg).eval()
    corpus = list(range(256)) * 2  # ids only; the guard fires at the end
    try:
        distill_indexer(model, corpus, steps=0, stability_min=0.9)
        assert False, "stability_min without stability_probe must fail"
    except ValueError as e:
        assert "stability_probe" in str(e), e
    _pass("test_indexer_stability_min_gate",
          "floor inclusive pass; below/empty/malformed/out-of-range loud; "
          "offenders listed; distill wiring refuses probe-less gating")
