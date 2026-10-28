"""T104 — roadmap B2-4: learnable mHC read matrices with a Sinkhorn
projection (GLM-5.3-Flash direction), v5.66.

Pins:

1. off by default: no extra parameters; on, at init A == I exactly and
   the model output is bit-identical to the static-HC model;
2. the constraint: for random logits (including large ones) and any raw
   alpha (clamped to [0, 1]), A is non-negative, columns sum to 1 and
   rows sum to 1 within the Sinkhorn tolerance;
3. the constraint survives training: after optimizer steps every read
   matrix is still doubly stochastic; once the branches differ, alpha
   gets a gradient at its init value 0 (it can leave the identity; at
   exact init the gradient is 0 by branch symmetry);
4. cached decode == full forward with non-trivial read matrices;
5. regression for the v5.66 branch-fold fix in the static HC path;
6. validation.

Run from repo root: python3 helioslm_v5/tests/test_mhc_sinkhorn.py
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5, HeliosLMv5Layer  # noqa: E402

TOL = 1e-5


def _model(sinkhorn, **kw):
    cfg = HeliosLMv5Config(size="lite", use_hyper_connections=True,
                           hyper_connection_sinkhorn=sinkhorn, **kw)
    torch.manual_seed(0)
    return HeliosLMv5(cfg), cfg


def _assert_doubly_stochastic(A):
    assert (A >= 0).all()
    assert (A.sum(0) - 1).abs().max() < TOL, A.sum(0)
    assert (A.sum(1) - 1).abs().max() < TOL, A.sum(1)


def test_off_by_default_and_identity_at_init():
    off, cfg = _model(False)
    assert not any("logits" in n or "alpha" in n
                   for n, _ in off.named_parameters())
    on, _ = _model(True)
    for layer in on.layers:
        for which in ("attn", "moe"):
            assert torch.equal(layer.hc_read_matrix(which),
                               torch.eye(cfg.hyper_connection_branches))
    ids = torch.randint(3, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.equal(off.eval()(ids)[0], on.eval()(ids)[0])
    print("PASS test_off_by_default_and_identity_at_init")


def test_constraint_random_and_extreme():
    on, _ = _model(True)
    layer = on.layers[0]
    gen = torch.Generator().manual_seed(3)
    for scale, alpha in ((1.0, 0.3), (10.0, 0.9), (50.0, 1.0),
                         (5.0, -2.0), (5.0, 7.0)):
        with torch.no_grad():
            layer.hc_attn_logits.copy_(
                torch.randn(layer.hc_attn_logits.shape, generator=gen) * scale)
            layer.hc_attn_alpha.fill_(alpha)
        _assert_doubly_stochastic(layer.hc_read_matrix("attn"))
        S = HeliosLMv5Layer.sinkhorn(layer.hc_attn_logits,
                                     layer.hc_sinkhorn_iters)
        assert torch.isfinite(S).all()
    print("PASS test_constraint_random_and_extreme")


def test_constraint_survives_training():
    on, cfg = _model(True)
    on.train()
    opt = torch.optim.AdamW(on.parameters(), lr=5e-2)
    ids = torch.randint(3, cfg.vocab_size, (2, 10))
    # At init every branch is a copy of the same stream and Sinkhorn of
    # zero logits is uniform, so (S - I) @ streams == 0 and alpha's
    # gradient is exactly 0 by symmetry. Once the write matrices B make
    # the branches differ, alpha must get a gradient AT 0 (the clamp
    # boundary passes it), so it can leave the identity.
    with torch.no_grad():
        for layer in on.layers:
            layer.hc_attn_B.copy_(torch.randn(4, 4) * 0.1)
            layer.hc_moe_B.copy_(torch.randn(4, 4) * 0.1)
    on(ids)[0].logsumexp(-1).mean().backward()
    assert all(l.hc_attn_alpha.item() == 0.0 for l in on.layers)
    assert any(l.hc_attn_alpha.grad.abs() > 0 for l in on.layers)
    for _ in range(5):
        opt.zero_grad()
        on(ids)[0].logsumexp(-1).mean().backward()
        opt.step()
    moved = False
    for layer in on.layers:
        for which in ("attn", "moe"):
            A = layer.hc_read_matrix(which).detach()
            _assert_doubly_stochastic(A)
            moved |= not torch.equal(A, torch.eye(A.shape[0]))
    assert moved, "training never left the identity"
    print("PASS test_constraint_survives_training")


def test_decode_equals_full():
    on, cfg = _model(True)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in on.layers:
            for which in ("attn", "moe"):
                getattr(layer, f"hc_{which}_logits").copy_(
                    torch.randn(4, 4, generator=gen))
                getattr(layer, f"hc_{which}_alpha").fill_(0.5)
                getattr(layer, f"hc_{which}_B").copy_(
                    torch.randn(4, 4, generator=gen) * 0.1)
    on.eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        ref = on(ids)[0]
        lg, _, past = on(ids[:, :6], use_cache=True)
        outs = [lg]
        for t in range(6, 12):
            lg, _, past = on(ids[:, t:t + 1], past_key_values=past,
                             use_cache=True)
            outs.append(lg)
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
    print("PASS test_decode_equals_full")


def test_static_hc_fold_regression():
    """v5.66 fix: folding [B, L, n, d] branches into the batch dim with a
    plain reshape interleaved positions and branches. With the static
    read matrix and a non-zero write matrix B (any trained HC model),
    cached decode differed from the full forward by ~0.5. Now: decode ==
    full, the stack stays causal, and each branch's sublayer input is the
    branch's own unscrambled sequence."""
    off, cfg = _model(False)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in off.layers:
            layer.hc_attn_B.copy_(torch.randn(4, 4, generator=gen) * 0.1)
            layer.hc_moe_B.copy_(torch.randn(4, 4, generator=gen) * 0.1)
    off.eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 12), generator=gen)
    with torch.no_grad():
        ref = off(ids)[0]
        lg, _, past = off(ids[:, :6], use_cache=True)
        outs = [lg]
        for t in range(6, 12):
            lg, _, past = off(ids[:, t:t + 1], past_key_values=past,
                              use_cache=True)
            outs.append(lg)
        assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
        ids2 = ids.clone()
        ids2[:, 9] = (ids2[:, 9] + 1) % cfg.vocab_size
        # v5.110: torch.equal -> tolerance. fp32 reduction order differs
        # between the full forward and the cached decode path, so bitwise
        # equality flakes on pure noise: measured max|diff| = 5.96e-07
        # over 18432 elements (15772 non-zero) on a Windows host. The
        # 1e-5 bound sits 5 orders below this test's own 1e-4 decode
        # tolerance and 6 orders below the ~0.5 leakage the v5.66 fold
        # bug produced, so it cannot mask a real causality leak.
        causal_diff = (off(ids2)[0][:, :9] - ref[:, :9]).abs().max().item()
        assert causal_diff < 1e-5, causal_diff

    layer = off.layers[0]
    seen = []
    layer.pre_attn_norm.register_forward_hook(
        lambda m, i, o: seen.append(i[0].detach().clone()))
    streams = torch.randn(1, 5, 4, cfg.hidden_size)    # distinct branches
    with torch.no_grad():
        layer._hc_step(streams, torch.eye(4), torch.zeros(4, 4),
                       layer.attention, layer.pre_attn_norm, None, None,
                       False, None, returns_kv=True)
    folded = seen[0].view(1, 4, 5, cfg.hidden_size)
    assert torch.equal(folded, streams.permute(0, 2, 1, 3))
    print("PASS test_static_hc_fold_regression")


def test_validation():
    for kw, needle in ((dict(hyper_connection_sinkhorn=True),
                        "requires use_hyper_connections"),
                       (dict(use_hyper_connections=True,
                             hyper_connection_sinkhorn=1), "must be a bool"),
                       (dict(use_hyper_connections=True,
                             hyper_connection_sinkhorn=True,
                             hyper_connection_sinkhorn_iters=0), "iters")):
        try:
            HeliosLMv5Config(size="lite", **kw)
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


if __name__ == "__main__":
    test_off_by_default_and_identity_at_init()
    test_constraint_random_and_extreme()
    test_constraint_survives_training()
    test_decode_equals_full()
    test_static_hc_fold_regression()
    test_validation()
