"""T106 — roadmap U5: gpt-oss clamped SwiGLU for MoE experts, v5.68.

Pins:

1. defaults unchanged: identical state dict and output, and the expert
   still takes the original ``F.silu`` path;
2. oracle: with gpt-oss settings (limit 7, alpha 1.702, up_offset 1) the
   expert equals an independent re-implementation of gpt-oss's
   ``swiglu`` (gate/linear split, clamps, sigmoid gate, +1);
3. the bound: for inputs scaled up to 1e4, the pre-w2 activation stays
   within the corner products of the gate range [-0.2785/alpha, limit]
   and the linear range [up_offset - limit, up_offset + limit] (gpt-oss:
   [-42, 56]), while the unclamped expert's grows without bound;
4. a huge limit with alpha 1 and offset 0 reduces to SiLU (allclose);
5. a model with gpt-oss experts: cached decode == full forward, finite
   gradients;
6. validation.

Run from repo root: python3 evastar/tests/test_clamped_swiglu.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import EvastarV5Config, MoEConfig  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.moe.sigmoid_moe import SwiGLUExpert  # noqa: E402

OSS = dict(limit=7.0, alpha=1.702, up_offset=1.0)


def _gpt_oss_reference(x, w13, w2, alpha=1.702, limit=7.0):
    """Independent re-implementation of gpt-oss's MoE ``swiglu``: the
    fused projection splits into (glu, linear) halves, glu is clamped
    from above, linear on both sides, out_glu = glu * sigmoid(alpha*glu),
    and the result is out_glu * (linear + 1)."""
    h = x @ w13.t()
    x_glu, x_linear = h[..., :h.shape[-1] // 2], h[..., h.shape[-1] // 2:]
    x_glu = torch.clamp(x_glu, min=None, max=limit)
    x_linear = torch.clamp(x_linear, min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)) @ w2.t()


def test_defaults_unchanged():
    torch.manual_seed(0)
    base = EvastarV5(EvastarV5Config(size="lite"))
    torch.manual_seed(0)
    expl = EvastarV5(EvastarV5Config(size="lite", moe=MoEConfig(
        swiglu_limit=None, swiglu_alpha=1.0, swiglu_up_offset=0.0)))
    sa, sb = base.state_dict(), expl.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    ids = torch.randint(3, 1000, (2, 9))
    with torch.no_grad():
        assert torch.equal(base.eval()(ids)[0], expl.eval()(ids)[0])
    e = base.layers[0].moe.experts[0]
    assert not e.clamped
    x = torch.randn(5, 256)
    a, b = e.w13(x).chunk(2, dim=-1)
    with torch.no_grad():
        assert torch.equal(e(x), e.w2(F.silu(a) * b))
    print("PASS test_defaults_unchanged")


def test_matches_gpt_oss_reference():
    torch.manual_seed(1)
    e = SwiGLUExpert(64, 96, 64, **OSS)
    x = torch.randn(10, 64) * 5          # large enough to hit the clamps
    with torch.no_grad():
        ref = _gpt_oss_reference(x, e.w13.weight, e.w2.weight)
        assert torch.allclose(e(x), ref, atol=1e-5), (e(x) - ref).abs().max()
        h = e.w13(x)
        assert (h.abs() > OSS["limit"]).any(), "clamps were not exercised"
    print("PASS test_matches_gpt_oss_reference")


def test_activation_bound():
    torch.manual_seed(2)
    lim, alpha, off = OSS["limit"], OSS["alpha"], OSS["up_offset"]
    # gate term g = a*sigmoid(alpha*a) in [g_lo, g_hi], linear term
    # u = b + off in [off - lim, off + lim]; the product's extremes are
    # at the interval corners.
    g_lo, g_hi = -0.2785 / alpha, lim
    corners = [g * u for g in (g_lo, g_hi) for u in (off - lim, off + lim)]
    lo, hi = min(corners), max(corners)
    for clamped in (True, False):
        e = SwiGLUExpert(32, 48, 32, **(OSS if clamped else {}))
        seen = []
        e.w2.register_forward_hook(lambda m, i, o: seen.append(i[0]))
        for scale in (1.0, 100.0, 1e4):
            with torch.no_grad():
                e(torch.randn(64, 32) * scale)
        act = torch.cat([s.flatten() for s in seen])
        if clamped:
            assert act.min() >= lo - 1e-4 and act.max() <= hi + 1e-4, \
                (act.min().item(), act.max().item(), lo, hi)
        else:
            assert act.abs().max() > 100 * hi       # unbounded without clamp
    print(f"PASS test_activation_bound (bound [{lo:.2f}, {hi:.1f}])")


def test_huge_limit_is_silu():
    torch.manual_seed(3)
    e = SwiGLUExpert(32, 48, 32, limit=1e9)
    assert e.clamped
    x = torch.randn(8, 32)
    with torch.no_grad():
        a, b = e.w13(x).chunk(2, dim=-1)
        assert torch.allclose(e(x), e.w2(F.silu(a) * b), atol=1e-6)
    print("PASS test_huge_limit_is_silu")


def test_model_with_gpt_oss_experts():
    cfg = EvastarV5Config(size="lite", moe=MoEConfig(
        swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_up_offset=1.0))
    torch.manual_seed(0)
    model = EvastarV5(cfg)
    assert all(e.clamped for l in model.layers for e in l.moe.experts)
    assert all(e.clamped for l in model.layers for e in l.moe.shared_experts)
    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    model.train()
    model(ids)[0].logsumexp(-1).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    model.eval()
    with torch.no_grad():
        ref = model(ids)[0]
        lg, _, past = model(ids[:, :6], use_cache=True)
        outs = [lg]
        for t in range(6, 12):
            lg, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
            outs.append(lg)
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
    print("PASS test_model_with_gpt_oss_experts")


def test_validation():
    for kw, needle in ((dict(swiglu_limit=0), "swiglu_limit"),
                       (dict(swiglu_limit=True), "swiglu_limit"),
                       (dict(swiglu_alpha=0), "swiglu_alpha"),
                       (dict(swiglu_up_offset="1"), "swiglu_up_offset")):
        try:
            EvastarV5Config(size="lite", moe=MoEConfig(**kw))
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_matches_gpt_oss_reference()
    test_activation_bound()
    test_huge_limit_is_silu()
    test_model_with_gpt_oss_experts()
    test_validation()
