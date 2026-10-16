"""HeliosLM v5 test cases — MoE routing, balancing, SiTU-GLU.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_expect_raises, _pass)


def test_sigmoid_moe():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
    config = HeliosLMv5Config(size="lite")
    torch.manual_seed(102)
    moe = DeviceLimitedMoE(config)
    x = torch.randn(2, 6, config.hidden_size)
    N = 2 * 6

    # 1) Selection uses bias; gating weights are bias-free.
    flat = x.reshape(-1, config.hidden_size)
    idx0, gates0 = moe._route(flat)
    with torch.no_grad():
        moe.route_bias.add_(torch.randn(config.moe.num_experts) * 0.05)
    idx1, gates1 = moe._route(flat)
    same = idx0 == idx1
    assert same.any(), "sanity: expected some unchanged selections"
    assert torch.equal(gates0[same], gates1[same]), \
        "gating weights changed when only the selection bias changed"

    # 2) A large bias forces selection of a targeted expert (bias affects selection).
    target = 3
    with torch.no_grad():
        moe.route_bias.zero_()
        moe.route_bias[target] = 100.0
    moe.expert_load.zero_()
    out = moe(x)
    assert out.shape == x.shape, f"output shape {out.shape} != input {x.shape}"
    assert moe.expert_load[target].item() == N, \
        f"biased expert selected {moe.expert_load[target].item()}/{N} times"

    # 3) update_bias: overloaded expert's bias decreases, underloaded increases,
    #    and the accumulated statistics are consumed.
    load = moe.expert_load.clone()
    before = moe.route_bias.clone()
    moe.update_bias()
    delta = moe.route_bias - before
    over, under = load > load.mean(), load < load.mean()
    assert load[target] > load.mean(), "sanity: target expert must be overloaded"
    assert (delta[over] < 0).all(), f"overloaded experts not penalized: {delta[over]}"
    assert (delta[under] > 0).all(), f"underloaded experts not boosted: {delta[under]}"
    assert moe.expert_load.sum().item() == 0, "update_bias must consume expert_load"

    # 4) Gradient flows to the router but NOT to route_bias (aux-free).
    moe.zero_grad(set_to_none=True)
    with torch.no_grad():
        moe.route_bias.zero_()
    moe(x).sum().backward()
    assert moe.router.weight.grad is not None \
        and moe.router.weight.grad.abs().sum() > 0, "router received no gradient"
    assert moe.route_bias.grad is None, "route_bias must not be gradient-trained"
    assert not moe.route_bias.requires_grad
    _pass("test_sigmoid_moe",
          f"bias selects-only, update_bias sign correct, router grad "
          f"norm={moe.router.weight.grad.norm():.3e}, route_bias.grad=None")


# ----------------------------------------------------------------------
# v5.5 (F3): LatentMoE — latent dims wired through, output shape, grads
# flow in both latent and full-width variants
# ----------------------------------------------------------------------
def test_latent_moe():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

    config = HeliosLMv5Config(size="lite")
    config.moe.latent_dim = 64
    torch.manual_seed(142)
    moe = DeviceLimitedMoE(config)
    H = config.hidden_size

    # Latent wiring: router + routed experts live in latent space; the
    # shared expert stays full-width.
    assert moe.down_proj.weight.shape == (64, H)
    assert moe.up_proj.weight.shape == (H, 64)
    assert moe.router.weight.shape == (config.moe.num_experts, 64)
    assert moe.experts[0].w13.in_features == 64
    assert moe.experts[0].w2.out_features == 64
    assert moe.shared_experts[0].w13.in_features == H

    x = torch.randn(2, 6, H, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    moe.zero_grad(set_to_none=True)
    out.sum().backward()
    for name, p in (("down_proj", moe.down_proj.weight),
                    ("up_proj", moe.up_proj.weight),
                    ("router", moe.router.weight),
                    ("expert0.w13", moe.experts[0].w13.weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"no gradient through latent MoE: {name}"

    # Full-width variant (v5.4 default) still backprops end-to-end.
    config2 = HeliosLMv5Config(size="lite")
    torch.manual_seed(142)
    moe2 = DeviceLimitedMoE(config2)
    assert moe2.down_proj is None and moe2.up_proj is None
    assert moe2.experts[0].w13.in_features == H
    x2 = torch.randn(2, 6, H, requires_grad=True)
    moe2(x2).sum().backward()
    assert moe2.router.weight.grad is not None \
        and moe2.router.weight.grad.abs().sum() > 0

    # Output differs between latent and full-width (real mechanism, and
    # LatentMoE has fewer routed-expert parameters).
    p_lat = sum(p.numel() for p in moe.experts.parameters())
    p_full = sum(p.numel() for p in moe2.experts.parameters())
    assert p_lat < p_full, "latent experts should shrink parameter count"
    _pass("test_latent_moe",
          f"latent=64 wiring OK, grads OK both variants, expert params "
          f"{p_lat} < full-width {p_full}")


# ----------------------------------------------------------------------
# v5.5 (F4): quantile balancing — drives load toward uniform at least as
# well as the heuristic update; bias is actually updated
# ----------------------------------------------------------------------
def test_quantile_balancing():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE


    def make(strategy, seed):
        config = HeliosLMv5Config(size="lite")
        config.moe.balance_strategy = strategy
        torch.manual_seed(seed)
        return DeviceLimitedMoE(config).train()

    moe_h = make("heuristic", 143)
    moe_q = make("quantile", 143)
    # Identical initial weights -> any divergence comes from the strategy.
    moe_q.load_state_dict(moe_h.state_dict(), strict=False)

    # Strongly skewed input: a dominant fixed direction makes the initial
    # ranking nearly input-independent -> a few experts are always picked.
    g = torch.Generator().manual_seed(144)
    direction = torch.randn(1, moe_h.hidden_size, generator=g)
    direction = direction / direction.norm()

    def batch(i):
        noise = 0.3 * torch.randn(64, moe_h.hidden_size,
                                  generator=torch.Generator().manual_seed(1000 + i))
        return (5.0 * direction + noise).unsqueeze(0)  # [1, 64, H] tokens

    def run(moe, steps=25):
        for i in range(steps):
            moe(batch(i))  # training mode: stats + margins accumulate
            moe.update_bias()
        # Measure the post-update load distribution over fresh batches.
        moe.expert_load.zero_()
        for i in range(4):
            moe(batch(100 + i))
        load = moe.expert_load.clone()
        moe.expert_load.zero_()
        return load

    load_h = run(moe_h)
    load_q = run(moe_q)
    cv_h = (load_h.std() / load_h.mean().clamp(min=1e-12)).item()
    cv_q = (load_q.std() / load_q.mean().clamp(min=1e-12)).item()
    assert load_h.mean() > 0 and load_q.mean() > 0
    assert cv_q <= cv_h + 1e-6, \
        f"quantile balancing worse than heuristic: CV {cv_q:.3f} vs {cv_h:.3f}"
    assert moe_q.route_bias.abs().sum() > 0, "quantile update never moved bias"
    # Sanity: the skew really is hard for the heuristic at this step size
    # (otherwise the comparison above is vacuous).
    assert cv_h > 0.5, f"test is vacuous: heuristic already balanced (CV {cv_h:.3f})"
    _pass("test_quantile_balancing",
          f"load CV: quantile {cv_q:.3f} <= heuristic {cv_h:.3f}, "
          f"|bias| sum={moe_q.route_bias.abs().sum():.3f}")

    # Config validation of the strategy enum.
    def _bad_strategy():
        c = HeliosLMv5Config(size="lite")
        c.moe.balance_strategy = "softmax"
        c._validate()
    _expect_raises(ValueError, _bad_strategy, "unknown balance_strategy")


# ----------------------------------------------------------------------
# v5.5 (M1/M2 regression): quantile margin boundary — margin > 0 iff the
# expert is actually selected (P(margin>0 | selected) == 1.0), and the
# quantile update keeps route_bias bounded over 300 updates (fixed point,
# not the unbounded linear drift of the off-by-one boundary)
# ----------------------------------------------------------------------
def test_quantile_margin_boundary():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

    config = HeliosLMv5Config(size="lite")
    config.moe.balance_strategy = "quantile"
    torch.manual_seed(160)
    moe = DeviceLimitedMoE(config).train()
    E = moe.num_experts

    # Known selection: a huge bias on expert 3 forces it into every token's
    # top-k; the rest of the ranking is data-dependent.
    with torch.no_grad():
        moe.route_bias[3] = 100.0
    flat = torch.randn(12, config.hidden_size)
    topk_indices, _ = moe._route(flat)  # train mode + grad on: records margins
    N = flat.shape[0]
    n = int(moe.margin_count)
    assert n == N, f"expected {N} recorded margins, got {n}"
    window = moe.margin_buffer[:, moe._MARGIN_BUF - n:]
    selected = torch.zeros(E, N, dtype=torch.bool)
    selected.scatter_(0, topk_indices.t(), True)
    margin_pos = window > 0
    p_pos_given_sel = margin_pos[selected].float().mean().item()
    assert p_pos_given_sel == 1.0, \
        f"P(margin>0 | selected) = {p_pos_given_sel:.3f} != 1.0 " \
        "(boundary off-by-one regression)"
    p_pos_given_unsel = margin_pos[~selected].float().mean().item()
    assert p_pos_given_unsel == 0.0, \
        f"P(margin>0 | not selected) = {p_pos_given_unsel:.3f} != 0.0"
    assert bool(margin_pos[3].all()), "forced expert must have margin > 0"

    # Boundedness: 300 forward+update cycles on a strongly skewed input
    # distribution. With the fixed boundary the bias converges to a fixed
    # point (selection frequency -> K/E); the off-by-one boundary made the
    # K/E target unreachable and the bias drifted linearly without bound
    # (mean |bias| ~3.65 after 300 steps in the buggy version).
    torch.manual_seed(161)
    moe2 = DeviceLimitedMoE(config).train()
    g = torch.Generator().manual_seed(162)
    direction = torch.randn(1, config.hidden_size, generator=g)
    direction = direction / direction.norm()

    def batch(seed):
        noise = 0.3 * torch.randn(64, config.hidden_size,
                                  generator=torch.Generator().manual_seed(seed))
        return (5.0 * direction + noise).unsqueeze(0)

    hist = {}
    for i in range(300):
        moe2(batch(2000 + i % 50))
        moe2.update_bias()
        if i in (250, 299):
            hist[i] = moe2.route_bias.detach().clone()
    max_abs = moe2.route_bias.abs().max().item()
    assert torch.isfinite(moe2.route_bias).all()
    assert max_abs < 2.0, \
        f"route_bias unbounded after 300 updates: max|bias|={max_abs:.3f}"
    late_drift = (hist[299] - hist[250]).abs().max().item()
    assert late_drift < 0.5, \
        f"bias still drifting late in training (fixed point not reached): " \
        f"{late_drift:.3f}"
    # The K/E target is now reachable: post-convergence load is near uniform.
    moe2.expert_load.zero_()
    for i in range(4):
        moe2(batch(3000 + i))
    load = moe2.expert_load.clone()
    moe2.expert_load.zero_()
    cv = (load.std() / load.mean().clamp(min=1e-12)).item()
    assert cv < 0.3, f"post-convergence load CV {cv:.3f} not near-uniform"
    _pass("test_quantile_margin_boundary",
          f"P(margin>0|selected)=1.0 exactly, 300 updates: max|bias|="
          f"{max_abs:.3f} (bounded), late drift {late_drift:.3f}, "
          f"load CV {cv:.3f}")


# ----------------------------------------------------------------------
# v5.5 (F5b): SiTU-GLU — soft cap is bounded, expert output finite under
# huge inputs, gradients flow
# ----------------------------------------------------------------------
def test_situ_glu():
    from helioslm_v5.src.moe.sigmoid_moe import (
        DeviceLimitedMoE, SiTUExpert, SwiGLUExpert, _situ_cap, build_expert)

    softcap = 3.0
    x_big = torch.randn(4, 32) * 1000.0
    t = _situ_cap(x_big, softcap)
    # |t(x)| < softcap mathematically; fp32 tanh saturates to exactly 1.0
    # for large |x|, hence <= with a rounding allowance.
    assert bool((t.abs() <= softcap + 1e-6).all()), \
        f"soft cap violated: |t(x)| max {t.abs().max():.3e} > {softcap}"
    # Identity near zero (t(x) ~= x for |x| << softcap).
    x_small = torch.randn(8, 16) * 0.01
    assert ((_situ_cap(x_small, softcap) - x_small).abs().max()
            < 1e-4), "soft cap should be ~identity near zero"

    config = HeliosLMv5Config(size="lite")
    config.moe.activation = "situ"
    config.moe.situ_softcap = softcap
    torch.manual_seed(146)
    expert = build_expert(config, 32, 32)
    assert isinstance(expert, SiTUExpert)
    x = (torch.randn(4, 32) * 100.0).requires_grad_(True)
    out = expert(x)
    assert out.shape == (4, 32) and torch.isfinite(out).all(), \
        "SiTU expert output not finite under huge inputs"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() \
        and x.grad.abs().sum() > 0, "no gradient through SiTU-GLU"
    assert expert.w13.weight.grad is not None \
        and expert.norm.weight.grad is not None

    # MoE-level: "situ" experts are wired everywhere, "swiglu" is default.
    moe = DeviceLimitedMoE(config)
    assert all(isinstance(e, SiTUExpert) for e in moe.experts)
    assert all(isinstance(e, SiTUExpert) for e in moe.shared_experts)
    h = torch.randn(2, 5, config.hidden_size)
    out_m = moe(h)
    assert out_m.shape == h.shape and torch.isfinite(out_m).all()

    config_sw = HeliosLMv5Config(size="lite")
    assert isinstance(build_expert(config_sw, 32, 32), SwiGLUExpert)

    def _bad_act():
        c = HeliosLMv5Config(size="lite")
        c.moe.activation = "gelu"
        c._validate()
    _expect_raises(ValueError, _bad_act, "unknown activation")
    _pass("test_situ_glu",
          f"|t(x)| < {softcap} (max {t.abs().max():.3f} @|x|=1000), "
          f"finite output, grads OK, MoE wiring OK, bad activation raises")
