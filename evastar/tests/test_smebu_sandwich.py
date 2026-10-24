"""T111 — roadmap U7 (Arcee Trinity): SMEBU expert-bias updates and the
depth-scaled sandwich norm, v5.72.

Pins:

1. defaults unchanged: identical state dict keys and output; no momentum
   buffer and no post norms unless enabled;
2. SMEBU oracle: over a sequence of loads, route_bias and the momentum
   equal an independent re-implementation of
   v = (mean - load) / mean, m <- beta m + (1 - beta) tanh(kappa v),
   bias += lambda m; a balanced (or empty) load leaves the bias alone;
   every step is bounded by lambda, and a small violation moves the bias
   less than the sign-based update does;
3. sandwich norm: post-norm gains start at exactly 1/sqrt(num_layers)
   (also with zero-centered norms), the layer computes
   h + post(sublayer(pre(h))) (manual oracle), cached decode == full
   forward with and without hyper-connections, gradients reach the gains
   (plain path; HC's write-back matrix starts at 0 and blocks them at init);
4. validation;
5. the committed A/B artifact: SMEBU flips bias direction less often at
   every step size and lowers the steady-state MaxVio variance at the two
   larger ones; hash.

Run from repo root: python3 evastar/tests/test_smebu_sandwich.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import EvastarV5Config, MoEConfig  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402

_ART = _REPO / "benchmarks" / "smebu_balance_ab_2026-10-16.json"


def _moe(strategy, lam=1e-2, beta=0.5, kappa=2.0):
    cfg = EvastarV5Config(size="lite", moe=MoEConfig(
        balance_strategy=strategy, bias_update_rate=lam, smebu_beta=beta,
        smebu_kappa=kappa))
    torch.manual_seed(0)
    return DeviceLimitedMoE(cfg)


def test_defaults_unchanged():
    ids = torch.randint(3, 1000, (2, 9))
    outs, keys = [], []
    for cfg in (EvastarV5Config(size="lite"),
                EvastarV5Config(size="lite", sandwich_norm=False,
                                 moe=MoEConfig(smebu_beta=0.5, smebu_kappa=2.0))):
        torch.manual_seed(0)
        m = EvastarV5(cfg).eval()
        keys.append(list(m.state_dict()))
        with torch.no_grad():
            outs.append(m(ids)[0])
        assert all(l.post_attn_norm is None and l.post_moe_norm is None
                   for l in m.layers)
    assert keys[0] == keys[1] and torch.equal(outs[0], outs[1])
    assert not any("bias_momentum" in k or "post_" in k for k in keys[0])
    assert not hasattr(_moe("heuristic"), "bias_momentum")
    print("PASS test_defaults_unchanged")


def test_smebu_oracle():
    lam, beta, kappa = 1e-2, 0.5, 2.0
    moe = _moe("smebu", lam, beta, kappa)
    E = moe.num_experts
    gen = torch.Generator().manual_seed(3)
    b = torch.zeros(E, dtype=torch.float64)
    m = torch.zeros(E, dtype=torch.float64)
    for _ in range(12):
        load = torch.randint(0, 50, (E,), generator=gen).float()
        before = moe.route_bias.clone()
        moe.update_bias(load)
        lo = load.double()
        v = (lo.mean() - lo) / lo.mean()
        m = beta * m + (1 - beta) * torch.tanh(kappa * v)
        b = b + lam * m
        assert torch.allclose(moe.route_bias.double(), b, atol=1e-6)
        assert torch.allclose(moe.bias_momentum.double(), m, atol=1e-6)
        assert (moe.route_bias - before).abs().max() <= lam + 1e-7
    # balanced or empty load: momentum decays, no new push
    moe2 = _moe("smebu", lam)
    for load in (torch.full((E,), 7.0), torch.zeros(E)):
        moe2.update_bias(load)
        assert torch.equal(moe2.route_bias, torch.zeros(E))
    # a 5% overload moves SMEBU's bias far less than the sign step
    small = torch.full((E,), 100.0)
    small[0] = 105.0
    for strategy in ("heuristic", "smebu"):
        mm = _moe(strategy, lam)
        mm.update_bias(small)
        step = mm.route_bias[0].abs().item()
        if strategy == "heuristic":
            sign_step = step
    assert step < 0.1 * sign_step, (step, sign_step)
    print("PASS test_smebu_oracle")


def _sandwich_cfg(**kw):
    return EvastarV5Config(size="lite", num_hidden_layers=4,
                            sandwich_norm=True, **kw)


def test_sandwich_gain_and_layer_oracle():
    for zc in (False, True):
        torch.manual_seed(0)
        m = EvastarV5(_sandwich_cfg(zero_centered_norm=zc))
        for l in m.layers:
            for n in (l.post_attn_norm, l.post_moe_norm):
                gain = (1.0 + n.weight) if zc else n.weight
                assert torch.allclose(gain, torch.full_like(gain, 0.5)), gain[:3]
    # manual oracle for one layer (attention residuals off in lite)
    torch.manual_seed(0)
    m = EvastarV5(_sandwich_cfg()).eval()
    layer = m.layers[1]
    h = torch.randn(2, 6, m.config.hidden_size)
    with torch.no_grad():
        got = layer(h)[0]
        a = layer.post_attn_norm(layer.attention(layer.pre_attn_norm(h))[0])
        h1 = h + a
        ref = h1 + layer.post_moe_norm(layer.moe(layer.pre_moe_norm(h1)))
    assert torch.allclose(got, ref, atol=1e-6)
    print("PASS test_sandwich_gain_and_layer_oracle")


def test_sandwich_decode_and_grad():
    ids = torch.randint(3, 1000, (2, 11))
    for kw in (dict(), dict(use_hyper_connections=True)):
        torch.manual_seed(0)
        m = EvastarV5(_sandwich_cfg(**kw)).eval()
        with torch.no_grad():
            full = m(ids)[0]
            lg, _, past = m(ids[:, :4], use_cache=True)
            outs = [lg]
            for t in range(4, 11):
                lg, _, past = m(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
                outs.append(lg)
        assert (torch.cat(outs, 1) - full).abs().max() < 1e-4, kw
        if kw:
            # HC writes back through B, which starts at 0, so no gradient
            # reaches the sublayer output (or its post norm) at init.
            continue
        m.train()
        m(ids)[0].logsumexp(-1).mean().backward()
        g = m.layers[0].post_attn_norm.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print("PASS test_sandwich_decode_and_grad")


def test_validation():
    for kw, needle in (
            (dict(moe=MoEConfig(balance_strategy="sign")), "balance_strategy"),
            (dict(moe=MoEConfig(smebu_beta=1.0)), "smebu_beta"),
            (dict(moe=MoEConfig(smebu_beta=True)), "smebu_beta"),
            (dict(moe=MoEConfig(smebu_kappa=0)), "smebu_kappa"),
            (dict(sandwich_norm=1), "sandwich_norm")):
        try:
            EvastarV5Config(size="lite", **kw)
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = art["results"]
    by = {(r["lambda"], r["strategy"]): r for r in res}
    lams = sorted({r["lambda"] for r in res})
    for lam in lams:
        h, s = by[(lam, "heuristic")], by[(lam, "smebu")]
        assert s["bias_sign_flips_per_step"] < h["bias_sign_flips_per_step"]
    for lam in lams[1:]:
        assert by[(lam, "smebu")]["maxvio_tail_var"] < \
            by[(lam, "heuristic")]["maxvio_tail_var"], lam
    assert all(math.isfinite(r["maxvio_tail_mean"]) for r in res)
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(res, sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_smebu_oracle()
    test_sandwich_gain_and_layer_oracle()
    test_sandwich_decode_and_grad()
    test_validation()
    test_ab_artifact()
