"""v5.94 (#42 prep): moe.dispatch = "batched".

Pins:

1. the default stays "blocked" and an unknown value is rejected;
2. batched equals blocked to float tolerance (outputs, and gradients of
   every parameter) for SwiGLU, gpt-oss clamped SwiGLU, SiTU (incl. the
   zero-centred norm), LatentMoE, shared experts, and a skewed routing
   where one expert takes most tokens and others take none;
3. the routing statistics (expert_load) are identical, since routing runs
   before dispatch;
4. a quantized / converted expert falls back to the blocked path;
5. a whole model with dispatch="batched" matches the blocked model;
6. under CPU bf16 autocast both paths stay close and keep z's dtype.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config, MoEConfig  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402

CASES = {
    "swiglu": {},
    "clamped": dict(swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_up_offset=1.0),
    "situ": dict(activation="situ"),
    "latent": dict(latent_dim=32),
    "shared": dict(num_shared_experts=1),
}


def _pair(seed=0, zero_centered=False, **moe):
    cfg = HeliosLMv5Config(size="lite", zero_centered_norm=zero_centered,
                           moe=MoEConfig(**moe))
    torch.manual_seed(seed)
    blocked = DeviceLimitedMoE(cfg)
    batched = copy.deepcopy(blocked)
    batched.dispatch = "batched"
    return cfg, blocked, batched


def _run(m, x):
    x = x.clone().requires_grad_(True)
    y = m(x)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    return y.detach(), x.grad, {n: p.grad for n, p in m.named_parameters()
                                if p.grad is not None}


def _assert_match(a, b, tol=1e-5):
    ya, gxa, ga = a
    yb, gxb, gb = b
    assert torch.allclose(ya, yb, atol=tol, rtol=tol), (ya - yb).abs().max()
    assert torch.allclose(gxa, gxb, atol=tol, rtol=tol)
    assert ga.keys() == gb.keys() and ga
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=tol, rtol=tol), n


def test_config_default_and_validation():
    assert MoEConfig().dispatch == "blocked"
    assert DeviceLimitedMoE(HeliosLMv5Config(size="lite")).dispatch == "blocked"
    with pytest.raises(ValueError, match="dispatch"):
        HeliosLMv5Config(size="lite", moe=MoEConfig(dispatch="grouped"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_batched_matches_blocked(name):
    cfg, blocked, batched = _pair(**CASES[name])
    assert batched._batchable()
    x = torch.randn(3, 37, cfg.hidden_size)
    blocked.train()
    batched.train()
    _assert_match(_run(blocked, x), _run(batched, x))
    assert torch.equal(blocked.expert_load, batched.expert_load)


def test_situ_zero_centered_norm():
    cfg, blocked, batched = _pair(zero_centered=True, activation="situ")
    with torch.no_grad():
        for m in (blocked, batched):
            for e in m.experts:
                e.norm.weight.uniform_(-0.5, 0.5)
        for eb, ea in zip(batched.experts, blocked.experts):
            eb.norm.weight.copy_(ea.norm.weight)
    x = torch.randn(2, 20, cfg.hidden_size)
    _assert_match(_run(blocked, x), _run(batched, x))


def test_skewed_routing_and_idle_experts():
    cfg, blocked, batched = _pair()
    with torch.no_grad():
        for m in (blocked, batched):
            m.route_bias.zero_()
            m.route_bias[5:7] = 100.0        # experts 5, 6 get every token
    x = torch.randn(4, 50, cfg.hidden_size)
    _assert_match(_run(blocked, x), _run(batched, x))
    load = batched.expert_load
    assert load[5] == load[6] == 200 and load.sum() == 400


def test_fallback_when_expert_is_converted():
    cfg, blocked, batched = _pair()

    class Wrapped(nn.Linear):
        pass

    e = batched.experts[0]
    w = Wrapped(e.w2.in_features, e.w2.out_features, bias=False)
    w.weight.data.copy_(e.w2.weight.data)
    e.w2 = w
    assert not batched._batchable()
    x = torch.randn(2, 10, cfg.hidden_size)
    with torch.no_grad():
        assert torch.equal(blocked(x), batched(x))   # blocked path, bit equal


def test_model_level():
    ids = torch.randint(3, 1000, (2, 24))
    outs = []
    for dispatch in ("blocked", "batched"):
        torch.manual_seed(0)
        cfg = HeliosLMv5Config(size="lite", moe=MoEConfig(dispatch=dispatch))
        model = HeliosLMv5(cfg).eval()
        with torch.no_grad():
            out = model(ids)
        outs.append(out["logits"] if isinstance(out, dict) else out[0])
    assert torch.allclose(outs[0], outs[1], atol=1e-4, rtol=1e-4)


def test_bf16_autocast():
    cfg, blocked, batched = _pair()
    x = torch.randn(2, 30, cfg.hidden_size)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ya, yb = blocked(x), batched(x)
    assert ya.dtype == yb.dtype
    assert torch.allclose(ya.float(), yb.float(), atol=3e-2, rtol=3e-2)
