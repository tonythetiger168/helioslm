"""v5.95 (#42 prep): chunked-parallel delta rule (hybrid_attention.chunk_size).

Pins:

1. chunk_size defaults to 0 (the per-token loop) and must be an int >= 0;
2. chunked_delta_rule equals the per-token loop in float64 to 1e-10 for
   per-head and per-channel decay, with padding (keep), packed document
   starts, a carried state, and chunk lengths 1, 7, 32 and > T;
3. tiny decays (1e-6, the clamp floor) stay finite;
4. GatedDeltaAttention with chunk_size matches the loop (outputs, cache,
   gradients) incl. attention_mask padding, packed position_ids and the
   short conv; the returned cache is still a StateTensor;
5. a chunked prefill followed by per-token decode matches running the
   loop throughout;
6. a whole hybrid model with chunk_size matches the default model.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, HybridAttentionConfig)
from helioslm_v5.src.attention.linear_attention import (  # noqa: E402
    GatedDeltaAttention, chunked_delta_rule, is_recurrent_state)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402


def _cfg(**hybrid):
    return HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        enabled=True, **hybrid))


def _loop(q, k, v, decay, keep, seg_start, state):
    """The module's per-token recurrence, as a free function."""
    m = GatedDeltaAttention.__new__(GatedDeltaAttention)
    m.per_channel_decay = decay.dim() == 4
    return GatedDeltaAttention._sequential(m, q, k, v, decay, keep, seg_start, state)


def _inputs(per_channel, use_keep, use_seg, carry, B=2, H=3, T=90, D=8,
            dtype=torch.float64, low=None):
    g = torch.Generator().manual_seed(1)
    q = torch.randn(B, H, T, D, generator=g, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, D, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, D, generator=g, dtype=dtype)
    shape = (B, H, T, D) if per_channel else (B, H, T)
    decay = torch.sigmoid(2 * torch.randn(shape, generator=g, dtype=dtype) + 1)
    if low is not None:
        decay = torch.full(shape, low, dtype=dtype)
    decay = decay.clamp(1e-6, 1 - 1e-6)
    keep = None
    if use_keep:
        keep = (torch.rand(B, 1, T, generator=g) > 0.25).to(dtype)
        decay = 1.0 + (decay - 1.0) * (keep[..., None] if per_channel else keep)
    seg = None
    if use_seg:
        seg = torch.zeros(B, T, dtype=torch.bool)
        seg[:, 0] = True
        seg[0, 31] = seg[1, 32] = seg[1, 40] = True
    state = torch.randn(B, H, D, D, generator=g, dtype=dtype) if carry \
        else torch.zeros(B, H, D, D, dtype=dtype)
    return q, k, v, decay, keep, seg, state


def test_config():
    assert HybridAttentionConfig().chunk_size == 0
    assert GatedDeltaAttention(_cfg()).chunk_size == 0
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="chunk_size"):
            _cfg(chunk_size=bad)


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("use_keep", [False, True])
@pytest.mark.parametrize("use_seg,carry", [(False, False), (False, True),
                                           (True, False)])
@pytest.mark.parametrize("chunk", [1, 7, 32, 128])
def test_chunked_equals_loop_fp64(per_channel, use_keep, use_seg, carry, chunk):
    args = _inputs(per_channel, use_keep, use_seg, carry)
    o1, s1 = _loop(*args)
    o2, s2 = chunked_delta_rule(*args, chunk)
    assert o2.dtype == o1.dtype and s2.dtype == s1.dtype
    assert (o1 - o2).abs().max() < 1e-10
    assert (s1 - s2).abs().max() < 1e-10


@pytest.mark.parametrize("per_channel", [False, True])
def test_tiny_decay_finite(per_channel):
    args = _inputs(per_channel, False, False, True, dtype=torch.float32, low=1e-6)
    o1, s1 = _loop(*args)
    o2, s2 = chunked_delta_rule(*args, 32)
    assert torch.isfinite(o2).all() and torch.isfinite(s2).all()
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)


def _pair(chunk=16, **hybrid):
    cfg = _cfg(**hybrid)
    torch.manual_seed(0)
    a = GatedDeltaAttention(cfg)
    b = copy.deepcopy(a)
    b.chunk_size = chunk
    return cfg, a, b


def _fwd(m, x, **kw):
    x = x.clone().requires_grad_(True)
    y, present = m(x, use_cache=True, **kw)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    return y.detach(), present, x.grad, [p.grad for p in m.parameters()]


@pytest.mark.parametrize("hybrid", [{}, {"per_channel_decay": True},
                                    {"short_conv_kernel": 4}])
@pytest.mark.parametrize("layout", ["plain", "mask", "packed"])
def test_module_matches_loop(hybrid, layout):
    cfg, a, b = _pair(**hybrid)
    B, T = 2, 45
    x = torch.randn(B, T, cfg.hidden_size)
    kw = {}
    if layout == "mask":
        mask = torch.ones(B, T)
        mask[0, :6] = 0
        mask[1, 30:33] = 0
        kw["attention_mask"] = mask
    if layout == "packed":
        pos = torch.cat([torch.arange(20), torch.arange(25)]).expand(B, T)
        kw["position_ids"] = pos
    ya, pa, gxa, gpa = _fwd(a, x, **kw)
    yb, pb, gxb, gpb = _fwd(b, x, **kw)
    assert torch.allclose(ya, yb, atol=1e-5)
    assert is_recurrent_state(pb[0])
    for sa, sb in zip(pa, pb):
        assert torch.allclose(sa, sb, atol=1e-5)
    assert torch.allclose(gxa, gxb, atol=1e-5)
    for ga, gb in zip(gpa, gpb):
        assert torch.allclose(ga, gb, atol=1e-5)


def test_chunked_prefill_then_decode():
    cfg, a, b = _pair(chunk=8, per_channel_decay=True)
    x = torch.randn(1, 30, cfg.hidden_size)
    with torch.no_grad():
        full, _ = a(x)
        y, past = b(x[:, :24], use_cache=True)
        steps = [y]
        for t in range(24, 30):
            y, past = b(x[:, t:t + 1], past_key_value=past, use_cache=True)
            steps.append(y)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5)


def test_model_level():
    ids = torch.randint(3, 1000, (2, 40))
    outs = []
    for chunk in (0, 16):
        torch.manual_seed(0)
        model = HeliosLMv5(_cfg(per_channel_decay=True, chunk_size=chunk)).eval()
        assert any(getattr(m, "is_recurrent_attention", False)
                   for m in model.modules())
        with torch.no_grad():
            out = model(ids)
        outs.append(out["logits"] if isinstance(out, dict) else out[0])
    assert torch.allclose(outs[0], outs[1], atol=1e-4)
