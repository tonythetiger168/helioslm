"""T151 — ``hybrid_attention.recompute``: the chunked KDA recurrence is
recomputed in the backward pass instead of saving its fp32
intermediates, v5.157.

The first L4 run of the k3_llamacpp pilot (8 x 1024, 22 GB) ran out of
memory: each KDA layer saved about 260 MB per 1024-token row, against
about 40 MB for an MLA layer. Pins:

1. config validation and the pilot_config flag;
2. outputs, carried state and every gradient equal the saved path, for
   the k3 and the default per-channel style, with packed documents;
3. the layer saves far less for backward;
4. eval / no-grad forwards and the per-token loop are unchanged.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_kda_recompute.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, HybridAttentionConfig)
from helioslm_v5.src.attention.linear_attention import GatedDeltaAttention  # noqa: E402
from helioslm_v5.src.training.pretrain import pilot_config  # noqa: E402

STYLES = {"k3": dict(style="k3", short_conv_kernel=4),
          "default": dict()}


def _pair(style, chunk=16):
    torch.manual_seed(0)
    cfg = HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        enabled=True, per_channel_decay=True, chunk_size=chunk, **STYLES[style]))
    a = GatedDeltaAttention(cfg)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.1 * torch.randn_like(p))
    b = copy.deepcopy(a)
    b.recompute = True
    return a, b


def _fwd(m, x, **kw):
    x = x.clone().requires_grad_(True)
    y, present = m(x, use_cache=True, **kw)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    return y.detach(), present, x.grad, [p.grad for p in m.parameters()]


def _saved_bytes(fn):
    seen, total = set(), [0]

    def pack(t):
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total[0] += t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total[0]


def test_config():
    assert HybridAttentionConfig().recompute is False
    with pytest.raises(ValueError, match="recompute must be a bool"):
        HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
            enabled=True, recompute=1))
    for arch in ("k3_block", "k3_llamacpp"):
        cfg = pilot_config(arch, vocab_size=512, hidden_size=128, num_layers=4,
                           intermediate_size=256, kda_recompute=True)
        assert cfg.hybrid_attention.recompute is True
    assert pilot_config("k3_llamacpp").hybrid_attention.recompute is False


@pytest.mark.parametrize("style", list(STYLES))
@pytest.mark.parametrize("packed", [False, True])
def test_same_outputs_and_gradients(style, packed):
    a, b = _pair(style)
    B, T = 2, 45
    x = torch.randn(B, T, a.hidden_size, generator=torch.Generator().manual_seed(1))
    kw = {}
    if packed:
        kw["position_ids"] = torch.cat([torch.arange(20), torch.arange(25)]).expand(B, T)
    ya, pa, gxa, gpa = _fwd(a, x, **kw)
    yb, pb, gxb, gpb = _fwd(b, x, **kw)
    assert torch.equal(ya, yb)
    for sa, sb in zip(pa, pb):
        assert torch.equal(sa, sb)
    assert torch.allclose(gxa, gxb, atol=1e-6)
    for ga, gb in zip(gpa, gpb):
        assert torch.allclose(ga, gb, atol=1e-6)


def test_saves_less():
    a, b = _pair("k3", chunk=64)
    x = torch.randn(1, 256, a.hidden_size)
    full = _saved_bytes(lambda: a(x))
    rec = _saved_bytes(lambda: b(x))
    assert rec < 0.5 * full, (full, rec)


def test_no_grad_and_loop_unchanged():
    a, b = _pair("k3")
    x = torch.randn(1, 30, a.hidden_size)
    with torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])
    a.chunk_size = b.chunk_size = 0          # per-token loop ignores the flag
    with torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])
