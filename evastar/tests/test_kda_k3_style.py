"""T137 — KDA layer in the style of llama.cpp's kimi-k3 graph, v5.141.

``hybrid_attention.style="k3"`` makes GatedDeltaAttention compute what
llama.cpp's kimi-k3 KDA layer computes, so hybrid models can export to
GGUF. Pins:

1. config validation (needs per-channel decay and a short conv);
2. the layer equals an independent reference written from the llama.cpp
   graph (separate q/k/v convs + SiLU, L2 norm of q and k, log-decay
   ``lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``, decay before
   the retrieval, sigmoid beta, per-head RMS norm times a sigmoid gate);
3. the chunked form equals the per-token loop (outputs, cache, input and
   parameter gradients) with padding and packed documents;
4. chunked prefill then per-token decode equals a one-shot forward;
5. ``chunked_delta_rule(log_decay=True, decay_first=True)`` equals the
   loop in fp64; the default style is unchanged;
6. the initial decay is about 0.982, as in the default style.

Run from repo root: python3 -m pytest evastar/tests/test_kda_k3_style.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evastar.configs.config_v5 import (  # noqa: E402
    EvastarV5Config, HybridAttentionConfig)
from evastar.src.attention.linear_attention import (  # noqa: E402
    GatedDeltaAttention, chunked_delta_rule)
from evastar.src.model_v5 import EvastarV5  # noqa: E402

K3 = dict(enabled=True, style="k3", per_channel_decay=True, short_conv_kernel=4)


def _cfg(**hybrid):
    return EvastarV5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        **{**K3, **hybrid}))


def _layer(chunk=0, seed=0):
    torch.manual_seed(seed)
    m = GatedDeltaAttention(_cfg(chunk_size=chunk))
    with torch.no_grad():   # move off the init so every term matters
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return m


def test_config():
    for bad in ({"per_channel_decay": False}, {"short_conv_kernel": 0},
                {"gate_lower_bound": 0.0}, {"gate_rank": 0}, {"style": "x"}):
        with pytest.raises(ValueError):
            _cfg(**bad)
    m = GatedDeltaAttention(_cfg(gate_rank=8))
    assert m.f_a.out_features == 8 and not hasattr(m, "g_proj")


def _reference(m, x):
    """kimi-k3 KDA layer, written from llama.cpp src/models/kimi-k3.cpp."""
    B, T, _ = x.shape
    H, D, K = m.num_heads, m.head_dim, m.conv_kernel
    eps = m.norm_eps
    w = m.conv_weight.view(3, H * D, K)

    def conv(proj, wi):
        y = proj(x).transpose(1, 2)                       # [B, C, T]
        y = F.conv1d(F.pad(y, (K - 1, 0)), wi[:, None, :], groups=wi.shape[0])
        return F.silu(y.transpose(1, 2)).reshape(B, T, H, D)

    def l2(t):
        return t / torch.sqrt((t * t).sum(-1, keepdim=True) + eps)

    q = l2(conv(m.q_proj, w[0])) / D ** 0.5
    k = l2(conv(m.k_proj, w[1]))
    v = conv(m.v_proj, w[2])
    g = m.f_b(m.f_a(x)) + m.dt_bias
    g = g.view(B, T, H, D) * torch.exp(m.A_log)[:, None]
    g = m.gate_lower_bound * torch.sigmoid(g)
    beta = torch.sigmoid(m.beta_proj(x))                  # [B, T, H]
    S = x.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[:, t])[..., None]
        sk = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        d = (v[:, t] - sk) * beta[:, t, :, None]
        S = S + k[:, t, :, :, None] * d[:, :, None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    o = torch.stack(outs, 1)                              # [B, T, H, D]
    o = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + eps) * m.o_norm
    o = o * torch.sigmoid(m.out_gate(x)).view(B, T, H, D)
    return m.o_proj(o.reshape(B, T, H * D)), S


@pytest.mark.parametrize("chunk", [0, 8])
def test_matches_llamacpp_reference(chunk):
    m = _layer(chunk)
    x = torch.randn(2, 37, m.hidden_size)
    with torch.no_grad():
        y, present = m(x, use_cache=True)
        ref, S = _reference(m, x)
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()
    assert torch.allclose(present[0], S, atol=1e-5)


def _fwd(m, x, **kw):
    x = x.clone().requires_grad_(True)
    y, present = m(x, use_cache=True, **kw)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    return y.detach(), present, x.grad, [p.grad for p in m.parameters()]


@pytest.mark.parametrize("layout", ["plain", "mask", "packed"])
def test_chunked_matches_loop(layout):
    a = _layer(0)
    b = copy.deepcopy(a)
    b.chunk_size = 16
    B, T = 2, 45
    x = torch.randn(B, T, a.hidden_size)
    kw = {}
    if layout == "mask":
        mask = torch.ones(B, T)
        mask[0, :6] = 0
        mask[1, 30:33] = 0
        kw["attention_mask"] = mask
    if layout == "packed":
        kw["position_ids"] = torch.cat([torch.arange(20), torch.arange(25)]).expand(B, T)
    ya, pa, gxa, gpa = _fwd(a, x, **kw)
    yb, pb, gxb, gpb = _fwd(b, x, **kw)
    assert torch.allclose(ya, yb, atol=1e-5)
    for sa, sb in zip(pa, pb):
        assert torch.allclose(sa, sb, atol=1e-5)
    assert torch.allclose(gxa, gxb, atol=1e-5)
    for ga, gb in zip(gpa, gpb):
        assert torch.allclose(ga, gb, atol=1e-5)


def test_prefill_then_decode():
    m = _layer(8)
    x = torch.randn(1, 30, m.hidden_size)
    with torch.no_grad():
        full, _ = m(x)
        y, past = m(x[:, :24], use_cache=True)
        steps = [y]
        for t in range(24, 30):
            y, past = m(x[:, t:t + 1], past_key_value=past, use_cache=True)
            steps.append(y)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5)


@pytest.mark.parametrize("chunk,sub", [(7, 16), (32, 5), (128, 16)])
def test_decay_first_chunk_fp64(chunk, sub):
    g = torch.Generator().manual_seed(3)
    B, H, T, D = 2, 3, 90, 8
    q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, D, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
    logd = -5 * torch.rand(B, H, T, D, generator=g, dtype=torch.float64)
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    seg = torch.zeros(B, T, dtype=torch.bool)
    seg[:, 0] = True
    seg[0, 31] = seg[1, 40] = True
    S0 = torch.zeros(B, H, D, D, dtype=torch.float64)
    m = GatedDeltaAttention.__new__(GatedDeltaAttention)
    m.style, m.per_channel_decay = "k3", True
    o1, s1 = GatedDeltaAttention._sequential(m, q, k, v, logd, beta, seg, S0)
    o2, s2 = chunked_delta_rule(q, k, v, logd, beta, seg, S0, chunk, sub=sub,
                                log_decay=True, decay_first=True)
    assert (o1 - o2).abs().max() < 1e-10 and (s1 - s2).abs().max() < 1e-10


def test_initial_decay():
    m = GatedDeltaAttention(_cfg())
    with torch.no_grad():
        logd = m.decay_gate(torch.zeros(1, 2, m.hidden_size))
    assert torch.allclose(logd.exp(), torch.full_like(logd, 0.982), atol=1e-6)


def test_model_level():
    ids = torch.randint(3, 1000, (2, 40))
    outs = []
    for chunk in (0, 16):
        torch.manual_seed(0)
        model = EvastarV5(_cfg(chunk_size=chunk)).eval()
        with torch.no_grad():
            out = model(ids)
        outs.append(out["logits"] if isinstance(out, dict) else out[0])
    assert torch.allclose(outs[0], outs[1], atol=1e-4)
