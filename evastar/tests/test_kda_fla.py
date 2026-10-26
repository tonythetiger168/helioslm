"""T153 — ``hybrid_attention.kernel="fla"``: the k3 KDA chunked forward
through flash-linear-attention's ``chunk_kda`` on CUDA, v5.160.

FLA's Triton kernels need CUDA, so these CPU tests put a stand-in
``fla.ops.kda.chunk_kda`` (FLA's argument layout: [B, T, H, D], log
gate, beta, ``scale``, ``initial_state``, ``output_final_state``,
computed with ``chunked_delta_rule``) in place and pin:

1. config validation and the pilot_config flag;
2. the layer hands chunk_kda the FLA layout and gets the same outputs,
   carried state and gradients as the torch kernel;
3. the torch kernel is used off CUDA, with packed documents, and (with a
   warning) when fla does not import.
The GPU comparison against real FLA 0.5.2 (rel. error 4.6e-3 in bf16) is
in the v5.159 CHANGELOG entry.

Run from repo root: python3 -m pytest evastar/tests/test_kda_fla.py
"""
import copy
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evastar.configs.config_v5 import (  # noqa: E402
    EvastarV5Config, HybridAttentionConfig)
from evastar.src.attention import linear_attention as la  # noqa: E402
from evastar.src.training.pretrain import pilot_config  # noqa: E402

K3 = dict(enabled=True, style="k3", per_channel_decay=True, short_conv_kernel=4,
          chunk_size=16)


def _fake_chunk_kda(calls):
    def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None,
                  output_final_state=False, **kw):
        B, T, H, D = q.shape
        assert k.shape == (B, T, H, D) and g.shape == (B, T, H, D)
        assert beta.shape == (B, T, H) and scale == 1.0 and output_final_state
        assert initial_state.shape == (B, H, D, v.shape[-1])
        calls.append(q.dtype)
        tr = lambda x: x.transpose(1, 2)            # noqa: E731
        o, s = la.chunked_delta_rule(tr(q), tr(k), tr(v), tr(g), tr(beta), None,
                                     initial_state, 16, log_decay=True,
                                     decay_first=True)
        return o.transpose(1, 2), s
    return chunk_kda


@pytest.fixture
def fake_fla(monkeypatch):
    calls = []
    mod = types.ModuleType("fla.ops.kda")
    mod.chunk_kda = _fake_chunk_kda(calls)
    for name in ("fla", "fla.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", mod)
    monkeypatch.setattr(la, "_FLA", [])
    monkeypatch.setattr(la, "_fla_device", lambda x: True)
    return calls


def _pair():
    torch.manual_seed(0)
    a = la.GatedDeltaAttention(EvastarV5Config(
        size="lite", hybrid_attention=HybridAttentionConfig(**K3)))
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.1 * torch.randn_like(p))
    b = copy.deepcopy(a)
    b.kernel = "fla"
    return a, b


def _fwd(m, x, **kw):
    x = x.clone().requires_grad_(True)
    y, present = m(x, use_cache=True, **kw)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    return y.detach(), present, x.grad, [p.grad for p in m.parameters()]


def test_config():
    assert HybridAttentionConfig().kernel == "torch"
    with pytest.raises(ValueError, match="kernel must be"):
        EvastarV5Config(size="lite", hybrid_attention=HybridAttentionConfig(
            **{**K3, "kernel": "cuda"}))
    with pytest.raises(ValueError, match="needs style='k3'"):
        EvastarV5Config(size="lite", hybrid_attention=HybridAttentionConfig(
            enabled=True, per_channel_decay=True, kernel="fla"))
    cfg = pilot_config("k3_llamacpp", vocab_size=512, hidden_size=128, num_layers=4,
                       intermediate_size=256, kda_kernel="fla")
    assert cfg.hybrid_attention.kernel == "fla"


def test_fla_matches_torch(fake_fla):
    a, b = _pair()
    x = torch.randn(2, 45, a.hidden_size, generator=torch.Generator().manual_seed(1))
    mask = torch.ones(2, 45)
    mask[1, 30:33] = 0
    ya, pa, gxa, gpa = _fwd(a, x, attention_mask=mask)
    yb, pb, gxb, gpb = _fwd(b, x, attention_mask=mask)
    assert fake_fla == [torch.float32]
    assert torch.allclose(ya, yb, atol=1e-6)
    for sa, sb in zip(pa, pb):
        assert torch.allclose(sa, sb, atol=1e-6)
    assert torch.allclose(gxa, gxb, atol=1e-6)
    for ga, gb in zip(gpa, gpb):
        assert torch.allclose(ga, gb, atol=1e-6)


def test_fla_runs_in_autocast_dtype(fake_fla):
    _, b = _pair()
    x = torch.randn(1, 20, b.hidden_size)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        b(x)
    assert fake_fla == [torch.bfloat16]


def test_torch_kernel_for_packed_documents(fake_fla):
    a, b = _pair()
    x = torch.randn(2, 45, a.hidden_size)
    pos = torch.cat([torch.arange(20), torch.arange(25)]).expand(2, 45)
    with torch.no_grad():
        assert torch.equal(a(x, position_ids=pos)[0], b(x, position_ids=pos)[0])
    assert fake_fla == []


def test_torch_kernel_off_cuda_and_without_fla(monkeypatch):
    a, b = _pair()
    x = torch.randn(1, 20, a.hidden_size)
    with torch.no_grad():                       # CPU: FLA is never tried
        assert torch.equal(a(x)[0], b(x)[0])
    monkeypatch.setattr(la, "_FLA", [])
    monkeypatch.setattr(la, "_fla_device", lambda x: True)
    monkeypatch.setitem(sys.modules, "fla", None)          # import fails
    with pytest.warns(UserWarning, match="fla unavailable"), torch.no_grad():
        assert torch.equal(a(x)[0], b(x)[0])
