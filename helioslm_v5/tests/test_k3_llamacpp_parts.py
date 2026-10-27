"""T139 — Kimi K3 parts as llama.cpp's kimi-k3 graph computes them, v5.142.

Pins:

1. block attention residuals (``attention_residual_style="block"``): the
   layer inputs banked every ``attn_res_block_size`` layers, the stream
   restarting at a banked layer, a softmax mix scored on RMS-normalized
   candidates; zero scores give a uniform mix; a cached prefill + decode
   equals the one-shot forward; the scores get gradients;
2. ``moe.router_input="full"`` routes LatentMoE on the full-width input;
3. ``moe.activation="situ_k3"`` (routed, shared and dense FFNs), equal in
   the blocked and batched dispatch;
4. config validation;
5. ``k3_block_config(llamacpp=True)`` trains (loss falls) and exports to
   kimi-k3 for every size; the default k3_block preset does not.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_k3_llamacpp_parts.py
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config, k3_block_config  # noqa: E402
from helioslm_v5.src.export.llamacpp import llamacpp_arch  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5, block_res_mix  # noqa: E402
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE, k3_situ  # noqa: E402


def _cfg(**kw):
    return HeliosLMv5Config(size="lite", num_hidden_layers=kw.pop("layers", 5), **kw)


BLOCK = dict(use_attention_residuals=True, attention_residual_style="block")


def test_config_validation():
    for bad in ({"attention_residual_style": "x"}, {"attn_res_block_size": 0}):
        with pytest.raises(ValueError):
            _cfg(**bad)
    c = _cfg()
    c.moe.router_input = "x"
    with pytest.raises(ValueError, match="router_input"):
        c._validate()
    c = _cfg()
    c.moe.activation, c.moe.situ_beta = "situ_k3", 0.0
    with pytest.raises(ValueError, match="situ_beta"):
        c._validate()


def test_block_res_mix():
    torch.manual_seed(0)
    a, b, cur = torch.randn(3, 2, 5, 8).unbind(0)
    assert block_res_mix([], cur, torch.randn(8), 1e-6) is cur
    zero = block_res_mix([a, b], cur, torch.zeros(8), 1e-6)
    assert torch.allclose(zero, (a + b + cur) / 3, atol=1e-6)
    w = torch.randn(8)
    xs = torch.stack([a, b, cur], -2)
    p = (F.rms_norm(xs, (8,), eps=1e-6) * w).sum(-1).softmax(-1)
    assert torch.allclose(block_res_mix([a, b], cur, w, 1e-6), (p[..., None] * xs).sum(-2), atol=1e-6)


def _block_model(seed=0, **kw):
    torch.manual_seed(seed)
    m = HeliosLMv5(_cfg(**BLOCK, **kw)).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("res_score"):
                p.normal_(0, 0.3)
    return m


def test_block_layout_and_restart():
    m = _block_model(attn_res_block_size=2)
    assert [layer.res_bank for layer in m.layers] == [True, False, True, False, True]
    assert all(layer.attn_res_gate is None for layer in m.layers)
    assert m.output_res_score.shape == (m.config.hidden_size,)
    ids = torch.randint(3, 1000, (2, 12))
    with torch.no_grad():
        ref = m(ids)[0]
        m.layers[2].res_bank = False          # no restart -> different
        assert not torch.allclose(ref, m(ids)[0], atol=1e-3)


def test_block_cached_decode_matches():
    m = _block_model(attn_res_block_size=2)
    ids = torch.randint(3, 1000, (1, 14))
    with torch.no_grad():
        full = m(ids)[0]
        out, _, past = m(ids[:, :10], use_cache=True)
        steps = [out]
        for t in range(10, 14):
            out, _, past = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            steps.append(out)
    # the score softmax amplifies float rounding (1e-6 without block
    # residuals); a wrong bank or restart is off by O(1e-1)
    assert torch.allclose(full, torch.cat(steps, 1), atol=5e-4)


def test_block_scores_get_gradients():
    m = _block_model().train()
    ids = torch.randint(3, 1000, (2, 10))
    logits = m(ids)[0]
    F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    assert m.output_res_score.grad.abs().sum() > 0
    assert all(layer.attn_res_score.grad.abs().sum() > 0 for layer in m.layers[1:])


def test_router_input_full():
    c = _cfg()
    c.moe.latent_dim, c.moe.router_input = 32, "full"
    moe = DeviceLimitedMoE(c)
    assert moe.router.in_features == c.hidden_size
    assert moe.experts[0].w13.in_features == 32
    c.moe.router_input = "latent"
    assert DeviceLimitedMoE(c).router.in_features == 32


def test_situ_k3_formula_and_dispatch():
    a, b = torch.randn(2, 50) * 5
    assert torch.allclose(k3_situ(a, b, 3.0, 2.0),
                          3 * torch.tanh(a / 3) * torch.sigmoid(a) * 2 * torch.tanh(b / 2))
    assert torch.allclose(k3_situ(a, b, 3.0, 0.0), 3 * torch.tanh(a / 3) * torch.sigmoid(a) * b)
    c = _cfg()
    c.moe.activation, c.moe.situ_beta, c.moe.situ_linear_beta = "situ_k3", 2.0, 1.5
    c.moe.first_k_dense = 1
    torch.manual_seed(0)
    m = HeliosLMv5(c).eval()
    assert m.layers[0].moe.situ == (2.0, 1.5)
    moe = m.layers[1].moe
    x = torch.randn(2, 9, c.hidden_size)
    with torch.no_grad():
        moe.dispatch = "blocked"
        y1 = moe(x)
        moe.dispatch = "batched"
        y2 = moe(x)
        moe.dispatch = "grouped"
        y3 = moe(x)
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(y1, y3, atol=1e-5)


@pytest.mark.parametrize("size", ["lite", "mid", "full"])
def test_k3_block_llamacpp_preset_exports(size):
    c = k3_block_config(size, llamacpp=True)
    assert llamacpp_arch(c) == "kimi-k3"
    assert llamacpp_arch(k3_block_config(size)) is None


def test_k3_block_llamacpp_preset_trains():
    torch.manual_seed(0)
    c = k3_block_config("lite", llamacpp=True)
    c.hybrid_attention.chunk_size = 8
    m = HeliosLMv5(c).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.randint(3, 200, (4, 24))
    losses = []
    for _ in range(12):
        logits = m(ids)[0]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5, losses
