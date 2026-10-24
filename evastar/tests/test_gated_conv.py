"""T129 — LB9 gated short-convolution layers (LFM2), v5.126.

Pins:

1. off by default (no GatedShortConv anywhere); bad layer lists, a kernel
   below 2 and a list that removes every MLA layer are refused;
2. the mixer equals a reference built from F.conv1d (depthwise, causal
   left padding) with both gates;
3. a model with conv layers: cached decode (prefill 5, then one token at
   a time) equals the full forward; the cache entry is a fixed-size
   StateTensor that past_seq_len skips;
4. packed sequences: with a position restart the second document's
   outputs equal running it alone; padding (mask 0) writes nothing;
5. the pilot ``lfm_conv`` arch builds and keeps MLA on every third layer
   and the last.

Run from repo root: python3 evastar/tests/test_gated_conv.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import (  # noqa: E402
    EvastarV5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from evastar.src.attention.gated_conv import GatedShortConv  # noqa: E402
from evastar.src.attention.linear_attention import (  # noqa: E402
    is_recurrent_state, past_seq_len)
from evastar.src.attention.mla import MLA  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.training.pretrain import (  # noqa: E402
    count_params, pilot_config)


def _cfg(**kw):
    return EvastarV5Config(size="lite", num_hidden_layers=4, vocab_size=40,
                            hidden_size=64, intermediate_size=128,
                            hybrid_attention=HybridAttentionConfig(
                                enabled=False),
                            mtp=MTPConfig(enabled=False),
                            moe=MoEConfig(num_experts=2,
                                          num_activated_experts=1,
                                          expert_hidden_size=64),
                            use_attention_residuals=False, **kw)


def test_off_and_validation():
    m = EvastarV5(_cfg())
    assert not any(isinstance(l.attention, GatedShortConv) for l in m.layers)
    for kw in (dict(gated_conv_layers=(1, 1)), dict(gated_conv_layers=(4,)),
               dict(gated_conv_layers=(True,)),
               dict(gated_conv_layers=(1,), gated_conv_kernel=1),
               dict(gated_conv_layers=(0, 1, 2, 3))):
        try:
            _cfg(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kw}")
    c = _cfg(gated_conv_layers=[0, 2])
    assert c.gated_conv_layers == (0, 2)
    print("PASS test_off_and_validation")


def test_matches_conv1d():
    torch.manual_seed(0)
    cfg = _cfg(gated_conv_layers=(1,), gated_conv_kernel=3)
    mod = GatedShortConv(cfg)
    x = torch.randn(2, 7, 64)
    out, _ = mod(x)
    b, c, v = mod.in_proj(x).chunk(3, -1)
    u = (b * v).transpose(1, 2)                     # [B, d, T]
    y = F.conv1d(F.pad(u, (2, 0)), mod.conv_weight.unsqueeze(1), groups=64)
    ref = mod.out_proj(c * y.transpose(1, 2))
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    print("PASS test_matches_conv1d")


IDS = torch.randint(3, 40, (2, 11), generator=torch.Generator().manual_seed(0))


def test_cached_decode():
    torch.manual_seed(0)
    m = EvastarV5(_cfg(gated_conv_layers=(0, 2))).eval()
    assert isinstance(m.layers[0].attention, GatedShortConv)
    assert isinstance(m.layers[1].attention, MLA)
    with torch.no_grad():
        full = m(IDS)[0]
        lg, _, past = m(IDS[:, :5], use_cache=True)
        assert is_recurrent_state(past[0][0])
        assert past[0][0].shape == (2, 2, 64)
        assert past_seq_len(past) == 5
        outs = [lg]
        for t in range(5, IDS.shape[1]):
            lg, _, past = m(IDS[:, t:t + 1], past_key_values=past,
                            use_cache=True)
            outs.append(lg)
    err = (torch.cat(outs, 1) - full).abs().max().item()
    assert err < 1e-4, err
    print("PASS test_cached_decode")


def test_packed_and_padding():
    torch.manual_seed(0)
    cfg = _cfg(gated_conv_layers=(1,))
    mod = GatedShortConv(cfg)
    a, b = torch.randn(1, 4, 64), torch.randn(1, 5, 64)
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4]])
    packed, _ = mod(torch.cat([a, b], 1), position_ids=pos)
    alone, _ = mod(b, position_ids=torch.arange(5).unsqueeze(0))
    assert torch.allclose(packed[:, 4:], alone, atol=1e-6)
    # without positions the taps would cross the boundary
    leaky, _ = mod(torch.cat([a, b], 1))
    assert not torch.allclose(leaky[:, 4:], alone, atol=1e-4)
    # left padding writes nothing into the convolution
    pad = torch.randn(1, 3, 64)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    padded, _ = mod(torch.cat([pad, b], 1), attention_mask=mask)
    assert torch.allclose(padded[:, 3:], mod(b)[0], atol=1e-6)
    print("PASS test_packed_and_padding")


def test_pilot_arch():
    cfg = pilot_config("lfm_conv", vocab_size=96, hidden_size=64,
                       num_layers=8, intermediate_size=128, max_seq=64)
    assert cfg.gated_conv_layers == (0, 1, 3, 4, 6)
    m = EvastarV5(cfg).eval()
    kinds = ["conv" if isinstance(l.attention, GatedShortConv) else "mla"
             for l in m.layers]
    assert kinds == ["conv", "conv", "mla", "conv", "conv", "mla", "conv",
                     "mla"], kinds
    with torch.no_grad():
        out = m(torch.randint(0, 96, (2, 16)))[0]
    assert torch.isfinite(out).all()
    assert count_params(m)["total"] > 0
    print("PASS test_pilot_arch")


if __name__ == "__main__":
    test_off_and_validation()
    test_matches_conv1d()
    test_cached_decode()
    test_packed_and_padding()
    test_pilot_arch()
