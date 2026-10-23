"""T128 — LB8 per-layer embeddings (Gemma 3n / Gemma 4 E2B), v5.125.

Pins:

1. off by default: no ``ple`` module, no new state-dict keys; bad values
   and the combination with hyper-connections are refused (the latter by
   its OPTION_RULES rule);
2. at init a PLE model computes exactly the model without it (same
   weights loaded, the zero-initialised output projections add nothing);
3. it trains: the output projections get gradient at once, the table
   and gates after one step, and the loss falls;
4. with non-zero weights the cached decode equals the full forward, and
   the per-layer input depends on the token (two tokens differ);
5. count_params puts the per-layer table under "embedding", so
   non_embedding grows only by the gates, projections and norms.

Run from repo root: python3 helioslm_v5/tests/test_ple.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig,
    OptionConflictError)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training.pretrain import count_params  # noqa: E402


def _cfg(**kw):
    return HeliosLMv5Config(size="lite", num_hidden_layers=3, vocab_size=40,
                            hidden_size=64, intermediate_size=128,
                            hybrid_attention=HybridAttentionConfig(
                                enabled=False),
                            mtp=MTPConfig(enabled=False),
                            moe=MoEConfig(num_experts=2,
                                          num_activated_experts=1,
                                          expert_hidden_size=64),
                            use_attention_residuals=False, **kw)


IDS = torch.randint(3, 40, (2, 12), generator=torch.Generator().manual_seed(0))


def test_off_and_validation():
    torch.manual_seed(0)
    m = HeliosLMv5(_cfg())
    assert m.ple is None and not any(k.startswith("ple.")
                                     for k in m.state_dict())
    for bad in (-1, True, 2.0):
        try:
            _cfg(per_layer_embed_dim=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")
    try:
        _cfg(per_layer_embed_dim=8, use_hyper_connections=True)
    except OptionConflictError as e:
        assert e.rule == "per_layer_embeddings_exclude_hyper_connections"
    else:
        raise AssertionError("PLE + HC accepted")
    print("PASS test_off_and_validation")


def test_identity_at_init():
    torch.manual_seed(0)
    off = HeliosLMv5(_cfg()).eval()
    torch.manual_seed(1)
    on = HeliosLMv5(_cfg(per_layer_embed_dim=8)).eval()
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("ple.")
                                              for k in missing)
    with torch.no_grad():
        assert torch.equal(on(IDS)[0], off(IDS)[0])
    print("PASS test_identity_at_init")


def test_trains():
    torch.manual_seed(0)
    m = HeliosLMv5(_cfg(per_layer_embed_dim=8))
    x, y = IDS[:, :-1], IDS[:, 1:]

    def loss():
        lg = m(x)[0]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))

    loss().backward()
    assert m.ple.up[0].weight.grad.abs().sum() > 0
    assert m.ple.embed.weight.grad.abs().sum() == 0   # behind a zero up
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        opt.zero_grad()
        lv = loss()
        first = lv.item() if first is None else first
        lv.backward()
        if m.ple.embed.weight.grad is not None and _ > 0:
            assert m.ple.embed.weight.grad.abs().sum() > 0
        opt.step()
    assert loss().item() < first - 0.5, (first, loss().item())
    print("PASS test_trains")


def test_decode_and_token_dependence():
    torch.manual_seed(0)
    m = HeliosLMv5(_cfg(per_layer_embed_dim=8)).eval()
    with torch.no_grad():
        for up in m.ple.up:
            up.weight.normal_(0, 0.5)
        full = m(IDS)[0]
        lg, _, past = m(IDS[:, :5], use_cache=True)
        outs = [lg]
        for t in range(5, IDS.shape[1]):
            lg, _, past = m(IDS[:, t:t + 1], past_key_values=past,
                            use_cache=True)
            outs.append(lg)
        assert (torch.cat(outs, 1) - full).abs().max() < 1e-4
        h0 = m.embed_tokens(torch.tensor([[5, 6]]))
        x = m.ple.inputs(torch.tensor([[5, 6]]), h0)
        assert x.shape == (1, 2, 3, 8) and not torch.allclose(x[0, 0],
                                                              x[0, 1])
    print("PASS test_decode_and_token_dependence")


def test_count_params():
    torch.manual_seed(0)
    a = count_params(HeliosLMv5(_cfg()))
    b = count_params(HeliosLMv5(_cfg(per_layer_embed_dim=8)))
    table = 40 * 3 * 8
    assert b["embedding"] - a["embedding"] == table
    # model_proj 64*24 + proj_norm 8 + per layer (gate 64*8 + up 8*64 +
    # post_norm 64)
    extra = 64 * 24 + 8 + 3 * (64 * 8 + 8 * 64 + 64)
    assert b["non_embedding"] - a["non_embedding"] == extra, (a, b)
    print("PASS test_count_params")


if __name__ == "__main__":
    test_off_and_validation()
    test_identity_at_init()
    test_trains()
    test_decode_and_token_dependence()
    test_count_params()
