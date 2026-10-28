"""T103 — roadmap B2-3: ``moe.first_k_dense`` (GLM-5.3 / MiMo), v5.65.

Pins:

1. default 0: every layer is a routed MoE, and the state dict / output
   are identical to an explicit ``first_k_dense=0`` (old checkpoints keep
   loading strictly);
2. k = 2 of 4: layers 0-1 hold a DenseFFN of width intermediate_size
   (no router, no route bias, no expert_load), layers 2-3 a MoE; the
   parameter count changes by exactly the swapped modules;
3. cached decode == full forward with dense layers in the stack;
4. routing statistics exclude dense layers: training forwards fill
   expert_load only in MoE layers, and the global balance loss averages
   over the MoE layers only;
5. composes with the GDA hybrid and with hyper-connections;
6. validation.

Run from repo root: python3 helioslm_v5/tests/test_first_k_dense.py
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, HybridAttentionConfig, MoEConfig)
from helioslm_v5.src.model_v5 import (  # noqa: E402
    DenseFFN, HeliosLMv5, balance_loss)
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402


def _model(k, layers=4, **kw):
    cfg = HeliosLMv5Config(size="lite", num_hidden_layers=layers,
                           moe=MoEConfig(first_k_dense=k), **kw)
    torch.manual_seed(0)
    return HeliosLMv5(cfg), cfg


def test_default_unchanged():
    torch.manual_seed(0)
    base = HeliosLMv5(HeliosLMv5Config(size="lite", num_hidden_layers=4))
    zero, _ = _model(0)
    assert list(base.state_dict()) == list(zero.state_dict())
    assert all(torch.equal(a, b) for a, b in
               zip(base.state_dict().values(), zero.state_dict().values()))
    assert all(isinstance(l.moe, DeviceLimitedMoE) for l in base.layers)
    zero.load_state_dict(base.state_dict(), strict=True)
    ids = torch.randint(3, 1000, (2, 9))
    with torch.no_grad():
        assert torch.equal(base.eval()(ids)[0], zero.eval()(ids)[0])
    print("PASS test_default_unchanged")


def test_layout_and_params():
    model, cfg = _model(2)
    kinds = ["dense" if isinstance(l.moe, DenseFFN) else "moe"
             for l in model.layers]
    assert kinds == ["dense", "dense", "moe", "moe"], kinds
    for l in model.layers[:2]:
        names = [n for n, _ in l.moe.named_parameters()]
        assert names == ["w13.weight", "w2.weight"], names
        assert l.moe.w13.weight.shape == (2 * cfg.intermediate_size,
                                          cfg.hidden_size)
        assert not hasattr(l.moe, "expert_load")
    full, _ = _model(0)
    moe_params = sum(p.numel() for p in full.layers[0].moe.parameters())
    dense_params = sum(p.numel() for p in model.layers[0].moe.parameters())
    n_full = sum(p.numel() for p in full.parameters())
    n_k = sum(p.numel() for p in model.parameters())
    assert n_k == n_full - 2 * moe_params + 2 * dense_params
    assert any("layers.0.moe.w13.weight" == k for k in model.state_dict())
    print("PASS test_layout_and_params")


def test_decode_equals_full():
    model, cfg = _model(2)
    model.eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 14))
    with torch.no_grad():
        ref = model(ids)[0]
        lg, _, past = model(ids[:, :7], use_cache=True)
        outs = [lg]
        for t in range(7, 14):
            lg, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
            outs.append(lg)
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
    print("PASS test_decode_equals_full")


def test_routing_stats_exclude_dense():
    cfg = HeliosLMv5Config(size="lite", num_hidden_layers=4,
                           moe=MoEConfig(first_k_dense=2,
                                         balance_loss_scope="global"))
    torch.manual_seed(0)
    model = HeliosLMv5(cfg).train()
    ids = torch.randint(3, 1000, (2, 10))
    model(ids)
    moes = [l.moe for l in model.layers if isinstance(l.moe, DeviceLimitedMoE)]
    assert len(moes) == 2
    assert all(m.expert_load.sum() > 0 for m in moes)
    want = 0.0
    for m in moes:
        counts, p_sum, n = m._balance_records[0]
        want += (m.num_experts * ((counts / (n * m.top_k)) * (p_sum / n)).sum()).item()
    got = balance_loss(model).item()
    assert abs(got - want / 2) < 1e-6, (got, want / 2)
    print("PASS test_routing_stats_exclude_dense")


def test_composes_with_hybrid_and_hc():
    for kw in (dict(hybrid_attention=HybridAttentionConfig(
                    enabled=True, full_attention_every=2)),
               dict(use_hyper_connections=True)):
        model, cfg = _model(1, **kw)
        ids = torch.randint(3, cfg.vocab_size, (1, 8))
        out = model(ids)[0]
        assert torch.isfinite(out).all()
        out.logsumexp(-1).mean().backward()
        assert model.layers[0].moe.w13.weight.grad is not None
    print("PASS test_composes_with_hybrid_and_hc")


def test_validation():
    for k, needle in ((-1, "first_k_dense"), (True, "first_k_dense"),
                      (5, "exceeds num_hidden_layers")):
        try:
            HeliosLMv5Config(size="lite", num_hidden_layers=4,
                             moe=MoEConfig(first_k_dense=k))
        except ValueError as e:
            assert needle in str(e), (k, e)
        else:
            raise AssertionError(k)
    print("PASS test_validation")


if __name__ == "__main__":
    test_default_unchanged()
    test_layout_and_params()
    test_decode_equals_full()
    test_routing_stats_exclude_dense()
    test_composes_with_hybrid_and_hc()
    test_validation()
