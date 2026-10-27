"""T91 — roadmap Phase B: Kimi-K3-style block preset (v5.57).

Pins:

1. defaults are untouched: no ``o_gate_proj`` exists unless
   ``attention.output_gate`` is on, and with ``first_layer_full=True``
   the interleave equals the v5.5 rule for every tested ``every``;
2. ``first_layer_full=False`` gives the K3 order (KDA, KDA, KDA, MLA)
   and validation refuses a layout with no MLA layer or non-bool flags;
3. Gated MLA math, BITWISE: with shared weights, the gated layer's
   output equals ``o_proj(values * sigmoid(o_gate_proj(x)))`` where
   ``values`` is what the ungated twin feeds o_proj — in both the
   expanded and the absorbed path; and the two paths still agree with
   the gate on (prefill and cached decode);
4. the k3_block preset: layer types, KDA per-channel decay, attention
   residuals, LatentMoE, cached decode == full forward, gradients reach
   the output gates and the residual gates;
5. an ungated checkpoint does not load silently into a gated model.

Run from repo root: python3 helioslm_v5/tests/test_k3_block.py
"""
import copy
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, k3_block_config)
from helioslm_v5.src.attention.linear_attention import (  # noqa: E402
    GatedDeltaAttention)
from helioslm_v5.src.attention.mla import MLA  # noqa: E402
from helioslm_v5.src.model_v5 import (  # noqa: E402
    HeliosLMv5, _is_full_attention_layer)


def _expect_value_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def test_defaults_untouched():
    cfg = HeliosLMv5Config(size="lite")
    assert cfg.attention.output_gate is False
    assert cfg.hybrid_attention.first_layer_full is True
    names = [n for n, _ in HeliosLMv5(cfg).named_parameters()]
    assert not any("o_gate_proj" in n for n in names)
    for every in (2, 4, 8):
        cfg = HeliosLMv5Config(size="lite", num_hidden_layers=16)
        cfg.hybrid_attention.enabled = True
        cfg.hybrid_attention.full_attention_every = every
        got = [_is_full_attention_layer(cfg, i) for i in range(16)]
        want = [i == 0 or i % every == every - 1 for i in range(16)]
        assert got == want, every
    print("PASS test_defaults_untouched")


def test_k3_order_and_validation():
    cfg = HeliosLMv5Config(size="lite", num_hidden_layers=8)
    cfg.hybrid_attention.enabled = True
    cfg.hybrid_attention.first_layer_full = False
    cfg._validate()
    got = ["M" if _is_full_attention_layer(cfg, i) else "L" for i in range(8)]
    assert "".join(got) == "LLLMLLLM", got

    def no_mla():
        c = HeliosLMv5Config(size="lite", num_hidden_layers=3)
        c.hybrid_attention.enabled = True
        c.hybrid_attention.first_layer_full = False
        c._validate()
    _expect_value_error(no_mla, "leaves no MLA layer")

    def bad_flag():
        c = HeliosLMv5Config(size="lite")
        c.hybrid_attention.first_layer_full = "no"
        c._validate()
    _expect_value_error(bad_flag, "first_layer_full must be a bool")

    def bad_gate():
        c = HeliosLMv5Config(size="lite")
        c.attention.output_gate = 1
        c._validate()
    _expect_value_error(bad_gate, "output_gate must be a bool")
    print("PASS test_k3_order_and_validation")


def _twins(use_absorption):
    """(ungated, gated) MLA sharing every common weight."""
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.use_absorption = use_absorption
    torch.manual_seed(7)
    plain = MLA(cfg).eval()
    gcfg = copy.deepcopy(cfg)
    gcfg.attention.output_gate = True
    gated = MLA(gcfg).eval()
    missing, unexpected = gated.load_state_dict(plain.state_dict(),
                                                strict=False)
    assert missing == ["o_gate_proj.weight"] and not unexpected
    return plain, gated


def _o_proj_input(module, x):
    seen = {}
    h = module.o_proj.register_forward_hook(
        lambda m, inp, out: seen.__setitem__("v", inp[0]))
    with torch.no_grad():
        out, _ = module(x)
    h.remove()
    return seen["v"], out


def test_gate_math_bitwise():
    x = torch.randn(2, 9, 256)
    for absorbed in (False, True):
        plain, gated = _twins(absorbed)
        values, _ = _o_proj_input(plain, x)
        with torch.no_grad():
            want = gated.o_proj(values * torch.sigmoid(gated.o_gate_proj(x)))
            got, _ = gated(x)
        assert torch.equal(got, want), f"absorbed={absorbed}"
    print("PASS test_gate_math_bitwise")


def test_gate_paths_agree():
    _, gated = _twins(use_absorption=True)
    x = torch.randn(2, 8, 256)
    steps = [torch.randn(2, 1, 256) for _ in range(3)]

    def run(absorbed):
        gated.use_absorption = absorbed
        with torch.no_grad():
            out, past = gated(x, use_cache=True)
            outs = [out]
            for s in steps:
                o, past = gated(s, past_key_value=past, use_cache=True)
                outs.append(o)
        return torch.cat(outs, 1)

    diff = (run(True) - run(False)).abs().max().item()
    assert diff < 1e-5, diff
    print(f"PASS test_gate_paths_agree (max|diff|={diff:.1e})")


def test_k3_preset():
    torch.manual_seed(0)
    cfg = k3_block_config()
    assert cfg.attention.output_gate and cfg.use_attention_residuals
    assert cfg.hybrid_attention.per_channel_decay
    assert cfg.moe.latent_dim == cfg.hidden_size // 2
    model = HeliosLMv5(cfg).eval()
    kinds = [type(layer.attention) for layer in model.layers]
    assert kinds == [GatedDeltaAttention] * 3 + [MLA], kinds

    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full, _, _ = model(ids)
        logits, _, past = model(ids[:, :8], use_cache=True)
        outs = [logits]
        for t in range(8, 12):
            logits, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                    use_cache=True)
            outs.append(logits)
    diff = (torch.cat(outs, 1) - full).abs().max().item()
    assert diff < 1e-4, diff

    model.train()
    model(ids)[0].logsumexp(-1).mean().backward()
    mla = model.layers[3].attention
    assert mla.o_gate_proj.weight.grad.abs().sum() > 0
    res_gates = [layer.attn_res_gate for layer in model.layers]
    assert all(g is not None for g in res_gates)
    assert any(g.grad is not None and g.grad.abs() > 0 for g in res_gates)
    print(f"PASS test_k3_preset (decode vs full max|diff|={diff:.1e})")


def test_ungated_checkpoint_fails_loudly():
    plain = HeliosLMv5(HeliosLMv5Config(size="lite"))
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.output_gate = True
    gated = HeliosLMv5(cfg)
    try:
        gated.load_state_dict(plain.state_dict())
    except RuntimeError as e:
        assert "o_gate_proj" in str(e)
    else:
        raise AssertionError("ungated checkpoint loaded into a gated model")
    print("PASS test_ungated_checkpoint_fails_loudly")


if __name__ == "__main__":
    test_defaults_untouched()
    test_k3_order_and_validation()
    test_gate_math_bitwise()
    test_gate_paths_agree()
    test_k3_preset()
    test_ungated_checkpoint_fails_loudly()
