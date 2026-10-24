"""HeliosLM v5 test cases — AWQ / GPTQ quantization.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch
import torch.nn as nn

from evastar.tests.case_common import (_expect_raises, _pass)


# ----------------------------------------------------------------------
# Quantization: AWQ/GPTQ/FP8 + QuantizationManager dispatch
# ----------------------------------------------------------------------
def test_quantization():
    from evastar.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, FP8Linear, QuantizationManager)

    torch.manual_seed(114)
    lin = nn.Linear(64, 48, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(48, 64) * 0.05)
        lin.bias.copy_(torch.randn(48) * 0.01)
    W_ref = lin.weight.detach()
    x = torch.randn(7, 64)

    # AWQ: 4-bit reconstruction error near the theoretical RTN floor (<10%).
    awq = AWQLinear.from_linear(lin, group_size=32)
    rel_awq = ((awq._dequantize() - W_ref).norm() / W_ref.norm()).item()
    assert rel_awq < 0.10, f"AWQ relative weight error {rel_awq:.3%} >= 10%"
    assert awq.bias is not None and torch.equal(awq.bias, lin.bias), \
        "AWQ must preserve the bias"
    out_awq = awq(x)
    rel_out = ((out_awq - lin(x)).norm() / lin(x).norm().clamp(min=1e-12)).item()
    assert out_awq.shape == (7, 48) and rel_out < 0.15

    # GPTQ without calibration data takes the documented RTN fallback
    # (the true Hessian-compensated path is covered by
    # test_gptq_calibration below): deterministic forward (weights rebuilt
    # from packed buffers), bias preserved, reconstruction error < 10%.
    gptq = GPTQLinear.from_linear(lin, group_size=32)  # no calibration -> RTN
    o1, o2 = gptq(x), gptq(x)
    assert torch.equal(o1, o2), "GPTQ forward is not deterministic"
    rel_gptq = ((gptq._dequantize() - W_ref).norm() / W_ref.norm()).item()
    assert rel_gptq < 0.10, f"GPTQ relative weight error {rel_gptq:.3%} >= 10%"
    assert gptq.bias is not None and torch.equal(gptq.bias, lin.bias)

    # M-Q1 regression (v5.3): fp16/bf16 inputs must not crash — the bias
    # is cast to the input dtype, like AWQ/FP8 already did.
    gptq_c = GPTQLinear.from_linear(lin, group_size=32,
                                    calibration_data=torch.randn(128, 64))
    for dt in (torch.float16, torch.bfloat16):
        for m, tag in ((gptq, "rtn"), (gptq_c, "gptq")):
            out_dt = m(x.to(dt))
            assert out_dt.dtype == dt and torch.isfinite(out_dt).all(), \
                f"GPTQ({tag}) forward broke for {dt}"

    # QuantizationManager: dispatch + unknown method.
    tiny = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 16))
    QuantizationManager(method="awq").quantize_model(tiny, group_size=16)
    assert isinstance(tiny[0], AWQLinear) and isinstance(tiny[2], AWQLinear)
    y = tiny(torch.randn(3, 32))
    assert torch.isfinite(y).all()
    _expect_raises(ValueError,
                   lambda: QuantizationManager(method="mx4").quantize_model(tiny),
                   "unknown quantization method")

    # FP8: real float8 storage when the torch build supports it, otherwise a
    # documented NotImplementedError.
    if FP8Linear is not None:
        f8 = FP8Linear.from_linear(lin)
        assert f8.weight_fp8.dtype == torch.float8_e4m3fn, \
            f"FP8 weights must be stored as float8_e4m3fn, got {f8.weight_fp8.dtype}"
        out_f8 = f8(x)
        assert out_f8.shape == (7, 48) and torch.isfinite(out_f8).all()
        fp8_note = "native float8_e4m3fn storage, forward OK"
    else:
        _expect_raises(NotImplementedError,
                       lambda: QuantizationManager(method="fp8").quantize_model(tiny),
                       "fp8 without float8 support")
        fp8_note = "no float8 in this torch -> NotImplementedError (as documented)"

    _pass("test_quantization",
          f"AWQ rel-err={rel_awq:.2%}, GPTQ rel-err={rel_gptq:.2%} & "
          f"deterministic, bias preserved, unknown method raises, fp8: {fp8_note}")


# ----------------------------------------------------------------------
# v5.2: true GPTQ Hessian error compensation beats RTN on layer OUTPUT
# ----------------------------------------------------------------------
def test_gptq_calibration():
    from evastar.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(120)
    lin = nn.Linear(256, 256, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(256, 256) * 0.05)
        lin.bias.copy_(torch.randn(256) * 0.01)

    # Correlated calibration activations (low-rank + small noise): the
    # Hessian H = X^T X is strongly anisotropic, which is exactly where
    # GPTQ's OBS error compensation wins over plain RTN.
    rank = 48
    basis = torch.randn(rank, 256)

    def make_data(n, seed):
        g = torch.Generator().manual_seed(seed)
        coeff = torch.randn(n, rank, generator=g)
        return coeff @ basis + 0.05 * torch.randn(n, 256, generator=g)

    calib = make_data(512, seed=700)

    rtn = GPTQLinear.from_linear(lin, group_size=128, bits=4)  # no calibration
    gptq = GPTQLinear.from_linear(lin, group_size=128, bits=4,
                                  calibration_data=calib)

    # Held-out data from the same distribution: GPTQ must reconstruct the
    # layer OUTPUT significantly better than RTN (measured ratio ~0.33;
    # threshold 0.80 leaves ample margin).
    eval_x = make_data(1024, seed=701)
    ref = lin(eval_x)
    err_rtn = ((rtn(eval_x) - ref).norm() / ref.norm()).item()
    err_gptq = ((gptq(eval_x) - ref).norm() / ref.norm()).item()
    assert err_gptq < 0.8 * err_rtn, \
        f"GPTQ output rel-err {err_gptq:.4f} not < 80% of RTN {err_rtn:.4f}"

    # Deterministic forward in both modes (weights rebuilt from packed
    # buffers every call).
    assert torch.equal(gptq(eval_x), gptq(eval_x)), "GPTQ forward not deterministic"
    assert torch.equal(rtn(eval_x), rtn(eval_x)), "RTN forward not deterministic"

    # Calibration data whose last dim != in_features raises ValueError.
    _expect_raises(ValueError,
                   lambda: GPTQLinear.from_linear(
                       lin, group_size=128,
                       calibration_data=torch.randn(64, 128)),
                   "calibration_data with wrong last dim")
    _pass("test_gptq_calibration",
          f"output rel-err GPTQ {err_gptq:.2%} vs RTN {err_rtn:.2%} "
          f"(ratio {err_gptq / err_rtn:.2f} < 0.80), deterministic, "
          f"bad calibration shape raises")


# ----------------------------------------------------------------------
# v5.3 (M-Q3): AWQ calibration path is never significantly worse than
# plain RTN, including under peaky/outlier activations
# ----------------------------------------------------------------------
def test_awq_calibration():
    from evastar.src.quantization.standard_quant import AWQLinear

    g = torch.Generator().manual_seed(42)
    lin = nn.Linear(128, 96, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(96, 128, generator=g) * 0.05)
        lin.bias.copy_(torch.randn(96, generator=g) * 0.01)

    def make(kind, n, gen):
        if kind == "gauss":
            return torch.randn(n, 128, generator=gen)
        if kind == "lognorm2":
            return (torch.exp(2.0 * torch.randn(n, 128, generator=gen))
                    * torch.where(torch.rand(n, 128, generator=gen) > 0.5,
                                  1., -1.))
        if kind == "peaky":  # a few outlier channels with ~100x magnitude
            xx = torch.randn(n, 128, generator=gen)
            idx = torch.randperm(128, generator=gen)[:5]
            xx[:, idx] *= 100.0
            return xx
        raise AssertionError(kind)

    def rel_err(mod, xx):
        ref = lin(xx)
        return ((mod(xx) - ref).norm() / ref.norm().clamp(min=1e-12)).item()

    detail = []
    for kind in ("gauss", "lognorm2", "peaky"):
        calib = make(kind, 512, torch.Generator().manual_seed(42))
        eval_x = make(kind, 1024, torch.Generator().manual_seed(43))
        rtn = AWQLinear.from_linear(lin, group_size=32)
        cal = AWQLinear.from_linear(lin, group_size=32, activations=calib)
        assert cal.act_scale is not None, "calibrated AWQ must store act_scale"
        e_rtn, e_cal = rel_err(rtn, eval_x), rel_err(cal, eval_x)
        detail.append(f"{kind} {e_cal:.2%}/RTN {e_rtn:.2%}")
        assert e_cal <= 1.3 * e_rtn + 1e-3, \
            f"AWQ calibrated path on {kind}: output rel-err {e_cal:.4f} " \
            f"significantly worse than RTN {e_rtn:.4f}"
    _pass("test_awq_calibration", "; ".join(detail) + " (all <= 1.3x RTN)")


# ----------------------------------------------------------------------
# v5.3: quantized .weight property == forward's effective weight matrix
# ----------------------------------------------------------------------
def test_quant_weight_property():
    from evastar.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, FP8Linear)

    torch.manual_seed(131)
    lin = nn.Linear(48, 40, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(40, 48) * 0.05)
        lin.bias.copy_(torch.randn(40) * 0.01)
    x = torch.randn(6, 48)
    calib = torch.randn(256, 48)

    mods = [
        ("awq-rtn", AWQLinear.from_linear(lin, group_size=16)),
        ("awq-calib", AWQLinear.from_linear(lin, group_size=16,
                                            activations=calib)),
        ("gptq-rtn", GPTQLinear.from_linear(lin, group_size=16)),
        ("gptq-calib", GPTQLinear.from_linear(lin, group_size=16,
                                              calibration_data=calib)),
    ]
    if FP8Linear is not None:
        mods.append(("fp8", FP8Linear.from_linear(lin)))

    for tag, m in mods:
        w = m.weight
        assert w.shape == (lin.out_features, lin.in_features), \
            f"{tag}: .weight shape {w.shape}"
        assert torch.equal(m.weight, m.weight), \
            f"{tag}: .weight is not deterministic across accesses"
        ref = nn.functional.linear(x, w.to(x.dtype), m.bias.to(x.dtype))
        d = (m(x) - ref).abs().max().item()
        assert d < 1e-5, \
            f"{tag}: forward disagrees with .weight property ({d:.3e})"
    _pass("test_quant_weight_property",
          f"{len(mods)} module variants: .weight == forward effective weight")


# ----------------------------------------------------------------------
# v5.4 (LOW): quantization input validation fails loudly
# ----------------------------------------------------------------------
def test_quant_input_validation():
    import warnings as _warnings
    from evastar.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, QuantizationManager)

    torch.manual_seed(136)
    lin = nn.Linear(16, 8)

    # AWQ calibration activations whose last dim != in_features must raise
    # ValueError (aligned with the GPTQ path) instead of silently
    # reshape-reinterpreting the data.
    _expect_raises(ValueError,
                   lambda: AWQLinear.from_linear(
                       lin, activations=torch.randn(10, 32)),
                   "AWQ activations with wrong last dim")
    # bits != 4 raises ValueError (no bare `assert`) on both packers.
    _expect_raises(ValueError,
                   lambda: AWQLinear.from_linear(lin, bits=8),
                   "AWQ bits=8")
    _expect_raises(ValueError,
                   lambda: GPTQLinear.from_linear(lin, bits=8),
                   "GPTQ bits=8")

    # quantize_model on a bare nn.Linear cannot replace it in place; it must
    # warn and leave the layer unquantized (previously a silent skip).
    bare = nn.Linear(16, 8)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        QuantizationManager(method="awq").quantize_model(bare, group_size=8)
    assert isinstance(bare, nn.Linear), \
        "bare nn.Linear should be left unquantized"
    assert any("bare nn.Linear" in str(w.message) for w in caught), \
        f"expected a bare-nn.Linear skip warning, got {[str(w.message) for w in caught]}"
    _pass("test_quant_input_validation",
          "AWQ bad calibration shape raises, bits!=4 raises (AWQ+GPTQ), "
          "bare nn.Linear skip warns")


# ----------------------------------------------------------------------
# v5.4: GPTQ odd in/out dims pack/unpack exactly (forward == .weight)
# ----------------------------------------------------------------------
def test_gptq_odd_dims():
    from evastar.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(137)
    # Odd in AND out features stress the pad-to-8 packing of qweight (rows)
    # and qzeros (cols).
    lin = nn.Linear(35, 27, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(27, 35) * 0.05)
        lin.bias.copy_(torch.randn(27) * 0.01)
    x = torch.randn(5, 35)

    for tag, mod in (
        ("rtn", GPTQLinear.from_linear(lin, group_size=16)),
        ("gptq", GPTQLinear.from_linear(lin, group_size=16,
                                        calibration_data=torch.randn(128, 35))),
    ):
        w = mod.weight
        assert w.shape == (27, 35), f"{tag}: .weight shape {w.shape}"
        out = mod(x)
        assert out.shape == (5, 27) and torch.isfinite(out).all()
        ref = nn.functional.linear(x, w.to(x.dtype), mod.bias.to(x.dtype))
        d = (out - ref).abs().max().item()
        assert d < 1e-5, \
            f"{tag}: forward disagrees with .weight on odd dims ({d:.3e})"
        rel = ((w - lin.weight).norm() / lin.weight.norm()).item()
        assert rel < 0.15, f"{tag}: odd-dim reconstruction error {rel:.3%}"
    _pass("test_gptq_odd_dims",
          "35x27 (odd in+out): forward == .weight property, RTN + calibrated")


def test_gptq_act_order():
    from evastar.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(801)
    in_f, out_f, group_size = 32, 16, 16
    # Heterogeneous column activation scales — the regime act-order is
    # designed for (high-variance columns are quantized first so the
    # still-dense low-variance columns can absorb their error).
    col_scale = torch.logspace(-2, 2, in_f)
    X = torch.randn(512, in_f) * col_scale
    W_lin = nn.Linear(in_f, out_f, bias=False)

    def out_err(mod):
        with torch.no_grad():
            ref = X @ W_lin.weight.t()
            got = mod(X)
        return ((got - ref).norm() / ref.norm()).item()

    q_rtn = GPTQLinear.from_linear(W_lin, group_size=group_size)
    q_plain = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                     calibration_data=X, act_order=False)
    q_act = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                   calibration_data=X, act_order=True)
    e_rtn, e_plain, e_act = out_err(q_rtn), out_err(q_plain), out_err(q_act)
    assert e_act <= e_plain * 1.05, (
        f"act-order hurt output error: {e_act:.4f} vs plain {e_plain:.4f}")
    assert e_act < e_rtn, (
        f"act-order GPTQ should beat RTN here: {e_act:.4f} vs {e_rtn:.4f}")

    # The act-order packed layout is still exact: g_idx maps every
    # original column to the group it was quantized in, and forward ==
    # matmul with the .weight property.
    assert q_act.g_idx.shape == (in_f,) and q_act.g_idx.max().item() == \
        (in_f + group_size - 1) // group_size - 1, "act-order g_idx broken"
    # v5.11: forward defaults to the FUSED path, whose group-wise matmul
    # reduction order differs from the dense reference by fp32 rounding —
    # bitwise equality with .weight is a property of the dense reference
    # path, so disable fusion for this exactness check.
    q_act.fused = False
    with torch.no_grad():
        w_eff = q_act.weight
        got = torch.nn.functional.linear(X, w_eff)
        ref = q_act(X)
    assert (got - ref).abs().max().item() == 0.0, (
        ".weight property inconsistent with forward after act-order")

    # Deterministic: same inputs -> same packed buffers.
    q_act2 = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                    calibration_data=X, act_order=True)
    assert torch.equal(q_act.qweight, q_act2.qweight) and \
        torch.equal(q_act.qzeros, q_act2.qzeros), "act-order not deterministic"

    # Plain GPTQ is untouched by the new flag (default path unchanged).
    q_plain2 = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                      calibration_data=X)
    assert torch.equal(q_plain.qweight, q_plain2.qweight), (
        "default GPTQ path changed")
    _pass("test_gptq_act_order",
          f"out err RTN {e_rtn:.4f} -> GPTQ {e_plain:.4f} -> act-order "
          f"{e_act:.4f}, layout exact, deterministic")
