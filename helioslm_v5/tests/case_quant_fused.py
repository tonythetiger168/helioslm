"""HeliosLM v5 test cases — MXFP4 / NVFP4 / QAT + fused quant kernels.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch
import torch.nn as nn

from helioslm_v5.tests.case_common import (_expect_raises, _lite_model, _pass)


def test_fused_quant_kernels():
    """v5.11: fused group-wise dequant x matmul == reference dense path."""
    import torch.nn.functional as F  # noqa: F401
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, MXFP4Linear)
    torch.manual_seed(7)
    lin = nn.Linear(256, 128, bias=True)
    x = torch.randn(4, 256)

    for name, mod in [
        ("awq", AWQLinear.from_linear(lin, group_size=64, bits=4)),
        ("gptq", GPTQLinear.from_linear(lin, group_size=64, bits=4,
                                        calibration_data=torch.randn(256, 256))),
        ("mxfp4", MXFP4Linear.from_linear(lin, block_size=32)),
    ]:
        mod.eval()
        with torch.no_grad():
            ref = mod.fused
            mod.fused = False
            y_ref = mod(x)
            mod.fused = True
            y_fused = mod(x)
            mod.fused = ref
        err = (y_ref - y_fused).abs().max().item()
        scale = y_ref.abs().max().item() + 1e-8
        assert err / scale < 1e-4, f"{name}: fused vs reference rel err {err/scale:.2e}"

    # act-order GPTQ (descending g_idx runs) fused path
    gmod = GPTQLinear.from_linear(lin, group_size=64, bits=4, act_order=True,
                                  calibration_data=torch.randn(256, 256))
    gmod.eval()
    with torch.no_grad():
        gmod.fused = False
        y_ref = gmod(x)
        gmod.fused = True
        y_fused = gmod(x)
    assert (y_ref - y_fused).abs().max().item() < 1e-3
    _pass("test_fused_quant_kernels",
          "AWQ/GPTQ/MXFP4 fused == reference (<1e-4 rel); act-order run-sliced")


# ----------------------------------------------------------------------
# v5.6: MXFP4 microscaling FP4 (E2M1 codes + E8M0 per-block power-of-two
# scales, MX block 32) — format validity, error bound, odd dims, manager
# integration, MTP rebind
# ----------------------------------------------------------------------
def test_mxfp4():
    from helioslm_v5.src.quantization.standard_quant import (
        MXFP4Linear, QuantizationManager)

    torch.manual_seed(501)
    # Odd in/out exercises the pad path both directions.
    lin = nn.Linear(35, 27, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(27, 35) * 0.05)
        lin.bias.copy_(torch.randn(27) * 0.01)
    x = torch.randn(5, 35)

    mod = MXFP4Linear.from_linear(lin, block_size=32)
    w = mod.weight
    assert w.shape == (27, 35), f".weight shape {w.shape}"
    # E8M0: every stored scale is a power of two.
    lg = torch.log2(mod.scales)
    assert torch.equal(lg, lg.round()), "scales are not all powers of two"
    # Every dequantized magnitude must be a valid E2M1 code times its scale.
    pad = (-35) % 32
    w_blocks = torch.cat([w, w.new_zeros(27, pad)], 1).view(27, -1, 32)
    ratio = w_blocks / mod.scales.unsqueeze(-1)
    mags = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    for v in ratio.flatten().tolist():
        assert abs(v) in mags or abs(abs(v) - round(abs(v), 6)) < 1e-9 and \
            min(abs(abs(v) - m) for m in mags) < 1e-6, \
            f"dequantized value {v} is not an E2M1 code x scale"
    # Reconstruction error: FP4 is coarse (8 magnitudes) but must be bounded.
    rel = ((w - lin.weight).norm() / lin.weight.norm()).item()
    assert rel < 0.35, f"MXFP4 reconstruction error {rel:.3%} too high"
    # forward == F.linear with .weight; deterministic; bias preserved.
    out = mod(x)
    ref = nn.functional.linear(x, w.to(x.dtype), mod.bias.to(x.dtype))
    d = (out - ref).abs().max().item()
    assert d < 1e-5, f"forward disagrees with .weight ({d:.3e})"
    mod2 = MXFP4Linear.from_linear(lin, block_size=32)
    assert torch.equal(mod.qweight, mod2.qweight) \
        and torch.equal(mod.scales, mod2.scales), "not deterministic"

    # QuantizationManager integration: whole-model quantize + MTP rebind.
    model, config = _lite_model(seed=502)
    QuantizationManager(method="mxfp4").quantize_model(model)
    assert isinstance(model.lm_head, MXFP4Linear), \
        f"lm_head should be MXFP4Linear, got {type(model.lm_head)}"
    for i, mtp in enumerate(model.mtp_modules):
        assert mtp.lm_head is model.lm_head, f"mtp[{i}] lm_head not re-bound"
    ids = torch.randint(3, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _, _ = model(ids)
    assert torch.isfinite(logits).all(), "mxfp4 model produced non-finite logits"
    assert logits.shape == (1, 6, config.vocab_size)
    _pass("test_mxfp4",
          f"E2M1+E8M0 valid (rel-err {rel:.1%}), odd dims exact, deterministic, "
          "manager + MTP rebind OK")


# ----------------------------------------------------------------------
# v5.22: NVFP4-format QAT target (E2M1 x 16 blocks, E4M3 block scales)
# ----------------------------------------------------------------------
def test_nvfp4_qat():
    from helioslm_v5.src.quantization.qat import (
        FakeQuantLinear, _qdq_mxfp4, _qdq_nvfp4, apply_qat)

    torch.manual_seed(322)
    # 1) Accuracy: on Gaussian weights the NVFP4 hierarchy (16-wide blocks,
    #    FP8 block scales) must beat MXFP4 (32-wide blocks, power-of-2
    #    scales) in relative RTN error.
    w = torch.randn(8, 96)
    dq = _qdq_nvfp4(w, 16)
    assert dq.shape == w.shape
    err_nv = (dq - w).norm() / w.norm()
    err_mx = (_qdq_mxfp4(w, 32) - w).norm() / w.norm()
    assert err_nv < 0.08, f"NVFP4 RTN error unexpectedly large: {err_nv:.4f}"
    assert err_nv < err_mx, \
        f"NVFP4 ({err_nv:.4f}) should beat MXFP4 ({err_mx:.4f}) on RTN error"

    # 2) Near-exactness: grid values under a power-of-two block scale with
    #    EVERY block carrying the max grid value (absmax = 3*2^-3 exactly,
    #    so bscale = 2^-4 and every value/eff lands on the E2M1 grid).
    #    Residual error is fp32 rounding of the per-row scale (2^-4/448 is
    #    not a dyadic rational), <= ~6 * 2^-4 * 2^-24 ~ 1e-7.
    g = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    w_grid = g[torch.randint(0, 5, (6, 32))] * (2.0 ** -3)
    w_grid = w_grid * torch.where(torch.rand(6, 32) < 0.5, -1.0, 1.0)
    w_grid[:, 0::16] = 3.0 * (2.0 ** -3)        # per-block absmax anchor
    dq_grid = _qdq_nvfp4(w_grid, 16)
    gdiff = (dq_grid - w_grid).abs().max().item()
    assert gdiff < 1e-6, \
        f"grid-valued weights should survive to fp32 scale rounding: {gdiff}"

    # 3) Non-divisible input width: zero-pad internally, discard padding.
    w_odd = torch.randn(4, 37)
    dq_odd = _qdq_nvfp4(w_odd, 16)
    assert dq_odd.shape == w_odd.shape and torch.isfinite(dq_odd).all()

    # 4) STE gradient: values on the grid, gradient as if qdq were the
    #    identity — equals the analytic grad of the dequantized linear.
    torch.manual_seed(323)
    lin = nn.Linear(48, 8, bias=False)
    fq = FakeQuantLinear(lin, method="nvfp4")
    assert fq.group_size == 16, "NVFP4 default block width must be 16"
    x = torch.randn(3, 48)
    loss = fq(x).pow(2).sum()
    loss.backward()
    with torch.no_grad():
        wdq = fq.fake_quant_weight()
        out = x @ wdq.T
    manual = (2.0 * out).T @ x                   # grad of ||x wdq^T||^2 w.r.t w
    gdiff = (fq.weight.grad - manual).abs().max().item()
    assert gdiff < 1e-5, f"STE gradient mismatch: {gdiff:.3e}"

    # 5) apply_qat integration: every Linear wrapped at block 16, a full
    #    forward/backward step runs, gradients reach the fp32 weights.
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4))
    applied = apply_qat(model, method="nvfp4")
    assert len(applied) == 2 and all(m.method == "nvfp4"
                                     and m.group_size == 16
                                     for _, m in applied)
    x = torch.randn(4, 16)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
               for _, m in applied)
    _expect_raises(ValueError, lambda: apply_qat(model, method="fp3"),
                   "unknown QAT method")
    _pass("test_nvfp4_qat",
          f"RTN err nvfp4 {err_nv:.4f} < mxfp4 {err_mx:.4f}, grid-exact 0 err, "
          f"STE grad diff {gdiff:.1e}, apply_qat {len(applied)} linears")


def test_qat():
    import torch.nn.functional as F
    from helioslm_v5.src.quantization.qat import FakeQuantLinear, apply_qat
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, MXFP4Linear)

    torch.manual_seed(11)
    lin = nn.Linear(64, 32)
    fq = FakeQuantLinear(lin, method="mxfp4")
    x = torch.randn(5, 64)
    out = fq(x)
    # Forward runs on the quantization grid (exactly the qdq matmul).
    ref_w = fq.fake_quant_weight()
    ref = F.linear(x, ref_w, fq.bias)
    assert (out - ref).abs().max().item() == 0.0, (
        "QAT forward must equal the quantize-dequantize matmul exactly")

    # STE backward: the weight gradient equals the DENSE gradient evaluated
    # at the quantized point (identity through the staircase).
    ref_leaf = ref_w.clone().requires_grad_(True)
    bias_leaf = fq.bias.detach().clone().requires_grad_(True)
    F.linear(x, ref_leaf, bias_leaf).pow(2).sum().backward()
    fq.zero_grad()
    out.pow(2).sum().backward()
    assert (fq.weight.grad - ref_leaf.grad).abs().max().item() == 0.0, (
        "STE gradient != dense gradient at the quantized point")

    # QAT training actually converges on a toy least-squares task (the
    # optimizer moves the full-precision weight behind the fake quantizer).
    torch.manual_seed(12)
    w_target = torch.randn(16, 16)
    fq2 = FakeQuantLinear(nn.Linear(16, 16), method="awq", group_size=8)
    opt = torch.optim.Adam(fq2.parameters(), lr=0.05)
    xt = torch.randn(64, 16)
    yt = xt @ w_target.t()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (fq2(xt) - yt).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (
        f"QAT did not converge: {losses[0]:.3f} -> {losses[-1]:.3f}")

    # apply_qat wraps every nn.Linear in place and skips quantized modules
    # (double-wrapping a reconstruction would fake-quantize noise).
    container = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    applied = apply_qat(container, method="mxfp4")
    assert len(applied) == 2 and all(
        isinstance(m, FakeQuantLinear) for m in container), "apply_qat wiring"
    q_container = nn.Sequential(
        MXFP4Linear.from_linear(nn.Linear(8, 8)), nn.Linear(8, 4))
    apply_qat(q_container, method="mxfp4")
    assert isinstance(q_container[0], MXFP4Linear), (
        "already-quantized module must not be re-wrapped")
    assert isinstance(q_container[1], FakeQuantLinear)
    _expect_raises(ValueError, lambda: FakeQuantLinear(
        nn.Linear(4, 4), method="bogus"), "unknown QAT method")
    _pass("test_qat",
          f"fwd==qdq exact, STE grad exact, toy loss "
          f"{losses[0]:.2f}->{losses[-1]:.2f}, wrap/skip OK")
