"""Quantization-Aware Training (QAT) — straight-through fake-quant wrapper.

The v5.6 MXFP4 recipe notes QAT as its missing half; this module provides
it in the standard STE (straight-through estimator) form used since
Bengio et al. 2013:

    forward :  y = x @ qdq(w)^T + b      (gradients NOT routed through qdq)
    backward:  dL/dw := dL/d(w_dq)       (identity — the "straight through")

where qdq(w) quantize-dequantizes w onto the target grid (MXFP4 E2M1
blocks, AWQ 4-bit groups, or — v5.22 — the NVFP4 E2M1/E4M3 two-level
hierarchy, reusing no external kernels), so the TRAINING forward sees the
quantized weights the deployed model will use, while the optimizer still
moves the underlying full-precision w.

Simplifications and deviations, honestly stated:
  - The fake-quant grid is recomputed from w on every forward (no cached
    codes), which is the correct-but-slow reference behaviour; a fused
    kernel would maintain qweight incrementally.
  - STE bias is real: the gradient pretends qdq is identity in a region
    where it is a staircase (zero a.e.). This is the accepted QAT
    approximation, not an exact gradient of the forward.
  - Only nn.Linear layers are wrapped; already-quantized modules
    (AWQLinear/GPTQLinear/MXFP4Linear) are left untouched — wrapping a
    quantized module would fake-quantize a reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from helioslm_v5.src.quantization.standard_quant import (
    AWQLinear, MXFP4Linear,
)


class _StraightThrough(torch.autograd.Function):
    """Identity backward for the fake-quant step (the STE)."""

    @staticmethod
    def forward(ctx, w_float, w_fake_quant):
        # Return the QUANTIZED values but keep w_float in the graph so the
        # optimizer's state attaches to the full-precision parameter; the
        # backward below routes gradients to w_float untouched.
        ctx.save_for_backward(w_float)
        return w_fake_quant

    @staticmethod
    def backward(ctx, grad_out):
        (w_float,) = ctx.saved_tensors
        # Straight-through: pretend the quantize-dequantize staircase is
        # the identity. grad w.r.t. the reference values is dropped.
        return grad_out.to(w_float.dtype), None


def _qdq_mxfp4(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Quantize-dequantize onto the MXFP4 E2M1 grid (scale-free helper).

    Reuses MXFP4Linear's kernel by packing through a throwaway container:
    the round-trip through codes makes the grid explicit rather than
    reimplementing the rounding rule.
    """
    out_f, in_f = weight.shape
    tmp = nn.Linear(in_f, out_f, bias=False, device=weight.device,
                    dtype=weight.dtype)
    tmp.weight = nn.Parameter(weight.detach())
    q = MXFP4Linear.from_linear(tmp, block_size=block_size)
    return q._dequantize().to(weight.dtype)


def _qdq_awq(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Quantize-dequantize onto the AWQ 4-bit per-group asymmetric grid
    (plain-RTN form, alpha=0 — no calibration scaling available per-step;
    the activation-aware scaling remains a post-hoc calibration tool)."""
    out_f, in_f = weight.shape
    tmp = nn.Linear(in_f, out_f, bias=False, device=weight.device,
                    dtype=weight.dtype)
    tmp.weight = nn.Parameter(weight.detach())
    q = AWQLinear.from_linear(tmp, group_size=group_size)
    return q._dequantize().to(weight.dtype)


# v5.22: NVFP4-format QAT target (NVIDIA Blackwell direction, also the
# verl/DeepSeek-V4-class QAT recipe). E2M1 values in blocks of 16 with an
# FP8 (E4M3) scale per block under a full-precision scale per output row.
# Two honest deviations from spec NVFP4, both FINER than the spec:
#   - the top-level scale is per OUTPUT ROW, not per tensor (the standard
#     GEMM-weight variant; a row's block scales share one fp32 factor);
#   - round-to-nearest on the E2M1 grid breaks ties toward the SMALLER
#     magnitude (argmin picks the first grid point), not RNE.
_E2M1_MAX = 6.0            # largest magnitude on the E2M1 grid
_E4M3_MAX = 448.0          # largest finite magnitude of FP8 E4M3


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round non-negative magnitudes to the nearest E2M1 grid point.

    Grid: {0, .5, 1, 1.5, 2, 3, 4, 5, 6} (the E2M1 mantissa step doubles
    past the 1.0 boundary, hence the non-uniform spacing). Ties go to the
    smaller magnitude (documented deviation; RTN either way).
    """
    grid = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = (x.unsqueeze(-1) - grid).abs()          # [..., 9]
    return grid[d.argmin(dim=-1)]


def _qdq_nvfp4(weight: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Quantize-dequantize onto the NVFP4 grid (scale-free helper).

    Blocks run along the input dimension (zero-padded to a multiple of
    ``block_size``; padding contributes zero to absmax and is discarded).
    Per block: scale_b = absmax/6 clamped up, stored as E4M3 under a per-row
    fp32 scale (row_scale = max(scale_b)/448 clamped up). Values are rounded
    to E2M1 against the EFFECTIVE scale row_scale*scale_b(E4M3), so the
    dequantized tensor is exactly what an NVFP4 kernel would reconstruct.
    """
    out_f, in_f = weight.shape
    pad = (-in_f) % block_size
    w = F.pad(weight.detach(), (0, pad))
    blocks = w.view(out_f, -1, block_size)
    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    bscale = (absmax / _E2M1_MAX).clamp_min(1e-12)          # fp32, per block
    # Two-level hierarchy: E4M3 block scales under a per-row fp32 scale.
    row_scale = (bscale.amax(dim=-2, keepdim=True) / _E4M3_MAX).clamp_min(1e-12)
    bscale_e4m3 = (bscale / row_scale).clamp(max=_E4M3_MAX) \
        .to(torch.float8_e4m3fn).to(torch.float32)
    eff = (row_scale * bscale_e4m3).clamp_min(1e-12)        # [out, nblk, 1]
    mag = _round_e2m1((blocks / eff).abs())
    deq = (torch.sign(blocks) * mag * eff).view(out_f, -1)[:, :in_f]
    return deq.to(weight.dtype)


_QDQ = {
    "mxfp4": _qdq_mxfp4,
    "awq": _qdq_awq,
    "nvfp4": _qdq_nvfp4,
}

_QDQ_DEFAULT_GROUP = {
    "mxfp4": 32,
    "nvfp4": 16,   # NVFP4 spec block width
    "awq": 128,
}


class FakeQuantLinear(nn.Module):
    """nn.Linear whose weight passes through quantize-dequantize (STE).

    Interface-compatible with nn.Linear for the purposes of this codebase
    (in_features / out_features / bias / weight / forward); the stored
    ``weight`` Parameter stays full-precision — only the forward value is
    on the quantization grid.
    """

    def __init__(self, linear: nn.Linear, method: str = "mxfp4",
                 group_size: int = None):
        super().__init__()
        if method not in _QDQ:
            raise ValueError(
                f"unknown QAT method {method!r}; available: {sorted(_QDQ)}"
            )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.method = method
        if group_size is None:
            group_size = _QDQ_DEFAULT_GROUP[method]
        self.group_size = group_size
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def fake_quant_weight(self) -> torch.Tensor:
        """The quantized values the forward sees (no gradient)."""
        with torch.no_grad():
            return _QDQ[self.method](self.weight.detach(), self.group_size)

    def forward(self, x):
        w_dq = _QDQ[self.method](self.weight.detach(), self.group_size)
        # STE: values are on the grid, gradients flow as if w_dq == w.
        w_st = _StraightThrough.apply(self.weight, w_dq)
        return F.linear(x, w_st.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


def apply_qat(model: nn.Module, method: str = "mxfp4", group_size: int = None):
    """Wrap every nn.Linear in ``model`` with FakeQuantLinear, in place.

    Already-quantized modules (AWQLinear / GPTQLinear / MXFP4Linear /
    FakeQuantLinear) are skipped — their weights already live on a grid.
    Returns the list of (module_name, wrapper) pairs applied.
    """
    if method not in _QDQ:
        raise ValueError(
            f"unknown QAT method {method!r}; available: {sorted(_QDQ)}"
        )
    skip = (FakeQuantLinear, AWQLinear, MXFP4Linear)
    try:  # GPTQLinear always exists; guard anyway for forward-compat.
        from helioslm_v5.src.quantization.standard_quant import GPTQLinear
        skip = skip + (GPTQLinear,)
    except ImportError:
        pass

    modules = dict(model.named_modules())
    targets = [(name, m) for name, m in modules.items()
               if isinstance(m, nn.Linear) and not isinstance(m, skip)]
    applied = []
    for name, module in targets:
        wrapper = FakeQuantLinear(module, method=method, group_size=group_size)
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, wrapper)
        applied.append((name, wrapper))
    return applied
