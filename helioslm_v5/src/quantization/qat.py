"""Quantization-Aware Training (QAT) — straight-through fake-quant wrapper.

The v5.6 MXFP4 recipe notes QAT as its missing half; this module provides
it in the standard STE (straight-through estimator) form used since
Bengio et al. 2013:

    forward :  y = x @ qdq(w)^T + b      (gradients NOT routed through qdq)
    backward:  dL/dw := dL/d(w_dq)       (identity — the "straight through")

where qdq(w) quantize-dequantizes w onto the target grid (MXFP4 E2M1
blocks or AWQ 4-bit groups, reusing the exact kernels from
``standard_quant``), so the TRAINING forward sees the quantized weights the
deployed model will use, while the optimizer still moves the underlying
full-precision w.

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


_QDQ = {
    "mxfp4": _qdq_mxfp4,
    "awq": _qdq_awq,
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
            group_size = 32 if method == "mxfp4" else 128
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

    # Re-bind MTP shared modules: wrapping replaces model.lm_head with a
    # FakeQuantLinear, leaving mtp_modules[i].lm_head pointing at the OLD
    # nn.Linear (a silent break of the weight tying — MTP aux-loss training
    # would forward an unquantized head while the main head is fake-quantized).
    # Same re-bind QuantizationManager.quantize_model performs after module
    # replacement. Defensive getattr: model may not have MTP.
    mtp_modules = getattr(model, "mtp_modules", None)
    if mtp_modules is not None:
        lm_head = getattr(model, "lm_head", None)
        embed_tokens = getattr(model, "embed_tokens", None)
        for mtp in mtp_modules:
            if lm_head is not None and hasattr(mtp, "lm_head"):
                mtp.lm_head = lm_head
            if embed_tokens is not None and hasattr(mtp, "embed_tokens"):
                mtp.embed_tokens = embed_tokens
    return applied
