"""Strict CUDA FP8 GEMMs for forward, input gradient and weight gradient.

FP32 master parameters and optimizer state are intentional mixed precision.
No simulated matmul fallback: unsupported hardware raises an error.
"""
import torch
import torch.nn.functional as F


def quantize(x, dtype):
    limit = torch.finfo(dtype).max
    scale = (x.detach().float().abs().amax() / limit).clamp_min(1e-12)
    return (x.float() / scale).clamp(-limit, limit).to(dtype), scale


def gemm(a, b, a_dtype=torch.float8_e4m3fn, b_dtype=torch.float8_e4m3fn):
    """Multiply arbitrary 2D shapes, padding to CUDA alignment requirements."""
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError('Native FP8 requires CUDA; no simulation fallback')
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError('FP8 GEMM inner dimensions differ')
    if not m or not n or not k:
        return torch.zeros((m, n), device=a.device, dtype=torch.float32)
    pm, pk, pn = (-m) % 16, (-k) % 16, (-n) % 16
    qa, sa = quantize(F.pad(a, (0, pk, 0, pm)), a_dtype)
    qb, sb = quantize(F.pad(b, (0, pn, 0, pk)), b_dtype)
    # cuBLAS FP8 requires row-major A and column-major B.
    qb = qb.t().contiguous().t()
    out = torch._scaled_mm(qa.contiguous(), qb, scale_a=sa, scale_b=sb,
                           out_dtype=torch.float32, use_fast_accum=False)
    if isinstance(out, tuple):
        out = out[0]
    return out[:m, :n]


class NativeFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        result = gemm(x.reshape(-1, x.shape[-1]), weight.t())
        if bias is not None:
            result = result + bias.float()
        return result.reshape(*x.shape[:-1], weight.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        g = grad.reshape(-1, weight.shape[0])
        flat_x = x.reshape(-1, x.shape[-1])
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = gemm(g, weight, a_dtype=torch.float8_e5m2).reshape_as(x).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = gemm(g.t(), flat_x, a_dtype=torch.float8_e5m2).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = g.float().sum(0).to(ctx.bias_dtype)
        return dx, dw, db
