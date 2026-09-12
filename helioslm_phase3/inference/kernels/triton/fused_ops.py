"""Fused operations Triton kernels.

Combines multiple operations into single kernels to reduce memory traffic:
  - LayerNorm + GELU
  - RMSNorm
  - Bias + Activation
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_layer_norm_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_m, stride_d,
    M, D,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm + GELU activation.

    Computes: output = GELU(LayerNorm(input) * weight + bias)
    """
    pid = tl.program_id(0)

    # Row index
    row_idx = pid

    # Load row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    input_ptrs = input_ptr + row_idx * stride_m + offs * stride_d
    x = tl.load(input_ptrs, mask=mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Load weight and bias
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)

    # Scale and shift
    x_scaled = x_norm * w + b

    # GELU activation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.tanh(
        0.7978845608028654 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)
    ))
    output = x_scaled * cdf

    # Store
    output_ptrs = output_ptr + row_idx * stride_m + offs * stride_d
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def rms_norm_kernel(
    input_ptr, weight_ptr, output_ptr,
    stride_m, stride_d,
    M, D,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm kernel."""
    pid = tl.program_id(0)
    row_idx = pid

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    input_ptrs = input_ptr + row_idx * stride_m + offs * stride_d
    x = tl.load(input_ptrs, mask=mask, other=0.0)

    # RMS
    rms = tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    x_norm = x / rms

    # Scale
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    output = x_norm * w

    output_ptrs = output_ptr + row_idx * stride_m + offs * stride_d
    tl.store(output_ptrs, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-6):
    """Fused LayerNorm + GELU."""
    M, D = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (M,)

    fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        x.stride(0), x.stride(1),
        M, D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return output


def fused_rms_norm(x, weight, eps=1e-6):
    """Fused RMSNorm."""
    M, D = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (M,)

    rms_norm_kernel[grid](
        x, weight, output,
        x.stride(0), x.stride(1),
        M, D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return output
