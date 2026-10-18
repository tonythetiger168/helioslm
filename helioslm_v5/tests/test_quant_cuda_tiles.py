"""Exercise CUDA tile arithmetic on CPU as well; actual CUDA dispatch when available."""
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helioslm_v5.src.quantization.standard_quant import (  # noqa: E402
    AWQLinear, GPTQLinear, MXFP4Linear, _cuda_tile_columns)


def make_quant(method, width, device='cpu'):
    torch.manual_seed(42)
    linear = torch.nn.Linear(width, 33).to(device)
    if method == 'awq':
        mod = AWQLinear.from_linear(linear, group_size=17, activations=torch.randn(48, width, device=device))
    elif method == 'gptq':
        mod = GPTQLinear.from_linear(linear, group_size=17)
        # Alternating groups exercises the pathological old run count and
        # does not rely on a random calibration ordering to hit the boundary.
        mod.g_idx.copy_(torch.arange(width, device=device) % mod.scales.shape[0])
    else:
        mod = MXFP4Linear.from_linear(linear, block_size=32)
    return mod


@pytest.mark.parametrize('method', ['awq', 'gptq', 'mxfp4'])
@pytest.mark.parametrize('width', [1, 65, 1031])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_tiles_outputs_and_input_gradients(method, width, dtype):
    mod = make_quant(method, width)
    # Noncontiguous 3D input and ragged input/output dimensions.
    x = torch.randn(2, width, 3, dtype=dtype).transpose(1, 2).requires_grad_()
    mod.fused = False
    ref = mod(x)
    bias = mod.bias.to(dtype)
    scaled = x / mod.act_scale.to(dtype) if method == 'awq' and mod.act_scale is not None else x
    actual = mod._tiled_forward(scaled, bias)
    tolerance = {torch.float32: 1e-4, torch.float16: 0.005, torch.bfloat16: 0.04}[dtype]
    assert (actual.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-8) < tolerance
    g_ref = torch.autograd.grad(ref.float().sum(), x, retain_graph=True)[0]
    g_actual = torch.autograd.grad(actual.float().sum(), x)[0]
    assert torch.isfinite(g_actual).all()
    torch.testing.assert_close(g_actual, g_ref, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('method', ['awq', 'gptq', 'mxfp4'])
def test_tiles_strict_checkpoint_and_no_dense_cache(method):
    mod = make_quant(method, 65)
    state_keys = set(mod.state_dict())
    stream = io.BytesIO()
    torch.save(mod.state_dict(), stream)
    stream.seek(0)
    clone = make_quant(method, 65)
    clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    x = torch.randn(65)
    scaled = x / mod.act_scale if method == 'awq' and mod.act_scale is not None else x
    a = mod._tiled_forward(scaled, mod.bias)
    b = clone._tiled_forward(scaled, clone.bias)
    torch.testing.assert_close(a, b)
    with torch.no_grad():
        mod.qweight.bitwise_xor_(1)
    changed = mod._tiled_forward(scaled, mod.bias)
    assert not torch.equal(changed, a)
    mod.fused = False
    torch.testing.assert_close(changed, mod(x), atol=1e-5, rtol=1e-4)
    assert set(mod.state_dict()) == state_keys


def test_tile_bound():
    for outputs in [1, 33, 1024, 14336, 1048576]:
        width = _cuda_tile_columns(outputs)
        assert 1 <= width <= 1024 and width * outputs <= 1048576


@pytest.mark.parametrize('method', ['awq', 'gptq', 'mxfp4'])
def test_tiles_without_bias(method):
    mod = make_quant(method, 65)
    mod.register_parameter('bias', None)
    x = torch.randn(4, 65)
    mod.fused = False
    scaled = x / mod.act_scale if method == 'awq' and mod.act_scale is not None else x
    torch.testing.assert_close(mod._tiled_forward(scaled, None), mod(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('method', ['awq', 'gptq', 'mxfp4'])
@pytest.mark.parametrize('storage_dtype', [torch.float16, torch.bfloat16])
def test_converted_scale_buffers_with_float32_input(method, storage_dtype):
    mod = make_quant(method, 65).to(dtype=storage_dtype)
    x = torch.randn(2, 65)
    mod.fused = False
    scaled = x / mod.act_scale.float() if method == 'awq' and mod.act_scale is not None else x
    bias = mod.bias.float()
    torch.testing.assert_close(mod._tiled_forward(scaled, bias), mod(x), atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA hardware required')
@pytest.mark.parametrize('method', ['awq', 'gptq', 'mxfp4'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_actual_cuda_dispatch(method, dtype, monkeypatch):
    mod = make_quant(method, 1031, 'cuda')
    x = torch.randn(2, 1031, device='cuda', dtype=dtype)
    mod.fused = False
    ref = mod(x)
    mod.fused = True
    # Detect regressions to host reads, including GPTQ's .tolist()/int(tensor).
    def host_read(*args, **kwargs):
        raise AssertionError('CUDA tile forward must not read tensor values on host')
    with monkeypatch.context() as m:
        for name in ('tolist', 'item', '__int__'):
            m.setattr(torch.Tensor, name, host_read)
        out = mod(x)
        torch.cuda.synchronize()
    tolerance = {torch.float32: 1e-4, torch.float16: 0.005, torch.bfloat16: 0.04}[dtype]
    torch.testing.assert_close(out, ref, atol=tolerance, rtol=tolerance)
