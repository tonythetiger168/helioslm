"""FP8 conversion preserves placement before optimizer masters are built."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helioslm_v5.src.training.fp8_trainer import FP8Linear, FP8Trainer  # noqa: E402


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float64), ('cpu', torch.bfloat16), ('meta', torch.float32)])
def test_conversion_preserves_device_and_dtype(device, dtype):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=True)).to(device=device, dtype=dtype)
    weight = model[0].weight.detach().clone()
    trainer = FP8Trainer(model, SimpleNamespace(lr=1e-3))
    assert isinstance(model[0], FP8Linear)
    assert model[0].weight.device.type == device
    assert model[0].weight.dtype == dtype
    assert all(p.device.type == device for p in model.parameters())
    assert all(b.device.type == device for b in model.buffers())
    assert all(master.dtype == torch.float32 and master.device.type == device
               for _, master in trainer.param_map)
    if device == 'cpu':
        assert torch.equal(model[0].weight, weight)
        out = model(torch.randn(2, 8, dtype=dtype))
        assert out.dtype == dtype and torch.isfinite(out).all()
        out.square().mean().backward()
        assert model[0].weight.grad is not None
        assert torch.isfinite(model[0].weight.grad).all()
