"""T154 — adaptive micro-batch and the grouped-matmul fallback, v5.161.

1. ``get_adaptive_batch_size``: 4 under 30 GB of CUDA memory, else 8
   (and 8 on CPU);
2. ``fit_micro_batch`` halves the micro-batch on CUDA out-of-memory and
   keeps micro_batch * grad_accum; the pilot's probe step runs;
3. ``safe_grouped_matmul`` equals ``inputs[i] @ weights[e_i]`` row by row,
   in the original token order, with and without ``F.grouped_mm``, for
   unsorted indices and idle experts, gradients included.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_adaptive_batch_fallback.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.src.moe import moe_fallback  # noqa: E402
from helioslm_v5.src.moe.moe_fallback import safe_grouped_matmul  # noqa: E402
from helioslm_v5.src.training import adaptive_batch  # noqa: E402
from helioslm_v5.src.training.adaptive_batch import (  # noqa: E402
    fit_micro_batch, get_adaptive_batch_size)
from helioslm_v5.src.training.pretrain import pilot_config  # noqa: E402


@pytest.mark.parametrize("gb,expected", [(22, 4), (24, 4), (48, 8), (80, 8)])
def test_adaptive_batch_size(monkeypatch, gb, expected):
    monkeypatch.setattr(adaptive_batch.torch.cuda, "get_device_properties",
                        lambda d: SimpleNamespace(total_memory=gb * 1024 ** 3))
    assert get_adaptive_batch_size("cuda") == expected
    assert get_adaptive_batch_size("cpu") == 8


def test_fit_micro_batch():
    tried = []

    def step(mb):
        tried.append(mb)
        if mb > 2:
            raise torch.cuda.OutOfMemoryError("fake")
    assert fit_micro_batch(step, 8, 3, log=lambda m: None) == (2, 12)
    assert tried == [8, 4, 2]
    assert fit_micro_batch(lambda mb: None, 8, 1) == (8, 1)

    def never(mb):
        raise torch.cuda.OutOfMemoryError("fake")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        fit_micro_batch(never, 4, 1, log=lambda m: None)


def test_pilot_probe_step():
    sys.path.insert(0, str(ROOT / "examples"))
    import pilot_100m
    cfg = pilot_config("k3_llamacpp", vocab_size=512, hidden_size=128,
                       num_layers=4, intermediate_size=256, max_seq=64)
    for ce_chunk in (0, 32):
        pilot_100m._probe_step(cfg, 2, 64, "cpu", torch.float32, ce_chunk)


def _reference(x, w, idx):
    return torch.stack([x[i] @ w[idx[i]] for i in range(len(idx))])


@pytest.mark.parametrize("grouped", [False, True])
def test_safe_grouped_matmul(grouped):
    if grouped and not moe_fallback.HAS_GROUPED_MM:
        pytest.skip("torch without F.grouped_mm")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(37, 16, generator=g, requires_grad=True)
    w = torch.randn(6, 16, 24, generator=g, requires_grad=True)
    idx = torch.randint(0, 6, (37,), generator=g)
    idx[idx == 4] = 5                                   # expert 4 idle
    y = safe_grouped_matmul(x, w, idx, use_grouped_mm=grouped)
    ref = _reference(x, w, idx)
    assert torch.allclose(y, ref, atol=1e-5)
    seed = torch.randn(y.shape, generator=g)
    gx, gw = torch.autograd.grad((y * seed).sum(), (x, w))
    rx, rw = torch.autograd.grad((ref * seed).sum(), (x, w))
    assert torch.allclose(gx, rx, atol=1e-5)
    assert torch.allclose(gw, rw, atol=1e-5)
    assert gw[4].abs().sum() == 0


def test_safe_grouped_matmul_shapes():
    with pytest.raises(ValueError, match="shapes"):
        safe_grouped_matmul(torch.randn(4, 8), torch.randn(2, 6, 3),
                            torch.zeros(4, dtype=torch.long))
