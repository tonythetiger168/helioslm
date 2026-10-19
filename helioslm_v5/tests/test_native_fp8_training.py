"""Numerical/native-operator and collective optimizer regressions."""
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helioslm_v5.src.training.fp8_trainer import FP8Linear, FP8Trainer


def test_native_rejects_cpu():
    with pytest.raises(RuntimeError, match='requires CUDA'):
        FP8Linear(8, 8, backend='native')(torch.randn(2, 8))


def test_data_parallel_launcher_routes_over_all_experts(monkeypatch):
    from scripts.train_native_fp8 import training_config
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
    assert training_config('mid').moe.device_group_size == 4
    config = training_config('mid', distributed=True)
    assert config.moe.device_group_size == 1
    # Small widths exercise routing without allocating a mid-sized model.
    config.hidden_size = 64
    config.moe.expert_hidden_size = 32
    moe = DeviceLimitedMoE(config).eval()
    x = torch.randn(3, 64)
    reference = moe._route(x)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)
    for rank in (0, 1):
        monkeypatch.setattr(dist, 'get_rank', lambda r=rank: r)
        actual = moe._route(x)
        assert all(torch.equal(a, b) for a, b in zip(reference, actual))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0:2] < (8, 9),
                    reason='FP8-capable CUDA required')
def test_native_forward_backward_operator_and_accuracy():
    torch.manual_seed(44)
    layer = FP8Linear(23, 19, bias=True, backend='native').cuda()
    x = torch.randn(2, 7, 23, device='cuda', requires_grad=True)
    grad = torch.randn(2, 7, 19, device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        y = layer(x)
        y.backward(grad)
    # Proves all three GEMMs execute the actual operator, not float8 casts + mm.
    calls = sum(e.count for e in prof.key_averages() if e.key == 'aten::_scaled_mm')
    assert calls == 3
    flat_x, flat_g = x.detach().flatten(0, 1), grad.flatten(0, 1)
    references = [flat_x @ layer.weight.detach().t(), flat_g @ layer.weight.detach(),
                  flat_g.t() @ flat_x]
    actual = [y.detach().flatten(0, 1), x.grad.flatten(0, 1), layer.weight.grad]
    for value, ref in zip(actual, references):
        assert torch.isfinite(value).all()
        assert (value - ref).norm() / ref.norm() < 0.12
    assert torch.allclose(layer.bias.grad, flat_g.sum(0), atol=1e-5)


class BranchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x, branch=0, poison=False):
        y = (self.a if branch == 0 else self.b)(x)
        return y.square().mean() * (float('nan') if poison else 1)


def _worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(100 + rank)
        model = BranchModel()
        trainer = FP8Trainer(model, SimpleNamespace(lr=1e-3), distributed=True)
        for i, (_, master) in enumerate(trainer.param_map):
            master.grad = (torch.ones_like(master) * (rank + 1)
                           if rank == 0 or i % 2 else None)
        trainer._average_master_gradients()
        for i, (_, master) in enumerate(trainer.param_map):
            assert torch.equal(master.grad, torch.full_like(master, 1.5 if i % 2 else 0.5))
        before = [p.detach().clone() for p in model.parameters()]
        # Opposite branches reproduce rank-local unused MoE parameters.
        trainer.train_step({'x': torch.randn(3, 8), 'branch': rank})
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
        for p in model.parameters():
            peer = p.detach().clone()
            dist.broadcast(peer, 0)
            assert torch.equal(p, peer)
        before = [p.detach().clone() for p in model.parameters()]
        trainer.train_step({'x': torch.randn(3, 8), 'poison': rank == 1})
        assert trainer.skipped_steps == 1
        assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
        hook = model.a.weight.register_hook(lambda g: g * float('nan')) if rank == 1 else None
        trainer.train_step({'x': torch.randn(3, 8)})
        assert trainer.skipped_steps == 2
        assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
        if hook is not None:
            hook.remove()
    finally:
        dist.destroy_process_group()


def test_two_process_collective_update_and_skip(tmp_path):
    mp.spawn(_worker, args=((tmp_path / 'store').as_uri(),), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason='Requires two FP8-capable GPUs and NCCL')
def test_two_gpu_native_nccl_training(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=2', 'scripts/train_native_fp8.py',
                             '--steps', '3', '--out', str(tmp_path / 'nccl.json')],
                            cwd=root, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
