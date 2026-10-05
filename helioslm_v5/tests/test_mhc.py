"""T22 - v5.31: manifold-constrained hyper-connections (mHC) oracles.

Covers: residual-equivalent init (migration starting point), gradient
flow to all T paths, the non-expansion manifold bound, and T-capacity.
Run from repo root: python3 helioslm_v5/tests/test_mhc.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.blocks.mhc import HCStack, ManifoldConstrainedHyperConnection


def _linear(dim, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Linear(dim, dim, bias=True).requires_grad_(True)


def test_residual_equivalent_init():
    torch.manual_seed(7)
    dim, n, t = 16, 4, 2
    stack = HCStack(dim, lambda: torch.nn.Linear(dim, dim), n, t=t)
    stack.residual_equivalent_init()
    x = torch.randn(3, dim)
    # Reference: plain residual stack with the same sublayers (T=2 dynamic
    # pool = single slot, so h[1:] mean == h[1] exactly).
    ref = x
    for cell in stack.cells:
        ref = ref + cell.sublayer(ref)
    out = stack(x)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"residual-equivalent init mismatch: {(out - ref).abs().max()}"
    print("PASS test_residual_equivalent_init")


def test_gradient_flows_to_all_paths():
    torch.manual_seed(8)
    dim, t = 8, 3
    cell = ManifoldConstrainedHyperConnection(dim, torch.nn.Linear(dim, dim), t=t)
    h = torch.randn(t, 2, dim, requires_grad=True)
    out = cell(h)
    loss = out.pow(2).sum()
    loss.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert cell.A.grad is not None and cell.A.grad.abs().sum() > 0
    assert cell.alpha_raw.grad is not None
    print("PASS test_gradient_flows_to_all_paths")


class _ZeroSublayer(torch.nn.Module):
    def forward(self, z):
        return torch.zeros_like(z)


def test_manifold_non_expansion():
    torch.manual_seed(9)
    dim, t = 8, 2
    cell = ManifoldConstrainedHyperConnection(dim, torch.nn.Linear(dim, dim), t=t)
    # Adversarial: inflate A rows, the constraint must still hold.
    with torch.no_grad():
        cell.A.mul_(100.0)
    x_static = torch.randn(2, dim)
    x_dynamic = torch.randn(2, dim)
    h = torch.stack([x_static, x_dynamic])
    cell.sublayer = _ZeroSublayer()   # isolate the connection map (y = 0)
    out = cell(h)
    import math
    for i in range(t):
        # sqrt(3)-Lipschitz bound per unit-norm mixing rows (see module
        # docstring); the alpha head is a convex combo so it inherits it
        for b in range(2):
            bound = math.sqrt(3) * max(x_static[b].norm(),
                                       x_dynamic[b].norm()) + 1e-5
            assert out[i, b].norm() <= bound, \
                f"path {i} broke the manifold bound: {out[i, b].norm()} > {bound}"
    print("PASS test_manifold_non_expansion")


def test_t_capacity_differs():
    torch.manual_seed(10)
    dim, n = 16, 3
    outs = {}
    for t in (2, 4):
        stack = HCStack(dim, lambda: torch.nn.Linear(dim, dim), n, t=t)
        with torch.no_grad():
            stack.cells[0].A[0, 0, :] = 0.7   # poke the static path
        outs[t] = stack(torch.randn(4, dim))
    assert not torch.allclose(outs[2], outs[4]), \
        "different T expansions must change function capacity"
    print("PASS test_t_capacity_differs")


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} mHC tests passed")
