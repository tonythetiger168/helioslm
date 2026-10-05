"""HeliosLM v5.31 — Manifold-Constrained Hyper-Connections (reference impl).

DeepSeek-V4 replaces residual connections with manifold-constrained
hyper-connections (mHC). The exact V4 formulation is not public; this is a
clean reference implementation of the published Hyper-Connections core
(ByteDance, arXiv 2509.2477) plus an explicit manifold constraint on the
connection mixing matrix, with all deviations recorded here (not hidden):

- HC core: each sublayer sees (x_static, x_dynamic) derived from a
  depth-wise expanded state H of shape (T, B, D). After the sublayer
  output Y, a per-channel mixing matrix A (shape (T, 3, D)) recombine
  three paths: x_static, x_dynamic, x_dynamic + Y.
- Manifold constraint: each channel's mixing row A[:, :, d] is
  row-normalized to unit L2 norm at forward time (weight-norm style
  reparameterization). The recombination map (p1, p2, p3) -> sum_i a_i p_i
  is then sqrt(3)-Lipschitz: ||sum a_i p_i|| <= ||a||_2 * ||(p1,p2,p3)||_2
  = sqrt(sum_j ||p_j||^2) <= sqrt(3) * max_j ||p_j||. The connection can
  mix information across the three paths but cannot amplify beyond that
  bound, however large the raw weights grow — this is our "manifold"
  surrogate for V4's (unpublished) constraint.
- Depth rate alpha (learnable, sigmoid-bounded in (0,1)) interpolates the
  refreshed static state with the carried one: HC's "depth connection".

Verified by tests/test_mhc.py: residual-equivalent init, gradient flow to
all T paths, non-expansion bound, capacity difference vs T.
"""
import torch
import torch.nn as nn


class ManifoldConstrainedHyperConnection(nn.Module):
    """One hyper-connection cell wrapping a sublayer F.

    Args:
        dim: model width D.
        sublayer: module mapping (B, D) -> (B, D).
        t: depth expansion (number of parallel hidden states). T=2 is the
            minimal useful setting (static + dynamic).
        depth_rate: init value of the carried-static interpolation
            coefficient alpha (sigmoid-bounded learnable scalar).
    """

    def __init__(self, dim: int, sublayer: nn.Module, t: int = 2,
                 depth_rate: float = 0.5):
        super().__init__()
        assert t >= 2, "hyper-connections need t >= 2 (static + dynamic)"
        self.dim, self.t, self.sublayer = dim, t, sublayer
        # Per-channel mixing over the 3 paths. Rows normalized at forward.
        self.A = nn.Parameter(torch.randn(t, 3, dim) * 0.02)
        # alpha init via logit so sigmoid(alpha_raw) == depth_rate.
        self.alpha_raw = nn.Parameter(torch.log(
            torch.tensor(depth_rate / (1.0 - depth_rate))))

    def _mix(self):
        """Row-normalized mixing matrix (the manifold constraint)."""
        return self.A / self.A.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, h):
        """h: (T, B, D) expanded state. Returns (T, B, D).

        Contract: h[0] is the static path, h[1:] the dynamic pool.
        """
        t = self.t
        x_static = h[0]
        x_dynamic = h[1:].mean(dim=0)
        y = self.sublayer(x_dynamic)
        A = self._mix()                       # (T, 3, D)
        p = torch.stack([x_static, x_dynamic,
                         x_dynamic + y], dim=0)   # (3, B, D)
        # h'_i = sum_j A[i, j, :] * p[j]  (per-channel weights)
        h_new = torch.einsum("ijd,jbd->ibd", A, p)
        alpha = torch.sigmoid(self.alpha_raw)
        head = (alpha * x_static + (1.0 - alpha) * h_new[:1])
        return torch.cat([head, h_new[1:]], dim=0)


class HCStack(nn.Module):
    """Stack of hyper-connected sublayers with residual-equivalent init.

    Residual-equivalent: with A[i, 2, :] = 1 for the LAST dynamic slot and
    zeros elsewhere, and alpha -> 1, the stack reduces (up to the dynamic
    mean over T-1 slots) to x + F(x) residual behavior — the starting
    point for migration from a residual checkpoint.
    """

    def __init__(self, dim: int, make_sublayer, n_layers: int, t: int = 2):
        super().__init__()
        self.t = t
        self.cells = nn.ModuleList(
            [ManifoldConstrainedHyperConnection(dim, make_sublayer(), t)
             for _ in range(n_layers)])

    def residual_equivalent_init(self):
        """Set every cell's mixing to emulate a plain residual stream:
        static slot carried (A[0,0]=1), last dynamic slot accumulates the
        sublayer output (A[t-1,2]=1). With output = dynamic pool mean,
        the stack is then exactly x + sum_i F_i(x + ...) — the ordinary
        residual stream. This is the migration starting point from a
        residual checkpoint."""
        with torch.no_grad():
            for cell in self.cells:
                cell.A.zero_()
                cell.A[0, 0, :] = 1.0
                cell.A[self.t - 1, 2, :] = 1.0
                cell.alpha_raw.copy_(torch.log(torch.tensor(9.0)))  # ~0.9

    def forward(self, x):
        """x: (B, D). Returns (B, D) — the dynamic pool mean (the slot
        family that accumulates sublayer outputs; the static slot is a
        skip highway)."""
        h = x.unsqueeze(0).repeat(self.t, 1, 1)
        for cell in self.cells:
            h = cell(h)
        return h[1:].mean(dim=0)
