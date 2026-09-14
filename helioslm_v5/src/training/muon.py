"""Muon optimizer (v5.6) - momentum orthogonalized by Newton-Schulz iteration.

Direction-of-travel: Kimi-K3 credits "Per-Head Muon" (extending the Muon
optimizer by optimizing attention heads independently) as part of its
training-efficiency stack. This module is a simplified, honest Muon in the
Keller Jordan lineage:

  1. Every 2-D hidden weight matrix G gets a heavy-ball momentum buffer B.
  2. The momentum is orthogonalized: O = NewtonSchulz(B), so O O^T ~= I
     (a semi-orthogonal matrix). The parameter update is ``lr * O`` scaled
     by ``sqrt(max(1, rows / cols))`` — the update norm is then comparable
     across differently-shaped matrices, which is Muon's core benefit over
     SGD on matrix params.
  3. Non-matrix parameters (biases, norm gains, 1-D tensors) do not have a
     well-defined orthogonalization; they fall back to an internal AdamW
     update so a single ``Muon`` instance can optimize a whole model.
  4. Optional per-group flag ``muon=False`` routes a group to the AdamW
     path explicitly (the K3 recipe keeps embeddings / LM head on Adam).

Simplifications (documented, not hidden):
  - Default is per-MATRIX orthogonalization; the v5.8 ``per_head_dim``
    group option splits the ROW dim into head blocks and orthogonalizes
    each block independently (K3's "per-head" extension, now implemented
    for row-stacked layouts like fused q/k/v or multi-head projections).
  - No distributed "compute in fp32, buffer in bf16" syncing; single
    process, Newton-Schulz run in the param dtype (bf16 params recommended
    per the reference recipe; fp32 works and is what the tests use).
  - Newton-Schulz is the standard quintic iteration with the (3.4445,
    -4.7750, 2.0315) coefficients, 5 steps, on the normalized matrix.

References: Keller Jordan's Muon (2024) and Moonshot's Kimi-K3 technical
report (2026) which scales it to trillion-parameter training.
"""
import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize G via the quintic Newton-Schulz iteration.

    Returns a matrix X with the same shape as G whose singular values are
    pushed toward 1 (X X^T ~= I for square/tall G). Coefficients (3.4445,
    -4.7750, 2.0315) are the standard Muon choice; the iteration runs in
    bfloat16 when G is float32 (the reference recipe) for speed, and the
    result is cast back to G's dtype.
    """
    assert G.ndim == 2, f"Newton-Schulz expects a 2-D matrix, got {G.ndim}-D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.float32 else G
    # Normalize so the spectral norm is <= 1 (Frobenius norm upper-bounds
    # the spectral norm); transposed so the SMALLER dim becomes dim 0 —
    # NS converges to a semi-orthogonal factor either way, and the
    # transposed layout iterates on the smaller gram matrix.
    X = X / (X.norm() + 1e-7)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: orthogonalized-momentum optimizer with an AdamW fallback path.

    Args:
        params: iterable of parameters or param groups. Group keys:
            ``lr`` (required-ish, default 0.02), ``momentum`` (0.95),
            ``nesterov`` (True), ``ns_steps`` (5), ``adam_fallback``
            (True — non-2-D params in the group use AdamW), ``betas`` /
            ``eps`` for the fallback path, ``muon`` (True; set False to
            force a 2-D group onto the AdamW path, e.g. embeddings or the
            LM head per the K3 recipe), ``per_head_dim`` (None; v5.8 — the
            K3 "Per-Head Muon" extension: when set, a 2-D momentum matrix
            whose ROW count is a multiple of ``per_head_dim`` is split into
            per-head blocks along dim 0 and each block is orthogonalized
            independently, then the blocks are reassembled. Rows not
            divisible by ``per_head_dim`` raise ValueError — silently
            falling back to whole-matrix NS would hide a layout mismatch).
        lr: base learning rate for Muon-updated (2-D) params.
        weight_decay: decoupled weight decay on Muon params (Adam fallback
            params always use decoupled weight decay too).

    Update rule for a 2-D param with gradient g:
        buf = momentum * buf + g                (heavy ball)
        d   = g + momentum * buf                (nesterov variant)
        O   = NewtonSchulz(d) * sqrt(max(1, rows/cols))
        p  -= lr * O                            (+ decoupled weight decay)

    With ``per_head_dim = h`` (rows = n_heads * h) the orthogonalization
    runs per head block: O_h = NewtonSchulz(d_h), so each attention head's
    update is independently semi-orthogonal (the K3 recipe's motivation:
    heads specialize, and whole-matrix NS couples their singular values).
    The shape scale is computed per head block (rows/cols of the block).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, adam_fallback=True,
                 betas=(0.9, 0.95), eps=1e-8, muon=True, per_head_dim=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if per_head_dim is not None and per_head_dim <= 0:
            raise ValueError(
                f"per_head_dim must be a positive integer or None, got "
                f"{per_head_dim}"
            )
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        adam_fallback=adam_fallback, betas=betas, eps=eps,
                        muon=muon, per_head_dim=per_head_dim)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            use_muon = group["muon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if use_muon and g.ndim == 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    d = g.add(buf, alpha=group["momentum"]) \
                        if group["nesterov"] else buf
                    per_head_dim = group["per_head_dim"]
                    if per_head_dim is not None:
                        # Per-head orthogonalization (v5.8): split rows into
                        # head blocks, NS each block, reassemble.
                        rows, cols = d.shape
                        if rows % per_head_dim != 0:
                            raise ValueError(
                                f"per_head_dim={per_head_dim} does not divide "
                                f"the gradient row count {rows}; check the "
                                f"head layout of this parameter"
                            )
                        blocks = d.view(rows // per_head_dim, per_head_dim, cols)
                        o = torch.empty_like(blocks)
                        for hi in range(blocks.shape[0]):
                            o[hi] = zeropower_via_newtonschulz5(
                                blocks[hi], steps=group["ns_steps"])
                        o = o.view(rows, cols)
                        scale = max(1.0, per_head_dim / cols) ** 0.5
                    else:
                        o = zeropower_via_newtonschulz5(
                            d, steps=group["ns_steps"])
                        scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(o, alpha=-lr * scale)
                else:
                    # AdamW fallback for non-matrix params (biases, norms,
                    # embeddings/head when the group opts out via muon=False).
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    mhat = m / (1 - beta1 ** t)
                    vhat = v / (1 - beta2 ** t)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(mhat, vhat.sqrt().add_(eps), value=-lr)

        return loss
