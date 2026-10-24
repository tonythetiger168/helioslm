"""HeliosLM v5 test cases — Muon optimizer.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch
import torch.nn as nn

from evastar.tests.case_common import (_expect_raises, _pass)


# ----------------------------------------------------------------------
# v5.6: Muon optimizer — Newton-Schulz orthogonalization quality, quadratic
# convergence vs SGD, AdamW fallback for non-matrix params, muon=False
# group routing
# ----------------------------------------------------------------------
def test_muon():
    from evastar.src.training.muon import (
        Muon, zeropower_via_newtonschulz5)

    torch.manual_seed(601)
    # 1) Orthogonalization quality: 5-step NS pushes every singular value
    #    into the documented [~0.3, ~1.35] band around 1 (approximate
    #    orthogonalization — the standard Muon recipe; exactness is NOT
    #    expected at 5 steps) with mean singular value ~= 1 (energy kept).
    A = torch.randn(24, 24)
    O = zeropower_via_newtonschulz5(A)
    sv = torch.linalg.svdvals(O)
    assert 0.25 < sv.min().item() and sv.max().item() < 1.4, \
        f"NS singular values out of band: [{sv.min():.3f}, {sv.max():.3f}]"
    mean_sv = sv.mean().item()
    assert 0.8 < mean_sv < 1.2, f"NS mean singular value {mean_sv:.3f} != 1"
    eye_err = (O @ O.T - torch.eye(24)).abs().max().item()
    assert eye_err < 0.6, f"NS far from orthogonal: |OO^T - I| max {eye_err:.3e}"
    # Condition number is dramatically improved vs the raw matrix.
    assert sv.max() / sv.min() < torch.linalg.svdvals(A).max() / \
        torch.linalg.svdvals(A).min(), "NS did not improve conditioning"
    # Tall/wide matrices are handled via the transposed iteration.
    for shape in [(32, 12), (12, 32)]:
        B = torch.randn(*shape)
        OB = zeropower_via_newtonschulz5(B)
        gram = OB @ OB.T if shape[0] <= shape[1] else OB.T @ OB
        e = (gram - torch.eye(min(shape))).abs().max().item()
        assert e < 0.6, f"NS({shape}) gram error {e:.3e}"

    # 2) Convergence: Muon vs plain SGD on a least-squares problem. Muon's
    #    update norm is ~constant (orthogonalized momentum), so like all
    #    fixed-step methods it needs a decaying schedule to reach a tight
    #    optimum; use a linear decay (standard practice in the Muon recipe).
    X = torch.randn(64, 16)
    W_true = torch.randn(16, 20)
    Y = X @ W_true

    def run(opt_cls, steps=300, decay=False, **kw):
        W = torch.zeros(16, 20, requires_grad=True)
        opt = opt_cls([W], **kw)
        for t in range(steps):
            if decay:
                for gparam in opt.param_groups:
                    gparam["lr"] = kw.get("lr", 0.02) * (1 - t / steps)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(X @ W, Y)
            loss.backward()
            opt.step()
        return loss.item()

    sgd_final = run(torch.optim.SGD, lr=0.05, momentum=0.9)
    muon_final = run(Muon, lr=0.1, decay=True)
    assert muon_final < sgd_final, \
        f"Muon ({muon_final:.3e}) should beat SGD ({sgd_final:.3e}) here"

    # 3) Mixed model: 2-D params on the Muon path, 1-D on AdamW fallback.
    lin = nn.Linear(16, 20)
    norm = nn.LayerNorm(20)
    opt = Muon(list(lin.parameters()) + list(norm.parameters()), lr=0.01)
    x = torch.randn(8, 16)
    named = [("lin." + n, p) for n, p in lin.named_parameters()] + \
            [("norm." + n, p) for n, p in norm.named_parameters()]
    before = {n: p.detach().clone() for n, p in named}
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(norm(lin(x)), torch.randn(8, 20))
        loss.backward()
        opt.step()
    after = {n: p for n, p in named}
    changed = {n: (after[n] - before[n]).abs().max().item() for n in after}
    assert all(torch.isfinite(p).all() for p in after.values())
    assert changed["lin.weight"] > 0, "Muon path did not update the Linear weight"
    assert changed["lin.bias"] > 0, "AdamW fallback did not update the bias"
    assert changed["norm.weight"] > 0, "AdamW fallback did not update the norm gain"

    # 4) muon=False routes a 2-D param to the AdamW path.
    w = torch.nn.Parameter(torch.randn(8, 8))
    opt2 = Muon([{"params": [w], "muon": False}], lr=0.01)
    for _ in range(2):
        opt2.zero_grad()
        (w ** 2).sum().backward()
        opt2.step()
    assert torch.isfinite(w).all() and opt2.state[w]["step"] == 2, \
        "muon=False group did not take the AdamW path"

    _pass("test_muon",
          f"NS orthog err {eye_err:.2e}, quadratic: muon {muon_final:.2e} "
          f"vs sgd {sgd_final:.2e}, fallback + routing OK")


def test_per_head_muon():
    from evastar.src.training.muon import (
        Muon, zeropower_via_newtonschulz5)

    torch.manual_seed(701)
    n_heads, head_dim, cols = 4, 8, 16
    # 1) Exact per-head semantics: the optimizer update equals a manual
    #    per-head Newton-Schulz of the (nesterov) momentum.
    p1 = torch.nn.Parameter(torch.zeros(n_heads * head_dim, cols))
    g = torch.randn(n_heads * head_dim, cols)
    opt = Muon([p1], lr=0.1, momentum=0.9, per_head_dim=head_dim)
    p1.grad = g.clone()
    opt.step()
    # First step: buf = g, d = g + 0.9*g = 1.9*g.
    d = 1.9 * g
    manual = torch.stack([
        zeropower_via_newtonschulz5(d[i * head_dim:(i + 1) * head_dim])
        for i in range(n_heads)])
    scale = max(1.0, head_dim / cols) ** 0.5
    want = -0.1 * scale * manual.reshape(n_heads * head_dim, cols)
    assert torch.allclose(p1.data, want, atol=1e-6), (
        f"per-head update mismatch: {(p1.data - want).abs().max().item():.3e}")

    # 2) Per-head blocks are independently (approximately) orthogonalized —
    #    each head's singular values sit in the documented NS band.
    sv_bands = []
    for i in range(n_heads):
        sv = torch.linalg.svdvals(manual[i])
        sv_bands.append((sv.min().item(), sv.max().item()))
    assert all(0.25 < lo and hi < 1.4 for lo, hi in sv_bands), (
        f"per-head NS singular values out of band: {sv_bands}")

    # 3) Convergence with per-head routing (quadratic toward a target).
    W = torch.zeros(n_heads * head_dim, cols, requires_grad=True)
    tgt = torch.randn(n_heads * head_dim, cols)
    opt2 = Muon([W], lr=0.1, per_head_dim=head_dim)
    steps = 200
    for t in range(steps):
        for gp in opt2.param_groups:
            gp["lr"] = 0.1 * (1 - t / steps)
        opt2.zero_grad()
        loss = ((W - tgt) ** 2).sum()
        loss.backward()
        opt2.step()
    err = ((W - tgt) ** 2).sum().item() / (tgt ** 2).sum().item()
    assert err < 0.2, f"per-head Muon did not converge: rel err {err:.3f}"

    # 4) Loud errors: a row count not divisible by per_head_dim raises
    #    (silently falling back to whole-matrix NS would hide a layout
    #    bug); non-positive per_head_dim is rejected at construction.
    p2 = torch.nn.Parameter(torch.randn(10, cols))
    opt3 = Muon([p2], lr=0.01, per_head_dim=head_dim)  # 10 % 8 != 0
    p2.grad = torch.randn_like(p2)
    _expect_raises(ValueError, opt3.step,
                   "per_head_dim not dividing rows")
    _expect_raises(ValueError, lambda: Muon([p2], per_head_dim=0),
                   "per_head_dim=0")
    _pass("test_per_head_muon",
          f"manual per-head NS exact (atol 1e-6), bands OK, converge rel "
          f"err {err:.3f}, guards OK")
