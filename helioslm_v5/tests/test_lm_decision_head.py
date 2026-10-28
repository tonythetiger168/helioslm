"""v5.41 — LM-backed DecisionHead oracles (Phase 1.2).

Tests the thin LM-side readout in helioslm_v5/src/decision_head.py (NOT the
toy-scale agent/decision_head.py — that one has its own test file).

Run from repo root: python helioslm_v5/tests/test_lm_decision_head.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.decision_head import DecisionHead


def test_output_shapes_and_leading_dims():
    h = DecisionHead(hidden_size=64, num_choices=7)
    for shape in [(64,), (8, 64), (8, 12, 64)]:
        out = h(torch.randn(*shape))
        assert out["noul"].shape == shape[:-1], (shape, out["noul"].shape)
        assert out["score"].shape == shape[:-1], (shape, out["score"].shape)
        assert out["choice"].shape == shape[:-1] + (7,), (
            shape, out["choice"].shape)
    print("PASS test_output_shapes_and_leading_dims "
          "[H], [B,H], [B,T,H] all map correctly")


def test_probability_validity():
    torch.manual_seed(0)
    h = DecisionHead(hidden_size=32, num_choices=5)
    out = h(torch.randn(256, 32))
    assert out["noul"].min() > 0.0 and out["noul"].max() < 1.0
    s = out["choice"].sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6), s[:4]
    assert (out["choice"] >= 0).all()
    print("PASS test_probability_validity noul in (0,1), choice on simplex")


def test_loud_errors():
    h = DecisionHead(hidden_size=16, num_choices=3)
    for bad_ctor in (lambda: DecisionHead(0, 3),
                     lambda: DecisionHead(16, 1),
                     lambda: DecisionHead(16, 0)):
        try:
            bad_ctor()
        except ValueError:
            pass
        else:
            raise AssertionError("constructor accepted an invalid config")
    try:
        h(torch.randn(4, 15))          # last-dim mismatch
    except ValueError:
        pass
    else:
        raise AssertionError("hidden-size mismatch did not fail loudly")
    print("PASS test_loud_errors bad sizes and wrong hidden dim all raise")


def test_gradients_flow_to_all_heads():
    torch.manual_seed(1)
    h = DecisionHead(hidden_size=24, num_choices=4)
    hidden = torch.randn(6, 24, requires_grad=True)
    out = h(hidden)
    loss = (out["noul"].sum() + out["choice"].sum() + out["score"].sum())
    loss.backward()
    for name in ("noul", "choice", "score"):
        w = getattr(h, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, name
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    print("PASS test_gradients_flow_to_all_heads all three heads + input")


def test_noul_learns_binary_target():
    """Smoke wiring check: BCE on a separable synthetic target must
    converge far below the 0.693 random-guess floor."""
    torch.manual_seed(2)
    h = DecisionHead(hidden_size=16, num_choices=3)
    xs = torch.randn(400, 16)
    ys = (xs[:, 0] > 0).float()
    opt = torch.optim.Adam(h.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        p = h(xs)["noul"]
        loss = torch.nn.functional.binary_cross_entropy(p, ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((h(xs)["noul"] > 0.5).float() == ys).float().mean()
        final_bce = float(loss.detach())
    assert acc > 0.95, f"noul acc {acc:.3f} did not converge"
    assert final_bce < 0.3, f"final BCE {final_bce:.3f}"
    print(f"PASS test_noul_learns_binary_target acc={float(acc):.3f} "
          f"bce={final_bce:.4f}")


if __name__ == "__main__":
    test_output_shapes_and_leading_dims()
    test_probability_validity()
    test_loud_errors()
    test_gradients_flow_to_all_heads()
    test_noul_learns_binary_target()
    print("\n5/5 LM DecisionHead tests passed")
