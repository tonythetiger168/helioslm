import math
import pytest
import torch
from helioslm_v5.src.dca_oracle import CrossLayerAttention


def test_scalar_oracle_and_gradients():
    torch.manual_seed(42)
    m = CrossLayerAttention(4).double()
    x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    history = [torch.randn_like(x).requires_grad_() for _ in range(3)]
    actual = m(x, history)
    rows = []
    for b in range(2):
        tokens = []
        for t in range(3):
            q = m.query(x[b, t])
            keys = [m.key(h[b, t]) for h in history]
            scores = torch.stack([torch.dot(q, k) / math.sqrt(4) for k in keys])
            weights = scores.softmax(0)
            tokens.append(sum(w * m.value(h[b, t]) for w, h in zip(weights, history)))
        rows.append(torch.stack(tokens))
    expected = torch.stack(rows)
    torch.testing.assert_close(actual, expected)
    inputs = [x, *history, *m.parameters()]
    a = torch.autograd.grad(actual.square().sum(), inputs)
    b = torch.autograd.grad(expected.square().sum(), inputs)
    for lhs, rhs in zip(a, b):
        torch.testing.assert_close(lhs, rhs)


def test_prefix_and_invalid_history():
    m = CrossLayerAttention(4)
    x = torch.randn(2, 5, 4)
    history = [torch.randn_like(x) for _ in range(2)]
    torch.testing.assert_close(m(x, history)[:, :2],
                               m(x[:, :2], [h[:, :2] for h in history]))
    with pytest.raises(ValueError):
        m(x, [])
    with pytest.raises(ValueError):
        m(x, [x[:, :1]])
