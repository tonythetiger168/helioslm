import pytest
import torch

from helioslm_v5.src.deep_cross_attention_oracle import CrossDepthAttention


def test_reference_and_gradients():
    torch.manual_seed(24)
    m = CrossDepthAttention(4).double()
    x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    h = [torch.randn_like(x, requires_grad=True) for _ in range(3)]
    y = m(x, h)
    rows = []
    for b in range(2):
        tokens = []
        for t in range(3):
            q = m.query(x[b, t])
            keys = torch.stack([m.key(s[b, t]) for s in h])
            values = torch.stack([m.value(s[b, t]) for s in h])
            tokens.append(torch.softmax(keys @ q / 2, dim=0) @ values)
        rows.append(torch.stack(tokens))
    ref = torch.stack(rows)
    torch.testing.assert_close(y, ref)
    inputs = (x, *h, *m.parameters())
    a = torch.autograd.grad(y.square().sum(), inputs, retain_graph=True)
    b = torch.autograd.grad(ref.square().sum(), inputs)
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()


def test_prefix_and_invalid_history():
    torch.manual_seed(24)
    m = CrossDepthAttention(4).double()
    x = torch.randn(2, 5, 4, dtype=torch.double)
    h = [torch.randn_like(x) for _ in range(3)]
    torch.testing.assert_close(m(x, h)[:, :2], m(x[:, :2], [s[:, :2] for s in h]))
    with pytest.raises(ValueError):
        m(x, [])
    with pytest.raises(ValueError):
        m(x, [h[0][:, :2]])
