import copy

import pytest
import torch
from torch import nn

from helioslm_v5.src.shared_depth_oracle import SharedDepth


class ToyCausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden, *, cache=None, use_cache=False):
        values = self.proj(hidden)
        prefix = torch.cat((cache, values), dim=1) if cache is not None else values
        result = hidden + prefix.cumsum(dim=1)[:, -hidden.shape[1]:]
        return result, prefix if use_cache else None


@pytest.mark.parametrize("repeats", [1, 2])
def test_manual_unroll_and_gradient(repeats):
    torch.manual_seed(298)
    block = ToyCausalBlock().double()
    reference = copy.deepcopy(block)
    model = SharedDepth(block, repeats)
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    actual, _ = model(x)
    expected = x
    for _ in range(repeats):
        expected, _ = reference(expected)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(block.proj.weight.grad, reference.proj.weight.grad)
    assert sum(p.numel() for p in model.parameters()) == 16


def test_incremental_cache_and_request_isolation():
    torch.manual_seed(298)
    model = SharedDepth(ToyCausalBlock().double())
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    full, _ = model(x)
    cache = None
    outputs = []
    for token in x.split(1, dim=1):
        out, cache = model(token, cache, use_cache=True)
        outputs.append(out)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
    assert cache[0].data_ptr() != cache[1].data_ptr()
    fresh, _ = model(x[:, :1])
    torch.testing.assert_close(fresh, full[:, :1])
    with pytest.raises(ValueError):
        model(x, (cache[0],), use_cache=True)
    with pytest.raises(ValueError):
        model(x, cache)


def test_invalid_repeat_count():
    for repeats in (0, 3, True, 1.5):
        with pytest.raises(ValueError):
            SharedDepth(ToyCausalBlock(), repeats)
