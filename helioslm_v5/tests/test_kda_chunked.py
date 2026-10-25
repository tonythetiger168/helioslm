"""v5.93: chunked-parallel KDA scan (hybrid_attention.chunk_size).

The chunked form must compute the same recurrence as the token loop; only
the float summation order changes. Pins, in float64 so differences are
pure reassociation noise:
1. outputs and final state match the loop for per-head and per-channel
   decay, with padding, a carried initial state, and T not a multiple of C;
2. gradients w.r.t. the input and every parameter match;
3. a chunk whose cumulative decay would overflow is split and still exact;
4. a whole k3_block model gives the same logits with and without chunking,
   and chunked prefill followed by cached decode matches the loop;
5. packed sequences (position restarts) still use the exact loop;
6. invalid chunk sizes are rejected by the config.
"""
import copy

import pytest
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config, k3_block_config
from helioslm_v5.src.attention.linear_attention import GatedDeltaAttention


def _pair(per_channel, chunk, seed=0):
    cfg = HeliosLMv5Config(size="lite")
    cfg.hybrid_attention.per_channel_decay = per_channel
    torch.manual_seed(seed)
    loop = GatedDeltaAttention(cfg).double()
    fast = copy.deepcopy(loop)
    fast.chunk_size = chunk
    return cfg, loop, fast


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("chunk", [1, 8, 64])
def test_forward_parity(per_channel, chunk):
    cfg, loop, fast = _pair(per_channel, chunk)
    B, T = 2, 77
    x = torch.randn(B, T, cfg.hidden_size, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, :9] = 0                                    # left padding
    mask[0, 40:43] = 0                                 # holes
    H, D = loop.num_heads, loop.head_dim
    s0 = torch.randn(B, H, D, D, dtype=torch.float64) * 0.1
    with torch.no_grad():
        y0, (st0,) = loop(x, attention_mask=mask, past_key_value=(s0,), use_cache=True)
        y1, (st1,) = fast(x, attention_mask=mask, past_key_value=(s0,), use_cache=True)
    keep = mask.bool()
    assert (y0 - y1)[keep].abs().max() < 1e-10
    assert (st0 - st1).abs().max() < 1e-10


@pytest.mark.parametrize("per_channel", [False, True])
def test_gradient_parity(per_channel):
    cfg, loop, fast = _pair(per_channel, 16, seed=1)
    x = torch.randn(2, 50, cfg.hidden_size, dtype=torch.float64)
    xs = [x.clone().requires_grad_(True) for _ in range(2)]
    for m, xi in zip((loop, fast), xs):
        y, _ = m(xi)
        (y * torch.linspace(-1, 1, y.shape[-1], dtype=y.dtype)).sum().backward()
    assert (xs[0].grad - xs[1].grad).abs().max() < 1e-9
    for (n, p0), p1 in zip(loop.named_parameters(), fast.parameters()):
        assert (p0.grad - p1.grad).abs().max() < 1e-9, n


def test_extreme_decay_splits_chunk():
    cfg, loop, fast = _pair(True, 64, seed=2)
    with torch.no_grad():                              # decay ~ 1e-6 per token
        loop.decay_bias.fill_(-30.0)
        fast.decay_bias.fill_(-30.0)
    calls = []
    orig = fast._chunk
    fast._chunk = lambda *a: (calls.append(a[0].shape[2]), orig(*a))[1]
    x = torch.randn(1, 64, cfg.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        y0, (s0,) = loop(x, use_cache=True)
        y1, (s1,) = fast(x, use_cache=True)
    assert min(calls) < 64                             # the chunk was split
    assert torch.isfinite(y1).all()
    assert (y0 - y1).abs().max() < 1e-10 and (s0 - s1).abs().max() < 1e-10


def _k3(chunk):
    cfg = k3_block_config()
    cfg.hybrid_attention.chunk_size = chunk
    cfg._validate()
    return cfg


def test_model_parity_and_decode():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(3)
    loop = HeliosLMv5(_k3(0)).double().eval()
    fast = HeliosLMv5(_k3(16)).double().eval()
    fast.load_state_dict(loop.state_dict())
    ids = torch.randint(0, loop.config.vocab_size, (2, 45))
    with torch.no_grad():
        a = loop(ids)[0]
        b = fast(ids)[0]
        assert (a - b).abs().max() < 1e-8
        # chunked prefill of 40 tokens, then 5 cached decode steps
        logits, _, past = fast(ids[:, :40], use_cache=True)
        steps = [logits[:, -1]]
        for t in range(40, 45):
            logits, _, past = fast(ids[:, t:t + 1], past_key_values=past,
                                   use_cache=True)
            steps.append(logits[:, -1])
    assert (a[:, 39:45] - torch.stack(steps, dim=1)).abs().max() < 1e-8


def test_packed_sequences_use_loop():
    cfg, loop, fast = _pair(True, 8, seed=4)
    x = torch.randn(1, 20, cfg.hidden_size, dtype=torch.float64)
    pos = torch.cat([torch.arange(12), torch.arange(8)])[None]
    fast._chunk = None                                 # must not be reached
    with torch.no_grad():
        y0, _ = loop(x, position_ids=pos)
        y1, _ = fast(x, position_ids=pos)
    assert torch.equal(y0, y1)


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_config_rejects_bad_chunk(bad):
    cfg = k3_block_config()
    cfg.hybrid_attention.chunk_size = bad
    with pytest.raises(ValueError, match="chunk_size"):
        cfg._validate()
