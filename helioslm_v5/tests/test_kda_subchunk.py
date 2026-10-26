"""T136 — sub-chunked KDA chunk form, v5.140.

On an RTX 4060 the per-channel chunked delta rule spent ~41% of GPU time
building one fp32 [B, H, n, n, Dk] pair-decay tensor per chunk (k3_block
at 3% MFU). v5.140 computes the state-free parts of all chunks at once and
factors pairs across sub-chunks through plain matmuls.

Pins:

1. the per-channel form equals the per-token loop in float64 for sub-chunk
   sizes 1..>n, ragged chunk/sub-chunk splits, with padding (keep), packed
   document starts and a carried state;
2. gradients equal the loop's in float64;
3. clamp-floor decays (1e-6) stay finite across sub-chunk boundaries;
4. fp32 at a realistic shape matches the loop.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_kda_subchunk.py
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.src.attention.linear_attention import chunked_delta_rule  # noqa: E402
from helioslm_v5.tests.test_kda_chunked import _inputs, _loop  # noqa: E402


@pytest.mark.parametrize("sub", [1, 3, 5, 16, 64])
@pytest.mark.parametrize("chunk", [32, 50])
@pytest.mark.parametrize("use_keep,use_seg,carry", [
    (False, False, False), (True, False, True), (False, True, False),
    (True, True, False)])
def test_subchunk_equals_loop_fp64(sub, chunk, use_keep, use_seg, carry):
    args = _inputs(True, use_keep, use_seg, carry)
    o1, s1 = _loop(*args)
    o2, s2 = chunked_delta_rule(*args, chunk, sub=sub)
    assert (o1 - o2).abs().max() < 1e-10
    assert (s1 - s2).abs().max() < 1e-10


def test_gradients_match_loop_fp64():
    q, k, v, decay, keep, seg, state = _inputs(True, True, True, False, T=48)
    grads = []
    for run in (lambda *a: _loop(*a), lambda *a: chunked_delta_rule(*a, 16, sub=5)):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v, decay)]
        o, s = run(*leaves, keep, seg, state)
        (o.sin().sum() + s.cos().sum()).backward()
        grads.append([t.grad for t in leaves])
    for g1, g2 in zip(*grads):
        assert (g1 - g2).abs().max() < 1e-9


def test_tiny_decay_finite_across_subchunks():
    args = _inputs(True, False, False, True, dtype=torch.float32, low=1e-6)
    o1, s1 = _loop(*args)
    o2, s2 = chunked_delta_rule(*args, 64, sub=4)
    assert torch.isfinite(o2).all() and torch.isfinite(s2).all()
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)


def test_fp32_realistic_shape():
    args = _inputs(True, True, True, True, B=2, H=2, T=200, D=32,
                   dtype=torch.float32)
    o1, s1 = _loop(*args)
    o2, s2 = chunked_delta_rule(*args, 64)
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s1, s2, atol=1e-4, rtol=1e-4)
