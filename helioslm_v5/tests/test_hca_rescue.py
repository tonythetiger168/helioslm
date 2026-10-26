"""T77 - B1-2: HCA high-compression + sparse top-k rescue oracles (B1-2).

HighCompressionAttention runs HCA at production-class compression ratios
(default chunk=128, i.e. 128x KV compression) with an optional sparse
top-k rescue of exact tokens from the compressed region. The
non-negotiable property is the same as the v5.31 modules: CAUSALITY —
perturbing any position p must not change outputs at positions < p,
across chunk boundaries, through the pooled memory slots, AND through
the rescue selection (which scores only strictly-past chunks).

Run from repo root: python3 helioslm_v5/tests/test_hca_rescue.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.attention.kv_compress import HighCompressionAttention


def _make(dim=16, n_heads=2, window=4, chunk=8, rescue_topk=3):
    torch.manual_seed(3)
    return HighCompressionAttention(dim, n_heads=n_heads, window=window,
                                    chunk=chunk, rescue_topk=rescue_topk)


def test_shapes_and_grads():
    for kw in ({"rescue_topk": 0}, {"rescue_topk": 3}):
        m = _make(**kw)
        x = torch.randn(2, 26, 16, requires_grad=True)  # spans 4 chunks
        out = m(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        out.pow(2).sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
    print("PASS test_shapes_and_grads")


def test_causality_across_chunks_and_rescue():
    dim, w, chunk = 16, 4, 8
    for kw in ({"rescue_topk": 0}, {"rescue_topk": 3}, {"rescue_topk": 1}):
        m = _make(dim=dim, window=w, chunk=chunk, **kw).eval()
        L = 3 * chunk + 2                            # spans 4 chunks
        x = torch.randn(1, L, dim)
        with torch.no_grad():
            base = m(x)
        for p in (0, w - 1, chunk - 1, chunk, 2 * chunk + 1, L - 1):
            x2 = x.clone()
            x2[0, p] = torch.randn(dim) * 5
            with torch.no_grad():
                out2 = m(x2)
            diff = (out2 - base).abs().amax(dim=(0, 2))
            changed = torch.nonzero(diff > 1e-6).flatten().tolist()
            assert all(c >= p for c in changed), \
                f"{kw}: future leak at pos {p}: changed {changed}"
            # sanity: the perturbed position itself must react
            assert diff[p] > 1e-6, f"{kw}: pos {p} did not react"
    print("PASS test_causality_across_chunks_and_rescue")


def test_rescue_zero_is_pure_hca():
    m = _make(rescue_topk=0)
    assert not hasattr(m, "rescue_indexer"), \
        "rescue_topk=0 must not create indexer parameters"
    assert not any("rescue" in k for k in m.state_dict())
    m2 = _make(rescue_topk=4)
    assert any("rescue_indexer" in k for k in m2.state_dict())
    print("PASS test_rescue_zero_is_pure_hca")


def test_rescue_recovers_compressed_token():
    """Behavioral oracle: craft an identity indexer, plant a distinctive
    token deep in the compressed past and a matching query two chunks
    later — the rescue arm (sharing qkv/out weights with the no-rescue
    arm) MUST produce a different output at the matched query, and the
    whole forward must be bitwise deterministic across two runs."""
    torch.manual_seed(5)
    dim, w, chunk = 16, 4, 8
    m0 = HighCompressionAttention(dim, 2, w, chunk, rescue_topk=0).eval()
    m1 = HighCompressionAttention(dim, 2, w, chunk, rescue_topk=2).eval()
    m1.qkv.load_state_dict(m0.qkv.state_dict())
    m1.out.load_state_dict(m0.out.state_dict())
    with torch.no_grad():
        m1.rescue_indexer.weight.copy_(torch.eye(dim))
        m1.rescue_indexer.bias.zero_()
    torch.manual_seed(6)
    x = torch.randn(1, 24, dim) * 0.1
    x[0, 3] = torch.ones(dim)      # distinctive token in the compressed past
    x[0, 20] = torch.ones(dim)     # matching query (chunk 2)
    with torch.no_grad():
        d = (m1(x) - m0(x))[0, 20].abs().max().item()
        assert d > 0, "rescue did not change the matched query's output"
        # determinism: identical inputs -> bitwise identical outputs
        assert torch.equal(m1(x), m1(x)), "rescue forward is not deterministic"
    print("PASS test_rescue_recovers_compressed_token")


def test_padding_excluded_bitwise():
    """Masked positions must contribute EXACTLY nothing: same real tokens
    with garbage in the padded tail vs zeros must give bitwise-identical
    outputs at real positions (pooling multiplies by 0, scores get -inf,
    exp(-inf) == 0 exactly)."""
    m = _make(rescue_topk=3).eval()
    torch.manual_seed(7)
    x = torch.randn(1, 20, 16)
    mask = torch.ones(1, 20, dtype=torch.long)
    mask[0, 17:] = 0
    x_dirty = x.clone()
    x_dirty[0, 17:] = 1e6                        # garbage in the pads
    with torch.no_grad():
        o_clean = m(x, attention_mask=mask)
        o_dirty = m(x_dirty, attention_mask=mask)
    assert torch.equal(o_clean[:, :17], o_dirty[:, :17]), \
        "masked positions leaked into real outputs"
    assert torch.isfinite(o_dirty).all()
    print("PASS test_padding_excluded_bitwise")


def test_compression_stats():
    m = _make(chunk=128, rescue_topk=4)
    st = m.get_compression_stats(1024)
    assert st["compression_vs_dense"] == 128.0
    assert st["rescue_values_per_query"] == 4 * 2 * 2 * 8
    for bad in (0, -1, 1.5):
        try:
            m.get_compression_stats(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad seq_len accepted: {bad}")
    print("PASS test_compression_stats")


ALL_TESTS = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} hca-rescue tests passed")
