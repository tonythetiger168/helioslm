"""T92 — roadmap Phase B: CSA (compress, then top-k select) and the
DeepSeek-V4-style attention stack preset (v5.58).

Pins:

1. reduction: ratio=1 with top-k covering every past token equals dense
   causal attention (SDPA) on the same weights;
2. the vectorized module equals a slow per-query reference (pool each
   chunk, score with the head-mean indexer, take top-k, attend over the
   picks + the query's own chunk) at ratio 4 / top-2, with padding;
3. causality: perturbing position p never changes outputs before p, and
   no query ever selects its own or a future chunk;
4. pads are excluded from pooling, selection and the local set;
5. gradients reach qkv/out but not the indexer (top-k is discrete);
6. the v4_attention_stack preset: layer pattern, IndexShare grouping,
   parameter saving, shared == copied-weights outputs, validation;
7. the committed A/B artifact: a recomputed slice matches it and the
   artifact is registered.

Run from repo root: python3 evastar/tests/test_csa.py
"""
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from evastar.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from evastar.src.attention.kv_compress import (  # noqa: E402
    CompressedSparseAttention, HighCompressionAttention, v4_attention_stack)

D, H = 32, 2


def _csa(**kw):
    torch.manual_seed(3)
    return CompressedSparseAttention(D, H, **kw).eval()


def test_ratio1_equals_dense():
    m = _csa(ratio=1, topk=64)
    x = torch.randn(2, 13, D)
    with torch.no_grad():
        got = m(x)
        q, k, v = m.qkv(x).view(2, 13, 3, H, D // H).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True)
        want = m.out(att.transpose(1, 2).reshape(2, 13, D))
    diff = (got - want).abs().max().item()
    assert diff < 1e-5, diff
    print(f"PASS test_ratio1_equals_dense (max|diff|={diff:.1e})")


def _reference(m, x, mask):
    """Slow per-query CSA: the spec written out loop by loop."""
    B, L, _ = x.shape
    r, kmax, dh = m.ratio, m.topk, D // H
    q, k, v = m.qkv(x).view(B, L, 3, H, dh).unbind(2)
    idx = m.indexer(x).view(B, L, H, dh)
    out = torch.zeros(B, L, H, dh)
    for b in range(B):
        chunks = [list(range(s, min(s + r, L))) for s in range(0, L, r)]
        entries = []
        for ch in chunks:
            real = [j for j in ch if mask[b, j]]
            if real:
                entries.append((k[b, real].mean(0), v[b, real].mean(0)))
            else:
                entries.append(None)
        for i in range(L):
            c = i // r
            past = [s for s in range(c) if entries[s] is not None]
            scores = {s: float((idx[b, i] * entries[s][0]).sum(-1).mean()
                               / math.sqrt(dh)) for s in past}
            picks = sorted(past, key=lambda s: -scores[s])[:kmax]
            keys = [entries[s][0] for s in picks]
            vals = [entries[s][1] for s in picks]
            for j in chunks[c]:
                if j <= i and mask[b, j]:
                    keys.append(k[b, j])
                    vals.append(v[b, j])
            if not keys:
                continue
            kk, vv = torch.stack(keys), torch.stack(vals)        # (S, H, dh)
            p = torch.softmax(torch.einsum("hd,shd->hs", q[b, i], kk)
                              / math.sqrt(dh), -1)
            out[b, i] = torch.einsum("hs,shd->hd", p, vv)
    return m.out(out.reshape(B, L, D))


def test_matches_slow_reference():
    m = _csa(ratio=4, topk=2)
    x = torch.randn(2, 18, D)
    mask = torch.ones(2, 18, dtype=torch.long)
    mask[1, :3] = 0                       # left padding on row 1
    with torch.no_grad():
        got = m(x, attention_mask=mask)
        want = _reference(m, x, mask)
    real = mask.bool()
    diff = (got - want)[real].abs().max().item()
    assert diff < 1e-5, diff
    print(f"PASS test_matches_slow_reference (max|diff|={diff:.1e})")


def test_causality_and_selection():
    m = _csa(ratio=4, topk=2)
    x = torch.randn(1, 20, D)
    with torch.no_grad():
        base, picks = m(x, return_selection=True)
        for p in (5, 11, 19):
            x2 = x.clone()
            x2[0, p] += 3.0
            out2 = m(x2)
            assert torch.equal(out2[0, :p], base[0, :p]), p
    chunk_of_query = torch.arange(20) // 4
    for i in range(20):
        sel = [s for s in picks[0, i].tolist() if s >= 0]
        assert all(s < chunk_of_query[i] for s in sel), (i, sel)
    print("PASS test_causality_and_selection")


def test_padding_excluded():
    m = _csa(ratio=4, topk=3)
    x = torch.randn(1, 16, D)
    mask = torch.ones(1, 16, dtype=torch.long)
    mask[0, 4:6] = 0
    with torch.no_grad():
        a = m(x, attention_mask=mask)
        x2 = x.clone()
        x2[0, 4:6] = 100.0                # garbage in padded positions
        b = m(x2, attention_mask=mask)
    real = mask[0].bool()
    assert torch.allclose(a[0, real], b[0, real], atol=1e-6)
    assert torch.isfinite(b).all()
    print("PASS test_padding_excluded")


def test_gradients():
    m = _csa(ratio=4, topk=2).train()
    m(torch.randn(2, 16, D)).sum().backward()
    assert m.qkv.weight.grad.abs().sum() > 0
    assert m.out.weight.grad.abs().sum() > 0
    assert m.indexer.weight.grad is None or \
        m.indexer.weight.grad.abs().sum() == 0
    print("PASS test_gradients (no gradient through top-k selection)")


def test_v4_stack_preset():
    torch.manual_seed(0)
    stack = v4_attention_stack(D, 6, H, pattern="CH", share_indexer_every=2)
    kinds = [type(layer) for layer in stack]
    assert kinds == [CompressedSparseAttention, HighCompressionAttention] * 3
    csa = [layer for layer in stack if isinstance(layer, CompressedSparseAttention)]
    assert csa[0].indexer is csa[1].indexer
    assert csa[2].indexer is not csa[0].indexer
    assert all(layer.rescue_topk == 0 for layer in stack
               if isinstance(layer, HighCompressionAttention))

    unshared = v4_attention_stack(D, 6, H, pattern="CH")
    n = lambda mods: sum(p.numel() for p in nn.ModuleList(mods).parameters())  # noqa: E731
    assert n(unshared) - n(stack) == D * D + D   # one indexer saved

    # shared indexer == a private copy with the same weights
    twin = copy.deepcopy(csa[1])
    twin.indexer = copy.deepcopy(csa[1].indexer)
    x = torch.randn(1, 12, D)
    with torch.no_grad():
        assert torch.equal(csa[1].eval()(x), twin.eval()(x))

    for bad in (dict(pattern=""), dict(pattern="CX"),
                dict(share_indexer_every=0)):
        try:
            v4_attention_stack(D, 2, H, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    for bad in (dict(ratio=0), dict(topk=0),
                dict(shared_indexer=nn.Linear(D, D + 1))):
        try:
            CompressedSparseAttention(D, H, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    stats = csa[0].get_compression_stats(1024)
    assert stats["compression_vs_dense"] == 4.0
    assert stats["entry_values_per_token"] * 4 == stats["dense_values_per_token"]
    print("PASS test_v4_stack_preset")


def test_committed_ab_slice():
    from csa_ab_passkey import (  # noqa: E402
        _wire_csa, probe_accuracy, selection_hit_rate)
    art = json.loads((_REPO / "benchmarks" / "csa_ab_2026-10-15.json").read_text())
    p = art["params"]
    block = next(b for b in art["results"] if b["seq_len"] == 512)

    def make():
        return _wire_csa(CompressedSparseAttention(
            p["dim"], n_heads=p["heads"], ratio=4, topk=8), p["dim"])

    class Args:
        trials, pairs, dim = p["trials"], p["pairs"], p["dim"]
        amp, boost, seed = p["amp"], p["boost"], p["seed"]

    acc = probe_accuracy(make, 512, p["pairs"], p["dim"], p["amp"],
                         p["boost"], p["trials"], p["seed"])
    assert acc == block["accuracy"]["csa-4x@k8"], acc
    hit = selection_hit_rate(make, 4, 512, Args)
    assert hit == block["selection_hit_rate"]["csa-4x@k8"], hit
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of("csa_ab_2026-10-15.json", reg) == "measured"
    print("PASS test_committed_ab_slice")


if __name__ == "__main__":
    test_ratio1_equals_dense()
    test_matches_slow_reference()
    test_causality_and_selection()
    test_padding_excluded()
    test_gradients()
    test_v4_stack_preset()
    test_committed_ab_slice()
