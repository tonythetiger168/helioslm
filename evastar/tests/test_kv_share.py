"""T109 — roadmap U6: cross-layer KV sharing (Gemma 3n/4), v5.71.

Pins:

1. defaults unchanged: kv_share_last_n=0 is bit-identical (state dict and
   output) and no layer shares;
2. structure: the last N MLA layers (hybrid GDA layers not counted) take
   their KV from the most recent non-shared MLA layer of the same window
   type; shared layers own no K/V projections (no parameters, no
   state-dict keys) and every kept weight equals the unshared model's;
3. oracle: a shared layer's output equals a hand-written causal softmax
   attention of its own queries over the source's expanded K/V, in both
   the absorbed and expanded paths;
4. cache: a shared layer's cache entry IS its source's tuple, and the
   unique cached bytes are exactly (L_mla - N) / L_mla of the unshared
   model's, absorbed and expanded;
5. cached decode == full forward and absorbed == expanded (plain, sliding/
   global interleave, hybrid, hyper-connections); the source's KV
   projections receive gradient from the shared layers;
6. validation;
7. the committed perplexity A/B artifact (ratios exact, hash).

Run from repo root: python3 evastar/tests/test_kv_share.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import (  # noqa: E402
    AttentionConfig, EvastarV5Config, HybridAttentionConfig)
from evastar.src.attention.mla import MLA  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

_ART = _REPO / "benchmarks" / "kv_share_ab_2026-10-16.json"
_KV_MODULES = ("kv_a_proj", "kv_b_proj", "k_rope_proj", "norm_kv")


def _cfg(n=0, layers=4, hybrid=False, **kw):
    extra = {}
    if "use_hyper_connections" in kw:
        extra["use_hyper_connections"] = kw.pop("use_hyper_connections")
        extra["use_attention_residuals"] = False
    return EvastarV5Config(
        size="lite", num_hidden_layers=layers,
        attention=AttentionConfig(kv_share_last_n=n, **kw),
        hybrid_attention=HybridAttentionConfig(
            enabled=hybrid, full_attention_every=2), **extra)


def _unique_bytes(past):
    seen, total = set(), 0
    for entry in past:
        for t in entry:
            ptr = t.untyped_storage().data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += t.untyped_storage().nbytes()
    return total


def _decode_matches(model, ids, split=5):
    with torch.no_grad():
        full = model(ids)[0]
        lg, _, past = model(ids[:, :split], use_cache=True)
        outs = [lg]
        for t in range(split, ids.shape[1]):
            lg, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
            outs.append(lg)
    return (torch.cat(outs, 1) - full).abs().max().item(), full, past


def test_defaults_unchanged():
    ids = torch.randint(3, 1000, (2, 9))
    outs, states = [], []
    for cfg in (EvastarV5Config(size="lite"),
                EvastarV5Config(size="lite",
                                 attention=AttentionConfig(kv_share_last_n=0))):
        torch.manual_seed(0)
        m = EvastarV5(cfg).eval()
        states.append(m.state_dict())
        with torch.no_grad():
            outs.append(m(ids)[0])
        assert m._kv_share_src == {}
    sa, sb = states
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(outs[0], outs[1])
    print("PASS test_defaults_unchanged")


def test_structure():
    torch.manual_seed(0)
    base = EvastarV5(_cfg(0))
    torch.manual_seed(0)
    shared = EvastarV5(_cfg(2))
    assert shared._kv_share_src == {2: 1, 3: 1}
    sb, ss = base.state_dict(), shared.state_dict()
    dropped = [k for k in sb if k not in ss]
    assert dropped and all(
        k.startswith(("layers.2.attention.", "layers.3.attention."))
        and k.split(".")[3] in _KV_MODULES for k in dropped), dropped
    assert all(torch.equal(ss[k], sb[k]) for k in ss)
    n_drop = sum(sb[k].numel() for k in dropped)
    assert sum(p.numel() for p in base.parameters()) - n_drop == \
        sum(p.numel() for p in shared.parameters())
    for j in (2, 3):
        a = shared.layers[j].attention
        assert all(getattr(a, name) is None for name in _KV_MODULES)
        assert a._kv_src is shared.layers[1].attention

    # hybrid: GDA layers are not counted; MLA layers are 0, 1, 3, 5
    torch.manual_seed(0)
    hyb = EvastarV5(_cfg(2, layers=6, hybrid=True))
    assert [i for i, l in enumerate(hyb.layers)
            if isinstance(l.attention, MLA)] == [0, 1, 3, 5]
    assert hyb._kv_share_src == {3: 1, 5: 1}

    # sliding/global: every 4 -> layers 0, 3 global, 1, 2 windowed
    torch.manual_seed(0)
    sw = EvastarV5(_cfg(2, sliding_window=4, global_attention_every=4))
    assert sw._kv_share_src == {2: 1, 3: 0}
    print("PASS test_structure")


def test_oracle_against_manual_attention():
    torch.manual_seed(3)
    src = MLA(_cfg(use_absorption=False), layer_idx=0).eval()
    dsts = {}
    for absorb in (False, True):
        torch.manual_seed(4)
        d = MLA(_cfg(use_absorption=absorb), layer_idx=1).eval()
        d.share_kv_from(src)
        dsts[absorb] = d
    B, L = 2, 7
    x_src = torch.randn(B, L, src.hidden_size)
    x_dst = torch.randn(B, L, src.hidden_size)
    pos = torch.arange(L).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        _, present = src(x_src, position_ids=pos, use_cache=True)
        k_nope, k_rope, v = present
        d = dsts[False]
        q_nope, q_rope = d._project_queries(x_dst, pos)
        scores = (q_nope @ k_nope.transpose(-2, -1)
                  + q_rope @ k_rope.transpose(-2, -1)) * d.softmax_scale
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        probs = F.softmax(scores.masked_fill(~causal, float("-inf")), -1)
        ref = d._project_out((probs @ v).transpose(1, 2).reshape(B, L, -1),
                             x_dst)
        out_exp, pres_exp = d(x_dst, position_ids=pos, shared_kv=present,
                              use_cache=True)
        assert (out_exp - ref).abs().max() < 1e-5
        assert pres_exp is present
        # absorbed consumer over the source's latent cache
        src.use_absorption = True
        _, latent = src(x_src, position_ids=pos, use_cache=True)
        src.use_absorption = False
        out_abs, _ = dsts[True](x_dst, position_ids=pos, shared_kv=latent)
        assert (out_abs - ref).abs().max() < 1e-4, (out_abs - ref).abs().max()
    try:
        d(x_dst, position_ids=pos)
    except ValueError as e:
        assert "shared_kv" in str(e)
    else:
        raise AssertionError("shared layer ran without shared_kv")
    print("PASS test_oracle_against_manual_attention")


def test_cache_aliasing_and_bytes():
    ids = torch.randint(3, 1000, (2, 11))
    for absorb in (True, False):
        sizes = {}
        for n in (0, 2):
            torch.manual_seed(0)
            m = EvastarV5(_cfg(n, use_absorption=absorb)).eval()
            with torch.no_grad():
                _, _, past = m(ids, use_cache=True)
            sizes[n] = _unique_bytes(past)
            if n:
                assert past[2] is past[1] and past[3] is past[1]
        assert sizes[2] * 4 == sizes[0] * 2, sizes          # (4 - 2) / 4
    # the analytic counter agrees
    from benchmarks.bench_cpu import cache_bytes
    a0, a2 = cache_bytes(_cfg(0), 64), cache_bytes(_cfg(2), 64)
    assert a2["kv_shared_layers"] == 2
    assert a2["helioslm_bytes"] * 4 == a0["helioslm_bytes"] * 2
    print("PASS test_cache_aliasing_and_bytes")


def test_decode_paths_and_grad():
    ids = torch.randint(3, 1000, (2, 12))
    cases = [dict(), dict(sliding_window=4, global_attention_every=4),
             dict(layers=6, hybrid=True),
             dict(use_hyper_connections=True)]
    for kw in cases:
        fulls = {}
        for absorb in (True, False):
            if kw.get("hybrid") and not absorb:
                continue
            torch.manual_seed(0)
            m = EvastarV5(_cfg(2, use_absorption=absorb, **kw)).eval()
            err, full, _ = _decode_matches(m, ids)
            assert err < 1e-4, (kw, absorb, err)
            fulls[absorb] = full
        if len(fulls) == 2:
            assert (fulls[True] - fulls[False]).abs().max() < 1e-4, kw
    # the shared layers' loss reaches the source's KV projections
    torch.manual_seed(0)
    m = EvastarV5(_cfg(2)).train()
    m.zero_grad()
    m(ids)[0].logsumexp(-1).mean().backward()
    g_with = m.layers[1].attention.kv_a_proj.weight.grad.clone()
    assert torch.isfinite(g_with).all() and g_with.abs().sum() > 0
    print("PASS test_decode_paths_and_grad")


def test_validation():
    for kw, needle in (
            (dict(kv_share_last_n=-1), "kv_share_last_n"),
            (dict(kv_share_last_n=True), "kv_share_last_n"),
            (dict(kv_share_last_n=1, sparse_top_k=8), "sparse_top_k"),
            (dict(kv_share_last_n=1, kv_cache_dtype="fp8"), "fp8")):
        try:
            EvastarV5Config(size="lite", attention=AttentionConfig(**kw))
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    for cfg, needle in ((_cfg(4), "smaller than"),
                        # windowed layers 1, 2 are both shared: no source
                        (_cfg(3, sliding_window=4, global_attention_every=4),
                         "same window type")):
        try:
            EvastarV5(cfg)
        except ValueError as e:
            assert needle in str(e), e
        else:
            raise AssertionError(needle)
    print("PASS test_validation")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = art["results"]
    L = art["model"]["layers"]
    assert res[0]["kv_share_last_n"] == 0
    for r in res:
        assert r["kv_ratio_vs_first"] == (L - r["kv_share_last_n"]) / L, r
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(res, sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact " + json.dumps(
        {r["kv_share_last_n"]: r["val_ppl"] for r in res}))


if __name__ == "__main__":
    test_defaults_unchanged()
    test_structure()
    test_oracle_against_manual_attention()
    test_cache_aliasing_and_bytes()
    test_decode_paths_and_grad()
    test_validation()
    test_ab_artifact()
