"""T108 — roadmap U4: Gemma 4 p-RoPE and K=V on global attention layers,
v5.70.

Pins:

1. defaults unchanged: identical state dict and output; global_rope_partial
   1.0 explicitly is bit-identical too;
2. p-RoPE: the kept pairs are the highest frequencies, the rest have
   inv_freq 0, so cos == 1 and sin == 0 there (an exact identity) at every
   position, and the rotated query/key dims beyond the kept pairs equal the
   unrotated ones;
3. only global layers change: with a sliding window + global interleave,
   windowed layers keep full RoPE and the original kv_b_proj;
4. K=V: kv_b_proj shrinks to exactly H * no_rope_head_dim outputs, the
   expanded cache's V is the same storage as K_nope, its byte accounting
   drops the V share, and W_UK is W_UV in the absorbed path;
5. with p-RoPE + K=V on: absorbed == expanded, cached decode == full
   forward, finite gradients;
6. validation.

Run from repo root: python3 evastar/tests/test_global_prope_kv.py
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import AttentionConfig, EvastarV5Config  # noqa: E402
from evastar.src.attention.mla import MLA, RotaryEmbedding  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402


def _cfg(layers=None, **kw):
    kw.setdefault("v_head_dim", 24)       # == lite no_rope_head_dim (24)
    extra = {} if layers is None else {"num_hidden_layers": layers}
    return EvastarV5Config(size="lite", attention=AttentionConfig(**kw),
                            **extra)


def _mlas(model):
    return [m for m in model.modules() if isinstance(m, MLA)]


def test_defaults_unchanged():
    ids = torch.randint(3, 1000, (2, 9))
    outs, states = [], []
    for kw in ({}, dict(global_rope_partial=1.0, global_k_eq_v=False)):
        torch.manual_seed(0)
        m = EvastarV5(EvastarV5Config(
            size="lite", attention=AttentionConfig(**kw))).eval()
        states.append(m.state_dict())
        with torch.no_grad():
            outs.append(m(ids)[0])
    sa, sb = states
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(outs[0], outs[1])
    assert all(not a.k_eq_v and a.rope.partial == 1.0 for a in _mlas(m))
    print("PASS test_defaults_unchanged")


def test_prope_identity_dims():
    full = RotaryEmbedding(16, 4096)
    part = RotaryEmbedding(16, 4096, partial=0.25)
    keep = 2                                    # round(0.25 * 8 pairs)
    assert torch.equal(part.inv_freq[:keep], full.inv_freq[:keep])
    assert (part.inv_freq[keep:] == 0).all()
    assert (full.inv_freq[keep:] > 0).all()
    pos = torch.tensor([[0, 1, 7, 100, 4000]])
    cos, sin = part(pos)
    cos, sin = cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1])
    ident = [i for i in range(16) if i % (16 // 2) >= keep]
    assert (cos[:, ident] == 1).all() and (sin[:, ident] == 0).all()
    assert (sin[1:, [0, 8]] != 0).all()         # kept pairs still rotate

    # the rotated q/k: identity dims equal the unrotated input
    torch.manual_seed(1)
    mla = MLA(_cfg(global_rope_partial=0.25), layer_idx=0)
    x = torch.randn(1, 5, mla.hidden_size)
    q_rope_t = []
    orig_rope = mla.rope
    with torch.no_grad():
        rotated = mla._project_new_tokens(x, pos)
        mla.rope = RotaryEmbedding(mla.rope_head_dim, 4096, partial=0.0)
        unrot = mla._project_new_tokens(x, pos)
    mla.rope = orig_rope
    half = mla.rope_head_dim // 2
    k = round(0.25 * half)
    ident = [i for i in range(mla.rope_head_dim) if i % half >= k]
    for a, b in zip(rotated, unrot):
        if a.shape[-1] == mla.rope_head_dim:
            assert torch.equal(a[..., ident], b[..., ident])
            q_rope_t.append(a)
    assert q_rope_t, "no rope-dim tensors returned"
    print("PASS test_prope_identity_dims")


def test_only_global_layers():
    # 4 layers, every 4: layers 0 and 3 global, 1 and 2 windowed
    cfg = _cfg(layers=4, sliding_window=8, global_attention_every=4,
               global_rope_partial=0.25, global_k_eq_v=True)
    torch.manual_seed(0)
    model = EvastarV5(cfg)
    mlas = _mlas(model)
    seen = set()
    for a in mlas:
        if a.sliding_window is None:
            assert a.k_eq_v and a.rope.partial == 0.25
            assert a.kv_b_proj.out_features == a.num_heads * a.no_rope_head_dim
        else:
            assert not a.k_eq_v and a.rope.partial == 1.0
            assert a.kv_b_proj.out_features == a.num_heads * (
                a.no_rope_head_dim + a.v_head_dim)
        seen.add(a.sliding_window is None)
    assert seen == {True, False}, "need both global and windowed MLA layers"
    print("PASS test_only_global_layers")


def test_k_eq_v_shapes_and_cache():
    torch.manual_seed(0)
    base = MLA(_cfg(use_absorption=False), layer_idx=0)
    torch.manual_seed(0)
    kv = MLA(_cfg(use_absorption=False, global_k_eq_v=True), layer_idx=0)
    H, dn = kv.num_heads, kv.no_rope_head_dim
    assert base.kv_b_proj.weight.numel() == 2 * kv.kv_b_proj.weight.numel()
    assert kv.kv_b_proj.weight.shape == (H * dn, kv.kv_latent_dim)
    w_uk, w_uv = kv._w_uk_w_uv()
    assert w_uk is w_uv and w_uk.shape == (H, dn, kv.kv_latent_dim)

    x = torch.randn(2, 6, kv.hidden_size)
    pos = torch.arange(6).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        out = kv(x, position_ids=pos, use_cache=True)
    present = [o for o in out if isinstance(o, tuple)][-1]
    k_nope, _, v = present
    assert v.data_ptr() == k_nope.data_ptr() and v is k_nope
    # decode one more token: still shared
    with torch.no_grad():
        out = kv(torch.randn(2, 1, kv.hidden_size),
                 position_ids=torch.full((2, 1), 6), past_key_value=present,
                 use_cache=True)
    k2, _, v2 = [o for o in out if isinstance(o, tuple)][-1]
    assert v2 is k2 and k2.shape[2] == 7
    # accounting: cached values per token drop the V share (one shared
    # K_nope/V tensor of H * no_rope per token per batch row)
    n = 7
    assert k2.numel() // (2 * n) == H * dn
    stats_b, stats_k = base.get_kv_cache_size(n), kv.get_kv_cache_size(n)
    assert stats_k["mla_expanded"] == n * (H * dn + kv.rope_head_dim)
    assert stats_b["mla_expanded"] - stats_k["mla_expanded"] == \
        n * H * base.v_head_dim
    print("PASS test_k_eq_v_shapes_and_cache")


def test_paths_and_decode():
    kw = dict(global_rope_partial=0.25, global_k_eq_v=True)
    ids = torch.randint(3, 1000, (2, 13))
    fulls = {}
    for absorb in (True, False):
        torch.manual_seed(0)
        m = EvastarV5(_cfg(use_absorption=absorb, **kw)).eval()
        with torch.no_grad():
            full = m(ids)[0]
            lg, _, past = m(ids[:, :5], use_cache=True)
            outs = [lg]
            for t in range(5, 13):
                lg, _, past = m(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
                outs.append(lg)
        assert (torch.cat(outs, 1) - full).abs().max() < 1e-4
        fulls[absorb] = full
    assert (fulls[True] - fulls[False]).abs().max() < 1e-4
    torch.manual_seed(0)
    m = EvastarV5(_cfg(**kw)).train()
    m(ids)[0].logsumexp(-1).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters()
               if p.grad is not None)
    assert any(a.kv_b_proj.weight.grad is not None and
               a.kv_b_proj.weight.grad.abs().sum() > 0 for a in _mlas(m))
    print("PASS test_paths_and_decode")


def test_validation():
    for kw, needle in (
            (dict(global_rope_partial=1.5), "global_rope_partial"),
            (dict(global_rope_partial=-0.1), "global_rope_partial"),
            (dict(global_rope_partial=True), "global_rope_partial"),
            (dict(global_rope_partial=0.25,
                  rope_scaling={"type": "linear", "factor": 2.0}),
             "rope_scaling"),
            (dict(global_k_eq_v=1), "must be a bool"),
            (dict(global_k_eq_v=True, v_head_dim=32), "v_head_dim")):
        try:
            EvastarV5Config(size="lite", attention=AttentionConfig(**kw))
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_prope_identity_dims()
    test_only_global_layers()
    test_k_eq_v_shapes_and_cache()
    test_paths_and_decode()
    test_validation()
