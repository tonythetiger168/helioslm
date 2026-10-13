"""HeliosLM v5 test cases — learned lightning indexer + sparse prefill.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_expect_raises, _pass)


def test_learned_lightning_indexer():
    from helioslm_v5.src.attention.mla import MLA
    from helioslm_v5.src.attention.lightning_indexer import (
        LearnedLightningIndexer,
    )

    # --- module-level contract: shapes, finiteness, determinism --------
    torch.manual_seed(101)
    idx = LearnedLightningIndexer(hidden_size=64, kv_latent_dim=16,
                                  num_heads=4, head_dim=8)
    hq = torch.randn(2, 5, 64)
    ck = torch.randn(2, 1, 9, 16)
    s = idx(hq, ck)
    assert s.shape == (2, 1, 5, 9), f"index score shape {tuple(s.shape)}"
    assert torch.isfinite(s).all() and (s >= 0).all(), (
        "index scores must be finite and non-negative (ReLU scorer)")
    assert torch.equal(idx(hq, ck), s), "indexer is not deterministic"
    s3d = idx(hq, ck.squeeze(1))  # [B, L, d_c] accepted too
    assert torch.equal(s3d, s), "3-d and 4-d cache forms must agree"
    _expect_raises(ValueError, lambda: idx(hq, torch.randn(2, 2, 9, 16)),
                   "non-shared latent cache")
    _expect_raises(ValueError, lambda: LearnedLightningIndexer(64, 16, 0, 8),
                   "indexer num_heads = 0")

    # --- config / layer loud validation --------------------------------
    cfg_bad = HeliosLMv5Config(size="lite")
    cfg_bad.attention.sparse_indexer = "magic"
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=cfg_bad.attention), "unknown sparse_indexer")
    cfg_bad2 = HeliosLMv5Config(size="lite")
    cfg_bad2.attention.sparse_indexer = "learned"  # without sparse_top_k
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=cfg_bad2.attention), "learned w/o top_k")
    cfg_bad3 = HeliosLMv5Config(size="lite")
    cfg_bad3.attention.indexer_num_heads = -1
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=cfg_bad3.attention), "indexer_num_heads < 0")
    cfg_bad4 = HeliosLMv5Config(size="lite")
    cfg_bad4.attention.sparse_prefill = True  # without sparse_top_k
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=cfg_bad4.attention), "prefill w/o top_k")

    # --- MLA decode with the learned indexer ----------------------------
    config = HeliosLMv5Config(size="lite")
    config.attention.sparse_top_k = 4
    config.attention.sparse_indexer = "learned"
    torch.manual_seed(303)
    mla = MLA(config).eval()
    assert hasattr(mla, "indexer"), "learned mode must build the indexer"
    L = 12
    h = torch.randn(1, L, config.hidden_size)
    with torch.no_grad():
        past = None
        outs = []
        for i in range(L):
            o, past = mla(h[:, i:i + 1], past_key_value=past, use_cache=True)
            outs.append(o)
        sparse_out = torch.cat(outs, dim=1)
    assert torch.isfinite(sparse_out).all(), "learned-indexer decode broke"

    # Cache is BIT-IDENTICAL to a dense run (selection gathers copies).
    config_d = HeliosLMv5Config(size="lite")
    mla_d = MLA(config_d).eval()
    mla_d.load_state_dict(
        {k: v for k, v in mla.state_dict().items()
         if not k.startswith("indexer")}, strict=False)
    with torch.no_grad():
        past_d = None
        for i in range(L):
            _, past_d = mla_d(h[:, i:i + 1], past_key_value=past_d,
                              use_cache=True)
    assert torch.equal(past[0], past_d[0]) and torch.equal(past[1], past_d[1]), (
        "learned-indexer decode must build exactly the dense cache")

    # k >= kv_len degenerates to EXACT dense (selection never kicks in).
    config_big = HeliosLMv5Config(size="lite")
    config_big.attention.sparse_top_k = 10**6
    config_big.attention.sparse_indexer = "learned"
    mla_big = MLA(config_big).eval()
    mla_big.load_state_dict(mla.state_dict())
    with torch.no_grad():
        out_big, _ = mla_big(h, use_cache=False)
        out_dense, _ = mla_d(h, use_cache=False)
    d_degen = (out_big - out_dense).abs().max().item()
    assert d_degen == 0.0, (
        f"learned indexer with k >= kv_len must equal dense exactly, "
        f"diff {d_degen:.3e}")

    # k = 1 at decode: only the current token (force-selected) attends.
    config1 = HeliosLMv5Config(size="lite")
    config1.attention.sparse_top_k = 1
    config1.attention.sparse_indexer = "learned"
    mla1 = MLA(config1).eval()
    mla1.load_state_dict(mla.state_dict())
    with torch.no_grad():
        past1 = None
        out1 = None
        for i in range(L):
            out1, past1 = mla1(h[:, i:i + 1], past_key_value=past1,
                               use_cache=True)
        w_uk, w_uv = mla1._w_uk_w_uv()
        c_last = past1[0][:, :, -1:, :].to(torch.float32).expand(
            -1, mla1.num_heads, -1, -1)
        exp = torch.einsum("bhqc,hvc->bqhv", c_last, w_uv)
        exp = mla1.o_proj(exp.reshape(1, 1, -1))
    d_k1 = (out1 - exp).abs().max().item()
    assert d_k1 < 1e-5, (
        f"learned indexer k=1 must be the last-token W_UV projection, "
        f"diff {d_k1:.3e}")

    # Padding masks are honored during selection.
    with torch.no_grad():
        mask = torch.ones(1, L)
        mask[0, :6] = 0
        past_m = None
        for i in range(L):
            o_m, past_m = mla(h[:, i:i + 1],
                              attention_mask=mask[:, :i + 1],
                              past_key_value=past_m, use_cache=True)
    assert torch.isfinite(o_m).all(), "learned decode with padding broke"

    # state_dict round-trip (engine/checkpoint compatibility).
    mla_rt = MLA(config).eval()
    mla_rt.load_state_dict(mla.state_dict())
    with torch.no_grad():
        p_rt = None
        o_rt = None
        for i in range(L):
            o_rt, p_rt = mla_rt(h[:, i:i + 1], past_key_value=p_rt,
                                use_cache=True)
    assert torch.equal(o_rt, outs[-1]), (
        "state_dict round-trip changed learned-indexer decode")

    # --- distill training (DSA recipe): the discrete top-k gets no LM
    # gradient, so the indexer is trained by distilling the detached
    # head-mean true-score distribution (the v5.8 free indexer) into it.
    mla_t = MLA(config)  # train mode
    mla_t.load_state_dict(mla.state_dict())
    h_t = torch.randn(2, 16, config.hidden_size)

    def _teacher_overlap(m, hx):
        with torch.no_grad():
            pos, _ = m._resolve_positions(hx, 0, None)
            qn, qr, c, kr = m._project_new_tokens(hx, pos)
            c = c.unsqueeze(1)
            w_uk2, _ = m._w_uk_w_uv()
            qa = torch.einsum("bhqd,hdc->bhqc", qn, w_uk2)
            teacher = (qa.mean(1, keepdim=True) @ c.transpose(-2, -1)
                       + qr.mean(1, keepdim=True) @ kr.transpose(-2, -1))
            student = m.indexer(hx, c)
            t4 = teacher.topk(4, -1).indices
            s4 = student.topk(4, -1).indices
            return (t4.unsqueeze(-1) == s4.unsqueeze(-2)
                    ).any(-1).float().mean().item()

    loss0 = mla_t.indexer_distill_loss(h_t).item()
    ov0 = _teacher_overlap(mla_t, h_t)
    opt = torch.optim.Adam(mla_t.indexer.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        mla_t.indexer_distill_loss(h_t).backward()
        opt.step()
    loss1 = mla_t.indexer_distill_loss(h_t).item()
    ov1 = _teacher_overlap(mla_t, h_t)
    # Measured on this fixture family (RNG variations of h_t): loss ratio
    # 0.41-0.43 after 400-800 steps, top-4 overlap 0.44-0.52 vs a
    # random-chance level of k/L = 0.25; overlap plateaus by ~600 steps.
    # The bounds below sit well inside the measured range. The residual
    # gap to a perfect 1.0 overlap is honest — near-tied teacher scores
    # make exact top-4 ordering unlearnable at this scale.
    assert loss1 < 0.5 * loss0, (
        f"distill loss did not halve: {loss0:.4f} -> {loss1:.4f}")
    assert ov1 >= 0.40 and ov1 > ov0 + 0.10, (
        f"distill did not raise teacher overlap: {ov0:.3f} -> {ov1:.3f}")
    # Only indexer parameters are updated by the distill optimizer.
    before = {n: p.detach().clone() for n, p in mla_t.named_parameters()
              if not n.startswith("indexer")}
    opt.zero_grad()
    mla_t.indexer_distill_loss(h_t).backward()
    opt.step()
    for n, p in mla_t.named_parameters():
        if not n.startswith("indexer"):
            assert torch.equal(p, before[n]), f"{n} moved under distill"
    _expect_raises(ValueError, lambda: mla_d.indexer_distill_loss(h_t),
                   "distill loss on head_mean indexer")

    # --- supervised learnability oracle (standalone module): a crisp
    # best-match task with a margin — the indexer must learn to pick, for
    # each query u_t, the key argmax_j u_t . c_j. Measured: accuracy
    # 0.042 (chance = 1/24) -> 0.979 after 500 Adam steps (~1.5 s CPU).
    torch.manual_seed(21)
    idx_s = LearnedLightningIndexer(256, 64, 4, 32)
    B_s, L_s = 4, 24
    h_s = torch.randn(B_s, L_s, 256)
    c_s = torch.randn(B_s, 1, L_s, 64)
    target = torch.einsum("btd,bijd->btj", h_s[..., :64], c_s).argmax(-1)

    def _acc():
        with torch.no_grad():
            return (idx_s(h_s, c_s).squeeze(1).argmax(-1) == target
                    ).float().mean().item()

    acc0 = _acc()
    opt_s = torch.optim.Adam(idx_s.parameters(), lr=3e-3)
    for _ in range(500):
        opt_s.zero_grad()
        scores = idx_s(h_s, c_s).squeeze(1)
        torch.nn.functional.cross_entropy(
            scores.reshape(-1, L_s), target.reshape(-1)).backward()
        opt_s.step()
    acc1 = _acc()
    assert acc1 >= 0.95 and acc0 < 0.2, (
        f"supervised best-match accuracy {acc0:.3f} -> {acc1:.3f}")
    _pass("test_learned_lightning_indexer",
          f"cache bit-equal, k>=L exact, k=1 W_UV form {d_k1:.1e}, "
          f"distill loss {loss0:.2f}->{loss1:.2f} overlap {ov0:.2f}->"
          f"{ov1:.2f}, supervised acc {acc0:.2f}->{acc1:.2f}")


def test_sparse_prefill_indexer():
    from helioslm_v5.src.attention.mla import MLA

    config = HeliosLMv5Config(size="lite")
    config.attention.sparse_top_k = 4
    config.attention.sparse_prefill = True
    torch.manual_seed(404)
    mla = MLA(config).eval()
    L = 12
    h = torch.randn(1, L, config.hidden_size)

    config_d = HeliosLMv5Config(size="lite")
    mla_d = MLA(config_d).eval()
    mla_d.load_state_dict(mla.state_dict())
    with torch.no_grad():
        out_s, past_s = mla(h, use_cache=True)
        out_d, past_d = mla_d(h, use_cache=True)
    assert torch.isfinite(out_s).all(), "sparse prefill produced non-finite"
    # The cache is bit-identical to dense (gathered copies only).
    assert torch.equal(past_s[0], past_d[0]) and torch.equal(
        past_s[1], past_d[1]), "sparse prefill must not alter the cache"
    # Queries with visible <= k keys select their whole visible set, so
    # they match dense up to float32 reassociation between the batched
    # matmul and the per-query gathered einsum — measured max 1.49e-07 on
    # this fixture; bound set at 5e-7 (> 3x the measurement, far below
    # any selection effect, which is O(1e-1) below).
    d_early = (out_s[:, :4] - out_d[:, :4]).abs().max().item()
    assert d_early < 5e-7, (
        f"early rows (visible <= k) must match dense, diff {d_early:.3e}")
    d_late = (out_s[:, 4:] - out_d[:, 4:]).abs().max().item()
    assert d_late > 1e-3, (
        f"rows beyond k must show real selection, diff {d_late:.3e}")

    # k >= kv_len: selection never kicks in, EXACT dense.
    config_big = HeliosLMv5Config(size="lite")
    config_big.attention.sparse_top_k = 10**6
    config_big.attention.sparse_prefill = True
    mla_big = MLA(config_big).eval()
    mla_big.load_state_dict(mla.state_dict())
    with torch.no_grad():
        out_big, _ = mla_big(h, use_cache=False)
    d_degen = (out_big - out_d).abs().max().item()
    assert d_degen == 0.0, (
        f"sparse prefill with k >= L must equal dense exactly, "
        f"diff {d_degen:.3e}")

    # Padding mask honored; fully-padded query rows stay finite.
    with torch.no_grad():
        mask = torch.ones(1, L)
        mask[0, :5] = 0
        out_m, _ = mla(h, attention_mask=mask, use_cache=False)
    assert torch.isfinite(out_m).all(), "sparse prefill with padding broke"

    # Learned indexer at prefill: finite + deterministic.
    config_l = HeliosLMv5Config(size="lite")
    config_l.attention.sparse_top_k = 4
    config_l.attention.sparse_indexer = "learned"
    config_l.attention.sparse_prefill = True
    torch.manual_seed(405)
    mla_l = MLA(config_l).eval()
    with torch.no_grad():
        o1, _ = mla_l(h, use_cache=False)
        o2, _ = mla_l(h, use_cache=False)
    assert torch.isfinite(o1).all() and torch.equal(o1, o2), (
        "learned-indexer sparse prefill must be finite and deterministic")

    # Analytic accounting (values, not wall-clock): closed form for
    # L = 8, k = 3 -> selected pairs 3*4/2 + 5*3 = 21 of 36 dense pairs.
    config_a = HeliosLMv5Config(size="lite")
    config_a.attention.sparse_top_k = 3
    config_a.attention.sparse_indexer = "learned"
    config_a.attention.sparse_prefill = True
    mla_a = MLA(config_a).eval()
    stats = mla_a.get_sparse_attention_stats(8)
    assert stats["prefill_selected_pairs"] == 21
    assert stats["prefill_dense_pairs"] == 36
    assert stats["decode_attended_keys_per_step"] == 3
    assert abs(stats["decode_key_ratio"] - 3 / 8) < 1e-12
    assert abs(stats["prefill_pair_ratio"] - 21 / 36) < 1e-12
    assert stats["indexer_params"] == sum(
        p.numel() for p in mla_a.indexer.parameters())
    assert stats["indexer_flops_per_scored_pair"] == 2 * 4 * 32
    # Without sparse_prefill the prefill side is dense by definition.
    config_a2 = HeliosLMv5Config(size="lite")
    config_a2.attention.sparse_top_k = 3
    mla_a2 = MLA(config_a2).eval()
    stats2 = mla_a2.get_sparse_attention_stats(8)
    assert stats2["prefill_selected_pairs"] == 36
    assert stats2["indexer"] == "head_mean"
    assert stats2["indexer_params"] == 0
    _expect_raises(ValueError,
                   lambda: mla_d.get_sparse_attention_stats(8),
                   "stats without sparse_top_k")
    _expect_raises(ValueError,
                   lambda: mla_a.get_sparse_attention_stats(0),
                   "stats with seq_len = 0")

    # Model-level: greedy decode with learned indexer + sparse prefill is
    # finite and deterministic end to end.
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(406)
    model = HeliosLMv5(config_l).eval()
    ids = torch.randint(0, config_l.vocab_size, (1, 6))
    with torch.no_grad():
        lg1, _, past = model(ids, use_cache=True)
        for _ in range(3):
            nxt = lg1[:, -1:].argmax(-1)
            lg1, _, past = model(nxt, past_key_values=past, use_cache=True)
    model2 = HeliosLMv5(config_l).eval()
    model2.load_state_dict(model.state_dict())
    with torch.no_grad():
        lg2, _, past2 = model2(ids, use_cache=True)
        for _ in range(3):
            nxt = lg2[:, -1:].argmax(-1)
            lg2, _, past2 = model2(nxt, past_key_values=past2,
                                   use_cache=True)
    assert torch.isfinite(lg1).all() and torch.equal(lg1, lg2), (
        "model-level sparse learned decode must be finite/deterministic")
    _pass("test_sparse_prefill_indexer",
          f"early-row match {d_early:.1e} (< 5e-7 reassoc bound), "
          f"selection active {d_late:.1e}, k>=L exact, stats closed-form "
          f"21/36, model-level deterministic")


# ----------------------------------------------------------------------
# limitations round 2026-10-13: IndexShare cross-layer-group ROUTING
# stress (daily-report candidate 2026-10-12 #1). The PR #11 module tests
# (test_indexer_share.py, 6/6) cover the happy path: share=2 over 4
# layers, short decode, distill routing, checkpoint roundtrip. This test
# stresses the boundaries they do not:
#   1. Group-layout edges: share=2 over 5 layers (trailing singleton
#      group), share=3 over 4, share=8 over 4 (share > layers — the
#      config validator only demands a positive int, so the degenerate
#      single-group layout must still wire correctly).
#   2. Routing-correctness oracle: a share=4 model must be BITWISE equal
#      to a share=1 model whose follower indexers are overwritten with
#      the leader's weights (sharing = per-layer with duplicated weights;
#      anything else is a wiring bug).
#   3. Long-sequence shared decode: 24-token prompt + 40 greedy steps
#      (kv_len 25..64 = 8x top_k) — finite logits every step and bitwise
#      determinism across two runs.
#   4. The kv_len == top_k -> top_k+1 transition (dense step to first
#      sparse step) crossed cleanly.
#   5. Scorer invariants under input stress: h x 1e3 magnitude and exact
#      zeros keep scores finite and deterministic (no NaN propagation
#      into top-k selection).
# ----------------------------------------------------------------------
def test_indexer_share_group_stress():
    from helioslm_v5.src.model_v5 import HeliosLMv5

    def _build(share, layers=4, top_k=8, seed=101):
        torch.manual_seed(seed)
        cfg = HeliosLMv5Config(size="lite")
        cfg.num_hidden_layers = layers
        cfg.attention.sparse_top_k = top_k
        cfg.attention.sparse_indexer = "learned"
        cfg.attention.indexer_share_every = share
        return HeliosLMv5(cfg).eval(), cfg

    # --- 1. group-layout edges -------------------------------------------
    for share, layers in ((2, 5), (3, 4), (8, 4)):
        model, _ = _build(share, layers)
        attns = [layer.attention for layer in model.layers]
        leaders = [m for m in range(layers) if m % share == 0]
        for m, attn in enumerate(attns):
            leader = m - (m % share)
            assert attn._effective_indexer() is attns[leader].indexer, (
                f"share={share} layers={layers}: layer {m} routed to "
                f"{attn._effective_indexer()}, expected leader {leader}'s")
            if m in leaders:
                assert attn.indexer is not None
            else:
                assert attn.indexer is None, (
                    f"follower layer {m} must not register an indexer")
        keys = model.state_dict().keys()
        n_registered = len({
            int(k.split(".")[1]) for k in keys
            if ".attention.indexer." in k})
        assert n_registered == len(leaders), (
            f"share={share} layers={layers}: {n_registered} registered "
            f"indexers, expected {len(leaders)}")
        single = sum(p.numel() for p in attns[0].indexer.parameters())
        total_ix = sum(p.numel() for a in attns if a.indexer is not None
                       for p in a.indexer.parameters())
        assert total_ix == len(leaders) * single

    # --- 2. routing oracle: shared == duplicated weights, bitwise -------
    shared, cfg = _build(share=4, layers=4)
    dup, _ = _build(share=1, layers=4)
    missing, unexpected = dup.load_state_dict(shared.state_dict(),
                                              strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # The only missing keys must be the follower indexer params (kept at
    # their seeded init until overwritten below).
    assert missing and all(".attention.indexer." in k for k in missing), (
        f"unexpected missing keys: {missing[:5]}")
    leader_sd = shared.layers[0].attention.indexer.state_dict()
    for i in range(1, 4):
        dup.layers[i].attention.indexer.load_state_dict(leader_sd)
    ids = torch.randint(3, cfg.vocab_size, (1, 7))
    with torch.no_grad():
        lg_s, _, _ = shared(ids)
        lg_d, _, _ = dup(ids)
    assert torch.equal(lg_s, lg_d), (
        "share=4 must equal per-layer-with-duplicated-weights bitwise; "
        f"max diff {(lg_s - lg_d).abs().max().item()}")
    del dup
    # 1M-position-table leftovers from other tests make memory tight on
    # this box; drop the big models deterministically.

    # --- 3+4. long shared decode; kv_len == top_k transition ------------
    model, cfg = _build(share=2, layers=4, top_k=8)

    def _greedy(ids, steps):
        with torch.no_grad():
            logits, _, past = model(ids, use_cache=True)
            out = []
            for _ in range(steps):
                assert torch.isfinite(logits).all(), \
                    "non-finite logits during shared decode"
                nxt = logits[0, -1].argmax().view(1, 1)
                out.append(nxt.item())
                logits, _, past = model(nxt, past_key_values=past,
                                        use_cache=True)
        return out, past

    ids = torch.randint(3, cfg.vocab_size, (1, 24))
    r1, past1 = _greedy(ids, 40)   # kv_len 25..64 = 8x top_k
    r2, _ = _greedy(ids, 40)
    assert r1 == r2, "long shared decode not deterministic"
    assert past1[0][0].shape[2] == 24 + 40

    # kv_len == top_k boundary: 7-token prompt -> decode steps see
    # kv_len 8 (== top_k: still dense), 9 (first sparse step), 10.
    ids_b = torch.randint(3, cfg.vocab_size, (1, 7))
    b1, _ = _greedy(ids_b, 3)
    b2, _ = _greedy(ids_b, 3)
    assert b1 == b2, "dense->sparse transition not deterministic"

    # --- 5. scorer invariants under input stress -------------------------
    leader_ix = model.layers[0].attention._effective_indexer()
    follower = model.layers[1].attention
    assert follower._effective_indexer() is leader_ix
    c_kv = past1[1][0]  # follower layer cache: [B, 1, kv, d_c]
    h = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        s_base = leader_ix(h, c_kv)
        s_big = leader_ix(h * 1e3, c_kv)
        s_zero = leader_ix(torch.zeros_like(h), c_kv)
        s_base2 = leader_ix(h, c_kv)
    assert torch.isfinite(s_base).all(), "index scores not finite"
    assert torch.isfinite(s_big).all(), "x1e3 hidden blew up the scorer"
    assert torch.isfinite(s_zero).all(), "zero hidden blew up the scorer"
    assert torch.equal(s_base, s_base2), "scorer not deterministic"
    assert s_base.shape[-1] == 64, (
        f"scorer must cover the full 64-token cache, got "
        f"{s_base.shape[-1]}")
    _pass("test_indexer_share_group_stress",
          "layouts (2,5)/(3,4)/(8,4) route to leaders exactly, shared=="
          "duplicated bitwise, 40-step decode deterministic to kv=64, "
          "k==kv_len transition clean, scorer finite under x1e3/zero")
