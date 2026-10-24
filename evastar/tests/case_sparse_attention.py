"""HeliosLM v5 test cases — sparse top-k, soft-capping, sliding window.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_expect_raises, _pass)


def test_sparse_top_k_attention():
    from evastar.src.attention.mla import MLA

    config = EvastarV5Config(size="lite")
    config.attention.sparse_top_k = 4
    torch.manual_seed(303)
    mla = MLA(config).eval()
    L = 12
    h = torch.randn(1, L, config.hidden_size)

    # Sparse cached decode stays finite and keeps the cache layout.
    with torch.no_grad():
        past = None
        step_outs = []
        for i in range(L):
            o, past = mla(h[:, i:i + 1], past_key_value=past, use_cache=True)
            step_outs.append(o)
        sparse_out = torch.cat(step_outs, dim=1)
    assert torch.isfinite(sparse_out).all(), "sparse decode produced non-finite"
    assert past[0].shape[2] == L and past[1].shape[2] == L, (
        "sparse attention must not alter the cache layout")

    # The cache tensors are BIT-IDENTICAL to a dense run's (selection
    # gathers copies for the score matmul; it never mutates the cache).
    config_d = EvastarV5Config(size="lite")
    mla_d = MLA(config_d).eval()
    mla_d.load_state_dict(mla.state_dict())
    with torch.no_grad():
        past_d = None
        for i in range(L):
            _, past_d = mla_d(h[:, i:i + 1], past_key_value=past_d,
                              use_cache=True)
    assert torch.equal(past[0], past_d[0]) and torch.equal(past[1], past_d[1]), (
        "sparse decode must build exactly the dense cache")

    # k >= kv_len degenerates to dense attention (no selection kicks in).
    config_big = EvastarV5Config(size="lite")
    config_big.attention.sparse_top_k = 10**6
    mla_big = MLA(config_big).eval()
    mla_big.load_state_dict(mla.state_dict())
    with torch.no_grad():
        out_big, _ = mla_big(h, use_cache=False)
        out_dense, _ = mla_d(h, use_cache=False)
    d_degen = (out_big - out_dense).abs().max().item()
    assert d_degen == 0.0, (
        f"sparse_top_k >= kv_len must equal dense exactly, diff {d_degen:.3e}")

    # k = 1 is fully determined at DECODE: only the current token is
    # selected (the indexer force-includes it), so the output equals the
    # o_proj of the LAST latent mapped through W_UV — no softmax
    # uncertainty. (Prefill stays dense by design, so exercise the
    # decode path token by token.)
    config1 = EvastarV5Config(size="lite")
    config1.attention.sparse_top_k = 1
    mla1 = MLA(config1).eval()
    mla1.load_state_dict(mla.state_dict())
    with torch.no_grad():
        past1 = None
        out1 = None
        for i in range(L):
            out1, past1 = mla1(h[:, i:i + 1], past_key_value=past1,
                               use_cache=True)
        w_uk, w_uv = mla1._w_uk_w_uv()
        B, H = 1, mla1.num_heads
        c_last = past1[0][:, :, -1:, :].to(torch.float32).expand(
            -1, H, -1, -1)  # [B, H, 1, d_c]
        exp = torch.einsum("bhqc,hvc->bqhv", c_last, w_uv)
        exp = mla1.o_proj(exp.reshape(B, 1, H * mla1.v_head_dim))
    d_k1 = (out1 - exp).abs().max().item()
    assert d_k1 < 1e-5, (
        f"k=1 sparse output must be the last-token W_UV projection, "
        f"diff {d_k1:.3e}")

    # Padding masks are honored during selection (masked-out keys cannot
    # be picked; output stays finite).
    with torch.no_grad():
        mask = torch.ones(1, L)
        mask[0, :6] = 0
        past_m = None
        for i in range(L):
            o_m, past_m = mla(h[:, i:i + 1],
                              attention_mask=mask[:, :i + 1],
                              past_key_value=past_m, use_cache=True)
    assert torch.isfinite(o_m).all(), "sparse decode with padding broke"

    # Model-level integration: a full lite model decodes with sparse
    # attention — finite, deterministic, and EXACTLY dense when
    # sparse_top_k covers the whole cache. (A bounded-drift assertion
    # against dense is meaningless on an untrained random model: dropping
    # half the cache legitimately renormalizes the softmax.)
    from evastar.src.model_v5 import EvastarV5
    cfg_s = EvastarV5Config(size="lite")
    cfg_s.attention.sparse_top_k = 4
    torch.manual_seed(304)
    model_s = EvastarV5(cfg_s).eval()
    ids = torch.randint(0, cfg_s.vocab_size, (1, 6))

    def decode(model, steps=3):
        with torch.no_grad():
            lg, _, past = model(ids, use_cache=True)
            for _ in range(steps):
                nxt = lg[:, -1:].argmax(-1)
                lg, _, past = model(nxt, past_key_values=past,
                                    use_cache=True)
        return lg

    lg_s = decode(model_s)
    assert torch.isfinite(lg_s).all(), "model-level sparse decode non-finite"
    model_s2 = EvastarV5(cfg_s).eval()
    model_s2.load_state_dict(model_s.state_dict())
    assert torch.equal(lg_s, decode(model_s2)), (
        "sparse top-k decode is not deterministic")

    cfg_big = EvastarV5Config(size="lite")
    cfg_big.attention.sparse_top_k = 10**6
    model_big = EvastarV5(cfg_big).eval()
    model_big.load_state_dict(model_s.state_dict())
    cfg_m = EvastarV5Config(size="lite")
    model_m = EvastarV5(cfg_m).eval()
    model_m.load_state_dict(model_s.state_dict())
    d_model = (decode(model_big) - decode(model_m)).abs().max().item()
    assert d_model == 0.0, (
        f"model-level sparse_top_k >= kv_len must equal dense exactly: "
        f"{d_model:.3e}")

    # Loud errors: non-absorbed cache rejects the flag; bad values fail at
    # config validation.
    bad = EvastarV5Config(size="lite")
    bad.attention.use_absorption = False
    bad.attention.sparse_top_k = 4
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad.attention), "sparse_top_k w/o absorption")
    bad2 = EvastarV5Config(size="lite")
    bad2.attention.sparse_top_k = 0
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad2.attention), "sparse_top_k = 0")
    _pass("test_sparse_top_k_attention",
          f"cache bit-equal dense, k>=L exact (unit+model), k=1 W_UV form "
          f"{d_k1:.1e}, deterministic, guards OK")


# ----------------------------------------------------------------------
# v5.9: Gemma/GLM-style attention logit soft-capping (both cache modes)
# ----------------------------------------------------------------------
def test_attention_logit_soft_cap():
    from evastar.src.attention.mla import MLA

    torch.manual_seed(901)
    L = 8

    def build(cap, use_absorption):
        cfg = EvastarV5Config(size="lite")
        cfg.attention.use_absorption = use_absorption
        cfg.attention.logit_soft_cap = cap
        torch.manual_seed(901)  # identical weights across builds
        return MLA(cfg).eval(), cfg

    # 1) Saturation semantics: with a tiny cap every score collapses to
    #    +/-cap, so softmax is EXACTLY uniform over the causal prefix.
    #    Reference: out_i = o_proj(mean_{j<=i} v_j) per head.
    mla, cfg = build(cap=1e-6, use_absorption=False)
    h = torch.randn(1, L, cfg.hidden_size)
    with torch.no_grad():
        out, _ = mla(h)
        pos = torch.arange(L).unsqueeze(0)
        _, _, c_kv, _ = mla._project_new_tokens(h, pos)
        kv = mla.kv_b_proj(c_kv).view(
            1, L, mla.num_heads, mla.no_rope_head_dim + mla.v_head_dim)
        v = kv[..., mla.no_rope_head_dim:].transpose(1, 2)  # [1,H,L,d_v]
        ref = torch.stack([
            mla.o_proj(v[:, :, : i + 1].mean(dim=2).reshape(
                1, mla.num_heads * mla.v_head_dim))
            for i in range(L)], dim=1)
    d_uni = (out - ref).abs().max().item()
    assert d_uni < 1e-4, (
        f"tiny cap must give exactly-uniform attention: diff {d_uni:.3e}")

    # 2) Boundedness: adversarially large inputs cannot produce NaN/inf,
    #    and the capped probabilities stay well-defined.
    mla_b, _ = build(cap=5.0, use_absorption=True)
    with torch.no_grad():
        out_big, _ = mla_b(h * 1e4)
    assert torch.isfinite(out_big).all(), "capped attention produced NaN/inf"

    # 3) Huge cap approaches the uncapped path (tanh(x/cap)*cap ~ x).
    mla_huge, _ = build(cap=1e9, use_absorption=True)
    mla_none, _ = build(cap=None, use_absorption=True)
    with torch.no_grad():
        d_huge = (mla_huge(h)[0] - mla_none(h)[0]).abs().max().item()
    assert d_huge < 1e-4, f"huge cap should be ~uncapped: {d_huge:.3e}"

    # 4) Absorbed vs expanded agree WITH the cap on (atol 1e-4, the
    #    suite's standard cross-mode tolerance).
    mla_a, _ = build(cap=5.0, use_absorption=True)
    mla_e, _ = build(cap=5.0, use_absorption=False)
    with torch.no_grad():
        d_mode = (mla_a(h)[0] - mla_e(h)[0]).abs().max().item()
    assert d_mode < 1e-4, f"absorbed/expanded diverge under cap: {d_mode:.3e}"

    # 5) Cached decode == one-shot with the cap on (both modes).
    for absorb in (True, False):
        mla_c, _ = build(cap=5.0, use_absorption=absorb)
        with torch.no_grad():
            full, _ = mla_c(h)
            past, outs = None, []
            for i in range(L):
                o, past = mla_c(h[:, i:i + 1], past_key_value=past,
                                use_cache=True)
                outs.append(o)
        d_dec = (full - torch.cat(outs, dim=1)).abs().max().item()
        assert d_dec < 1e-4, (
            f"capped decode mismatch (absorb={absorb}): {d_dec:.3e}")

    # 6) Loud guards: config validation and hand-built module.
    bad = EvastarV5Config(size="lite")
    bad.attention.logit_soft_cap = 0.0
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad.attention), "logit_soft_cap=0")
    hand = EvastarV5Config(size="lite")
    hand.attention.logit_soft_cap = -3.0
    _expect_raises(ValueError, lambda: MLA(hand), "MLA with cap=-3")
    _pass("test_attention_logit_soft_cap",
          f"tiny-cap uniform diff {d_uni:.1e}, huge-cap {d_huge:.1e}, "
          f"modes {d_mode:.1e}, decode OK, guards OK")


# ----------------------------------------------------------------------
# v5.9: per-head QK-norm (GLM-4.5 / Qwen3 / Gemma style), pre-RoPE
# ----------------------------------------------------------------------
def test_qk_norm():
    from evastar.src.attention.mla import MLA

    torch.manual_seed(902)
    L = 8

    def build(qk, use_absorption=True):
        cfg = EvastarV5Config(size="lite")
        cfg.attention.use_absorption = use_absorption
        cfg.attention.qk_norm = qk
        torch.manual_seed(902)  # identical weights across builds
        return MLA(cfg).eval(), cfg

    mla, cfg = build(qk=True)
    h = torch.randn(1, L, cfg.hidden_size)

    # 1) Wiring: the norm modules sit on the q/k_rope path, so the module
    #    is (near-)invariant to UPSCALING those projections (the RMSNorm
    #    divides the scale back out). The eps in the RMSNorm denominator
    #    breaks exact scale invariance once mean(x^2) ~ eps, so the
    #    invariance check uses upscale factors only (mean(x^2) grows,
    #    eps becomes negligible); the downscale regime is bounded by eps
    #    and documented, not silently exact.
    for proj, factor in (("q_b_proj", 100.0), ("k_rope_proj", 100.0)):
        mla_s, _ = build(qk=True)
        with torch.no_grad():
            getattr(mla_s, proj).weight.mul_(factor)
            d = (mla(h)[0] - mla_s(h)[0]).abs().max().item()
        assert d < 1e-4, (
            f"qk_norm must cancel a {factor}x {proj} rescale: {d:.3e}")

    # 2) Control: WITHOUT qk_norm the same rescale DOES change the output.
    mla_off, _ = build(qk=False)
    mla_off_s, _ = build(qk=False)
    with torch.no_grad():
        mla_off_s.q_b_proj.weight.mul_(100.0)
        d_off = (mla_off(h)[0] - mla_off_s(h)[0]).abs().max().item()
    assert d_off > 1e-2, (
        f"control failed: q rescale should matter without qk_norm "
        f"({d_off:.3e})")

    # 3) Absorbed vs expanded agree with qk_norm on; cached decode ==
    #    one-shot in both modes.
    mla_e, _ = build(qk=True, use_absorption=False)
    with torch.no_grad():
        d_mode = (mla(h)[0] - mla_e(h)[0]).abs().max().item()
    assert d_mode < 1e-4, f"absorbed/expanded diverge (qk_norm): {d_mode:.3e}"
    for absorb in (True, False):
        mla_c, _ = build(qk=True, use_absorption=absorb)
        with torch.no_grad():
            full, _ = mla_c(h)
            past, outs = None, []
            for i in range(L):
                o, past = mla_c(h[:, i:i + 1], past_key_value=past,
                                use_cache=True)
                outs.append(o)
        d_dec = (full - torch.cat(outs, dim=1)).abs().max().item()
        assert d_dec < 1e-4, (
            f"qk_norm decode mismatch (absorb={absorb}): {d_dec:.3e}")

    # 4) Default off keeps the v5.8 numerics exactly: qk_norm=False must
    #    be bit-identical to a config that never heard of the flag.
    cfg_plain = EvastarV5Config(size="lite")
    assert cfg_plain.attention.qk_norm is False
    torch.manual_seed(902)
    mla_p = MLA(cfg_plain).eval()
    with torch.no_grad():
        d_def = (mla_off(h)[0] - mla_p(h)[0]).abs().max().item()
    assert d_def == 0.0, f"default path changed: {d_def:.3e}"

    # 5) Loud guard: non-bool qk_norm rejected at config validation.
    bad = EvastarV5Config(size="lite")
    bad.attention.qk_norm = "yes"
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad.attention), "qk_norm='yes'")
    _pass("test_qk_norm",
          f"scale-invariance OK, control diff {d_off:.3f}, modes "
          f"{d_mode:.1e}, decode OK, default bit-identical, guards OK")


# ----------------------------------------------------------------------
# v5.9: sliding-window attention + StreamingLLM attention sinks
# ----------------------------------------------------------------------
def test_sliding_window_attention():
    from evastar.src.attention.mla import MLA

    torch.manual_seed(903)
    L = 10
    W, S = 4, 2

    def build(window, sink=0, use_absorption=True, sparse=None):
        cfg = EvastarV5Config(size="lite")
        cfg.attention.use_absorption = use_absorption
        cfg.attention.sliding_window = window
        cfg.attention.sliding_window_sink = sink
        cfg.attention.sparse_top_k = sparse
        torch.manual_seed(903)  # identical weights across builds
        return MLA(cfg).eval(), cfg

    mla, cfg = build(window=W)
    h1 = torch.randn(1, L, cfg.hidden_size)
    h2 = h1.clone()
    h2[0, 0] += 100.0  # perturb an out-of-window token

    # 1) Exact isolation: query i attends keys j > i - W, so perturbing
    #    token 0 must leave outputs at positions >= W EXACTLY unchanged
    #    (single layer, so no cross-layer receptive-field growth).
    with torch.no_grad():
        o1, _ = mla(h1)
        o2, _ = mla(h2)
    d_tail = (o1[:, W:] - o2[:, W:]).abs().max().item()
    d_head = (o1[:, 1:W] - o2[:, 1:W]).abs().max().item()
    assert d_tail == 0.0, f"out-of-window key leaked: {d_tail:.3e}"
    assert d_head > 1e-2, f"window never engages? head diff {d_head:.3e}"

    # 2) Window >= sequence length is bit-identical to full attention.
    mla_full, _ = build(window=None)
    mla_big, _ = build(window=10**6)
    with torch.no_grad():
        d_big = (mla_full(h1)[0] - mla_big(h1)[0]).abs().max().item()
    assert d_big == 0.0, f"oversized window must be exact: {d_big:.3e}"

    # 3) Cached decode == one-shot under the window (both modes), and the
    #    cache itself is NEVER truncated (grows to L despite the slice).
    for absorb in (True, False):
        mla_c, _ = build(window=W, use_absorption=absorb)
        with torch.no_grad():
            full, _ = mla_c(h1)
            past, outs, growth = None, [], []
            for i in range(L):
                o, past = mla_c(h1[:, i:i + 1], past_key_value=past,
                                use_cache=True)
                outs.append(o)
                growth.append(past[0].shape[2])
        d_dec = (full - torch.cat(outs, dim=1)).abs().max().item()
        assert d_dec < 1e-4, (
            f"window decode mismatch (absorb={absorb}): {d_dec:.3e}")
        assert growth == list(range(1, L + 1)), (
            f"window must not truncate the cache: {growth}")

    # 4) Attention sinks: with sink=S the first S tokens stay attendable
    #    from anywhere (perturbing token 0 now CHANGES the last output),
    #    while a mid-stream token at position 3 is only attendable by
    #    queries 3..6 (distance < W=4) and is neither sink nor in-window
    #    for queries >= 7, so those outputs stay EXACTLY unchanged.
    mla_sink, _ = build(window=W, sink=S)
    h3 = h1.clone()
    h3[0, 3] += 100.0
    with torch.no_grad():
        o_sink1, _ = mla_sink(h1)
        o_sink2, _ = mla_sink(h2)   # token 0 perturbed (a sink token)
        o_sink3, _ = mla_sink(h3)   # token 3 perturbed (not sink, not window)
    d_sink = (o_sink1[0, -1] - o_sink2[0, -1]).abs().max().item()
    d_mid = (o_sink1[:, 7:] - o_sink3[:, 7:]).abs().max().item()
    assert d_sink > 1e-3, f"sink token not attendable: {d_sink:.3e}"
    assert d_mid == 0.0, f"non-sink out-of-window key leaked: {d_mid:.3e}"

    # 5) Composition with sparse top-k (v5.8): with k >= windowed length
    #    the sparse selection degenerates to dense-over-window EXACTLY.
    mla_w, _ = build(window=W + S)  # keep == 6, no sink
    mla_ws, _ = build(window=W + S, sparse=W + S)
    with torch.no_grad():
        past_w, past_s = None, None
        for i in range(L):
            ow, past_w = mla_w(h1[:, i:i + 1], past_key_value=past_w,
                               use_cache=True)
            os_, past_s = mla_ws(h1[:, i:i + 1], past_key_value=past_s,
                                 use_cache=True)
        d_ws = (ow - os_).abs().max().item()
    assert d_ws == 0.0, f"sparse k >= window must be exact: {d_ws:.3e}"

    # 6) Loud guards: bad window/sink rejected at config validation and at
    #    hand-built module construction.
    bad = EvastarV5Config(size="lite")
    bad.attention.sliding_window = 0
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad.attention), "sliding_window=0")
    bad2 = EvastarV5Config(size="lite")
    bad2.attention.sliding_window_sink = 1  # sink without a window
    _expect_raises(ValueError, lambda: EvastarV5Config(
        size="lite", attention=bad2.attention), "sink w/o window")
    hand = EvastarV5Config(size="lite")
    hand.attention.sliding_window = 4
    hand.attention.sliding_window_sink = -1
    _expect_raises(ValueError, lambda: MLA(hand), "MLA with sink=-1")
    _pass("test_sliding_window_attention",
          f"isolation exact (0.0), oversized-window exact, decode OK both "
          f"modes, sinks OK (d_sink {d_sink:.1e}), sparse compose exact, "
          f"guards OK")


# ----------------------------------------------------------------------
# v5.9: Gemma-2 style final logit soft-capping on the LM head
# ----------------------------------------------------------------------
def test_final_logit_soft_cap():
    from evastar.src.model_v5 import EvastarV5

    torch.manual_seed(904)
    CAP = 15.0
    cfg_c = EvastarV5Config(size="lite")
    cfg_c.final_logit_soft_cap = CAP
    model_c = EvastarV5(cfg_c).eval()
    cfg_u = EvastarV5Config(size="lite")
    model_u = EvastarV5(cfg_u).eval()
    model_u.load_state_dict(model_c.state_dict())

    ids = torch.randint(0, cfg_u.vocab_size, (2, 6))
    with torch.no_grad():
        logits_c, _, _ = model_c(ids)
        logits_u, _, _ = model_u(ids)

    # 1) Bounded: every logit strictly inside (-cap, cap).
    assert (logits_c.abs() < CAP).all(), (
        f"capped logits exceed the cap: max {logits_c.abs().max().item()}")

    # 2) Exact mapping: capped == cap * tanh(uncapped / cap).
    ref = CAP * torch.tanh(logits_u / CAP)
    d_map = (logits_c - ref).abs().max().item()
    assert d_map < 1e-5, f"cap mapping mismatch: {d_map:.3e}"

    # 3) Default (None) is the untouched v5.8 path: config default and a
    #    bit-identical forward.
    assert cfg_u.final_logit_soft_cap is None

    # 4) Gradient flows through the cap (training usability).
    model_c.train()
    logits_t, _, _ = model_c(ids)
    loss = logits_t.sum()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model_c.parameters()
                if p.grad is not None)
    assert gnorm > 0 and math.isfinite(gnorm), (
        f"no gradient through the soft cap: {gnorm}")
    model_c.eval()

    # 5) generate() inherits the cap (it consumes forward's logits) —
    #    smoke-test greedy decode for a few tokens.
    with torch.no_grad():
        gen = model_c.generate(ids[:, :3], max_new_tokens=3,
                               temperature=0.0)
    assert gen.shape[1] == 6, f"generate with cap broke: {gen.shape}"

    # 6) Loud guards: non-positive caps rejected at config validation.
    for bad_cap in (0.0, -1.0):
        _expect_raises(ValueError, lambda: EvastarV5Config(
            size="lite", final_logit_soft_cap=bad_cap),
            f"final_logit_soft_cap={bad_cap}")
    _pass("test_final_logit_soft_cap",
          f"bounded < {CAP}, mapping diff {d_map:.1e}, grad norm "
          f"{gnorm:.1f}, generate OK, guards OK")
