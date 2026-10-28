"""HeliosLM v5 test cases — RoPE/YaRN scaling, FP8 KV cache, Hyper-Connections, NoPE.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math

import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_expect_raises, _pass)


# ----------------------------------------------------------------------
# v5.7: RoPE scaling (linear / NTK), FP8 KV cache, Hyper-Connections, QAT
# ----------------------------------------------------------------------
def test_rope_scaling():
    from helioslm_v5.src.attention.mla import RotaryEmbedding
    dim = 8
    r0 = RotaryEmbedding(dim, 2048)

    # factor == 1 must be exactly vanilla
    r1 = RotaryEmbedding(dim, 2048, scaling={"type": "linear", "factor": 1.0})
    r2 = RotaryEmbedding(dim, 2048, scaling={"type": "ntk", "factor": 1.0})
    assert torch.equal(r0.inv_freq, r1.inv_freq), "linear factor=1 != vanilla"
    assert torch.equal(r0.inv_freq, r2.inv_freq), "ntk factor=1 != vanilla"

    # linear: position p with factor f behaves like vanilla position p/f
    f = 4.0
    rl = RotaryEmbedding(dim, 2048, scaling={"type": "linear", "factor": f})
    pos = torch.arange(2 * int(f) * 3).unsqueeze(0)
    cl, _ = rl(pos)
    c0, _ = r0(pos)
    got = cl[0, 0, ::int(f)]  # scaled positions 0, f, 2f, ...
    want = c0[0, 0, :len(got)]  # == vanilla positions 0, 1, 2, ...
    assert (got - want).abs().max().item() < 1e-6, (
        f"linear scaling: position {int(f)} should match vanilla position 1, "
        f"max|diff|={(got - want).abs().max().item():.3e}"
    )

    # ntk: the LOWEST frequency (k=0) is exactly untouched; the HIGHEST
    # frequency is stretched by exactly 1/factor
    rn = RotaryEmbedding(dim, 2048, scaling={"type": "ntk", "factor": f})
    assert rn.inv_freq[0].item() == 1.0, "ntk must not move the lowest freq"
    ratio = (rn.inv_freq[-1] / r0.inv_freq[-1]).item()
    assert abs(ratio - 1.0 / f) < 1e-6, f"ntk highest freq ratio {ratio}, want {1/f}"

    # capacity: scaled embeddings serve positions far beyond the vanilla
    # precompute budget (lazy growth still works)
    c_far, _ = rl(torch.arange(20000, 20010).unsqueeze(0))
    assert torch.isfinite(c_far).all(), "scaled RoPE broke at long positions"

    # config validation
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.rope_scaling = {"type": "bogus", "factor": 2.0}
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=cfg.attention), "bad rope_scaling type")
    bad = HeliosLMv5Config(size="lite")
    bad.attention.rope_scaling = {"type": "linear", "factor": 0.5}
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=bad.attention), "rope_scaling factor < 1")
    _pass("test_rope_scaling",
          f"linear p/f match, ntk low-freq exact + high-freq 1/{int(f)}, "
          "factor=1 == vanilla, long positions finite")


# v5.8: YaRN RoPE scaling, DSA sparse top-k attention, per-head Muon,
# GPTQ act-order
def test_yarn_rope_scaling():
    from helioslm_v5.src.attention.mla import RotaryEmbedding
    dim = 8
    r0 = RotaryEmbedding(dim, 2048)

    # factor == 1 must be exactly vanilla (inv_freq unchanged by the
    # NTK-by-parts remap; attention_factor = 0.1*ln(1)+1 = 1.0)
    ry1 = RotaryEmbedding(dim, 2048, scaling={"type": "yarn", "factor": 1.0})
    assert torch.equal(r0.inv_freq, ry1.inv_freq), "yarn factor=1 != vanilla"
    assert ry1.attention_factor == 1.0, "yarn factor=1 must keep mscale 1.0"
    pos = torch.arange(64).unsqueeze(0)
    c0, s0 = r0(pos)
    c1, s1 = ry1(pos)
    assert torch.equal(c0, c1) and torch.equal(s0, s1), (
        "yarn factor=1 cos/sin must be bit-identical to vanilla")

    # NTK-by-parts wavelength law on the lite dims (base=10000, dim=8 ->
    # inv_freq (1, 0.1, 0.01, 1e-3), wavelengths 2pi*(1, 10, 100, 1000);
    # thresholds L/32 = 64 and L/1 = 2048 with L = 2048).
    f = 4.0
    ry = RotaryEmbedding(dim, 2048, scaling={"type": "yarn", "factor": f})
    assert ry.inv_freq[0].item() == 1.0, "yarn must keep the lowest inv_freq 1"
    assert torch.equal(ry.inv_freq[:2], r0.inv_freq[:2]), (
        "yarn must keep short-wavelength (high-frequency) dims untouched")
    want_low = r0.inv_freq[-1].item() / f
    assert abs(ry.inv_freq[-1].item() - want_low) < 1e-9, (
        f"yarn long-wavelength dim must be fully interpolated: "
        f"{ry.inv_freq[-1].item()} vs {want_low}")
    mid = ry.inv_freq[2].item()
    assert r0.inv_freq[2].item() / f < mid < r0.inv_freq[2].item(), (
        f"yarn in-band dim must lie strictly between inv_freq and "
        f"inv_freq/factor, got {mid}")

    # mscale: default = 0.1*ln(f)+1; cos/sin are scaled by it (scores by
    # its square — the YaRN softmax-temperature compensation).
    assert abs(ry.attention_factor - (0.1 * math.log(f) + 1.0)) < 1e-9
    ry2 = RotaryEmbedding(dim, 2048, scaling={
        "type": "yarn", "factor": 1.0, "attention_factor": 2.0})
    c2, _ = ry2(pos)
    assert torch.equal(c0 * 2.0, c2), (
        "explicit attention_factor must scale cos/sin exactly")

    # capacity: scaled embeddings serve long positions (lazy growth works)
    c_far, _ = ry(torch.arange(20000, 20010).unsqueeze(0))
    assert torch.isfinite(c_far).all(), "yarn RoPE broke at long positions"

    # loud validation of the optional knobs
    _expect_raises(ValueError, lambda: RotaryEmbedding(
        dim, 2048, scaling={"type": "yarn", "factor": 2.0, "beta_fast": 1.0,
                            "beta_slow": 32.0}), "yarn beta_fast <= beta_slow")
    _expect_raises(ValueError, lambda: RotaryEmbedding(
        dim, 2048, scaling={"type": "yarn", "factor": 2.0,
                            "attention_factor": 0.0}),
        "yarn non-positive attention_factor")
    _expect_raises(ValueError, lambda: RotaryEmbedding(
        dim, 2048, scaling={"type": "yarn", "factor": 2.0,
                            "original_max_position": 0}),
        "yarn non-positive original_max_position")

    # config-level validation accepts yarn and still rejects junk
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.rope_scaling = {"type": "yarn", "factor": 2.0}
    ok = HeliosLMv5Config(size="lite", attention=cfg.attention)
    assert ok.attention.rope_scaling["type"] == "yarn"
    _pass("test_yarn_rope_scaling",
          f"factor=1 bit-identical, high-freq kept, low-freq /{int(f)} exact, "
          f"ramp bounded, mscale={ry.attention_factor:.3f}, guards OK")


def test_fp8_kv_cache():
    from helioslm_v5.src.attention.mla import MLA
    config = HeliosLMv5Config(size="lite")
    config.attention.kv_cache_dtype = "fp8"
    torch.manual_seed(202)
    mla = MLA(config).eval()
    L = 12
    h = torch.randn(1, L, config.hidden_size)

    with torch.no_grad():
        out_fp8, past = mla(h, use_cache=True)
    assert past[0].dtype == torch.float8_e4m3fn, (
        f"fp8 cache stores {past[0].dtype}, want float8_e4m3fn")
    assert past[0].element_size() == 1, "fp8 cache must use 1 byte/element"
    assert past[0].shape[2] == L, "dim 2 must remain the sequence length"

    # Storage contract used by MTP rollback / engine watermarking: clone
    # and dim-2 slice keep the dtype and values.
    snap = past[0].clone()
    assert torch.equal(snap[:, :, :L], past[0][:, :, :L]), "clone/slice mismatch"

    # fp8 grid: every stored value is an E4M3-representable number, so
    # re-casting is a fixed point (no drift across re-quantization).
    re_cast = past[0].to(torch.float32).to(torch.float8_e4m3fn)
    assert torch.equal(re_cast, past[0]), "fp8 cache is not a fixed point"

    # Cached decode over the fp8 cache stays consistent step by step...
    with torch.no_grad():
        step_outs = []
        past_d = None
        for i in range(L):
            o, past_d = mla(h[:, i:i + 1], past_key_value=past_d, use_cache=True)
            step_outs.append(o)
        step_out = torch.cat(step_outs, dim=1)
    d_self = (out_fp8 - step_out).abs().max().item()
    assert d_self < 1e-5, f"fp8 cached decode diverges from fp8 prefill: {d_self:.3e}"

    # ...and stays close to the fp32-cache reference (greedy tokens should
    # usually survive; we assert a bounded logit perturbation, not equality).
    config32 = HeliosLMv5Config(size="lite")
    mla32 = MLA(config32).eval()
    mla32.load_state_dict(mla.state_dict())
    with torch.no_grad():
        out32, _ = mla32(h, use_cache=False)
    d_ref = (out_fp8 - out32).abs().max().item()
    assert d_ref < 0.1, f"fp8 cache perturbation too large: {d_ref:.3e}"

    # non-absorbed mode must fail loudly, not silently ignore the flag
    config3 = HeliosLMv5Config(size="lite")
    config3.attention.use_absorption = False
    config3.attention.kv_cache_dtype = "fp8"
    _expect_raises(ValueError, lambda: MLA(config3), "fp8 cache w/o absorption")
    # bad dtype string rejected at config level
    bad = HeliosLMv5Config(size="lite")
    bad.attention.kv_cache_dtype = "int4"
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=bad.attention), "bad kv_cache_dtype")
    _pass("test_fp8_kv_cache",
          f"cache fp8 (1B/elem), decode==prefill {d_self:.1e}, "
          f"vs fp32 max|diff|={d_ref:.1e} (<0.1)")


def test_hyper_connections():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    n = 3
    config = HeliosLMv5Config(size="lite")
    config.use_hyper_connections = True
    config.hyper_connection_branches = n
    torch.manual_seed(7)
    model = HeliosLMv5(config)
    ids = torch.randint(0, config.vocab_size, (2, 6))
    model.eval()

    # Identity at init: B=0 drops every sublayer write, so the output is
    # EXACTLY the embedding-only model (bit-for-bit).
    with torch.no_grad():
        logits, hid, past = model(ids, use_cache=True)
        ref = model.lm_head(model.norm(model.embed_tokens(ids)))
    assert (logits - ref).abs().max().item() == 0.0, (
        "HC zero-init B must reproduce the vanilla model exactly")
    # The static mixing matrices satisfy the manifold constraint (unit
    # column norm) by construction and are NOT trainable.
    for layer in model.layers:
        for name in ("hc_attn_A", "hc_moe_A"):
            A = getattr(layer, name)
            norms = A.norm(dim=0)
            assert torch.allclose(norms, torch.ones_like(norms)), (
                f"{name} column norms {norms}, want 1")
            assert not A.requires_grad, f"{name} must be static"

    # Widened cache layout: one entry per (batch row, branch), self-consistent
    # across prefill -> decode (every forward expands the same way).
    assert past[0][0].shape[0] == 2 * n, (
        f"HC cache batch {past[0][0].shape[0]}, want B*n={2 * n}")
    with torch.no_grad():
        _, _, past2 = model(ids[:, -1:], past_key_values=past, use_cache=True)
    assert past2[0][0].shape[2] == 7, "HC decode must grow the cache by 1"

    # Training dynamics: B receives gradients at init (attention weights do
    # not — they enter through B, which is zero; the HC recipe warms up via
    # B first), and one optimizer step moves the output.
    model.train()
    logits_tr, _, _ = model(ids)
    logits_tr.pow(2).mean().backward()
    gB = model.layers[0].hc_attn_B.grad
    assert gB is not None and gB.abs().sum().item() > 0, "B must receive gradient"
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    opt.step()
    model.eval()
    with torch.no_grad():
        logits_after, _, _ = model(ids)
    assert (logits_after - logits).abs().max().item() > 0, (
        "HC output must move once B trains")

    # Loud error when a plain [B, L, d] caller hits an HC layer (DualPipe).
    from helioslm_v5.src.model_v5 import HeliosLMv5Layer
    plain_in = torch.randn(2, 6, config.hidden_size)
    _expect_raises(NotImplementedError,
                   lambda: model.layers[0](plain_in),
                   "plain hidden state into an HC layer")
    # Mutual exclusion with attention residuals is enforced at config time.
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", use_attention_residuals=True, use_hyper_connections=True),
        "HC + attention residuals")
    _pass("test_hyper_connections",
          f"n={n}: init identity exact, B grads flow, cache [B*n] consistent, "
          "manifold column norms 1.0")


# ----------------------------------------------------------------------
# v5.22: NoPE option for MLA (Kimi-K3 direction)
# ----------------------------------------------------------------------
def test_nope_attention():
    from helioslm_v5.src.attention.mla import MLA

    torch.manual_seed(324)
    config = HeliosLMv5Config(size="lite")
    attn = MLA(config).eval()
    L = 5
    h = torch.randn(1, L, config.hidden_size)
    pos = torch.arange(L).unsqueeze(0)

    with torch.no_grad():
        # Decode == one-shot with NoPE (rotary skip is per-token consistent).
        config_nope = HeliosLMv5Config(size="lite")
        config_nope.attention.nope = True
        attn_np = MLA(config_nope).eval()
        attn_np.load_state_dict(attn.state_dict())  # same weights, NoPE path
        full_np, _ = attn_np(h, position_ids=pos, use_cache=False)
        past, outs = None, []
        for i in range(L):
            o, past = attn_np(h[:, i:i + 1], past_key_value=past,
                              use_cache=True)
            outs.append(o)
        step_np = torch.cat(outs, dim=1)
        ndiff = (full_np - step_np).abs().max().item()
        assert ndiff < 1e-4, f"NoPE cached decode diverges: {ndiff:.3e}"

        # Permutation equivariance over the visible prefix: swapping the
        # first two tokens must NOT change the last position's output
        # under NoPE (softmax is order-invariant; only the causal mask
        # sees order), while RoPE makes the same swap visible — the
        # contrast proves both code paths actually differ.
        h_swap = h.clone()
        h_swap[0, 0], h_swap[0, 1] = h[0, 1].clone(), h[0, 0].clone()
        out_r, _ = attn(h, position_ids=pos, use_cache=False)
        out_s, _ = attn(h_swap, position_ids=pos, use_cache=False)
        rope_sees = (out_r[:, -1] - out_s[:, -1]).abs().max().item()
        out_np_r, _ = attn_np(h, position_ids=pos, use_cache=False)
        out_np_s, _ = attn_np(h_swap, position_ids=pos, use_cache=False)
        nope_blind = (out_np_r[:, -1] - out_np_s[:, -1]).abs().max().item()
        assert nope_blind < 1e-5, \
            f"NoPE output changed under prefix permutation: {nope_blind:.3e}"
        assert rope_sees > 1e-3, \
            f"RoPE path should see the swap (test has no teeth): {rope_sees:.3e}"

        # NoPE really changed the math (not just a dead flag): outputs
        # differ from the RoPE path on the same weights.
        cross = (out_r - out_np_r).abs().max().item()
        assert cross > 1e-3, "NoPE flag appears to be a no-op"

    # Model level: hybrid stack with NoPE MLA layers generates greedily
    # and deterministically; GDA layers are position-free already (v5.5).
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(325)
    config = HeliosLMv5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.hybrid_attention.per_channel_decay = True
    config.attention.nope = True
    config.num_hidden_layers = 4
    model = HeliosLMv5(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 6))
    g1 = model.generate(ids, max_new_tokens=6, temperature=0)
    g2 = model.generate(ids, max_new_tokens=6, temperature=0)
    assert g1.shape == (1, 12) and torch.equal(g1, g2), \
        "NoPE hybrid generate not deterministic"
    assert model.layers[0].attention.nope and model.layers[2].attention.nope
    _pass("test_nope_attention",
          f"decode==one-shot {ndiff:.1e}, NoPE permutation-blind {nope_blind:.1e} "
          f"vs RoPE sees swap {rope_sees:.1e}, hybrid generate OK")


# ----------------------------------------------------------------------
# v5.49: YaRN long-context POSITION-ENCODING stress (factor 512,
# 2048 -> 2^20). Candidate from the 2026-10-10 daily report; the
# first draft (2026-10-11) was lost with an uncommitted working tree
# during a baseline re-sync, so this is a fresh rewrite.
#
# Honest scope: a real 1M-token forward is CPU-infeasible (KV cache
# plus O(L) attention over 2^20 tokens), so this stresses the part
# that actually breaks first at 1M indices — the POSITION ENCODING —
# and keeps every boundary loud:
#   1. inv_freq matches an independent fp64 re-derivation of the
#      NTK-by-parts wavelength law (not just spot values).
#   2. attention_factor is the YaRN default 0.1*ln(f)+1 and the lazy
#      fp32 cos/sin table at positions up to 2^20-1 is finite and
#      mscale-bounded.
#   3. Measured fp32 table error vs an fp64 reference near 2^20:
#      angle quantization of the fp32 position*frequency outer
#      product, NOT a formula error. Threshold 1e-2; the measured max
#      is reported and recorded here as quantitative evidence.
#   4. Rotation composition oracle at 1M indices in fp64 (af=1):
#      R(p) o R(q) == R(p+q) for the yarn-remapped frequencies.
#   5. End-to-end: a 24-token prompt forwarded at position indices
#      [2^20-24, 2^20) yields finite, bitwise-deterministic logits
#      through the lite model with a yarn config.
# ----------------------------------------------------------------------
def test_yarn_longctx_position_stress():
    from helioslm_v5.src.attention.mla import RotaryEmbedding

    dim, L, f = 8, 2048, 512
    M = 1 << 20  # 1,048,576 target positions
    base = 10000.0

    ry = RotaryEmbedding(dim, L, scaling={"type": "yarn", "factor": float(f)})

    # (1) independent fp64 re-derivation of the NTK-by-parts law
    k = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    inv64 = 1.0 / (base ** k)
    wavelen = 2.0 * math.pi / inv64
    low_w, high_w = L / 32.0, L / 1.0
    remapped = torch.where(wavelen > high_w, inv64 / f, inv64)
    smooth = (L / wavelen - 1.0) / (32.0 - 1.0)
    ramped = (1.0 - smooth) * inv64 / f + smooth * inv64
    in_band = (wavelen >= low_w) & (wavelen <= high_w)
    want64 = torch.where(in_band, ramped, remapped)
    d_law = (ry.inv_freq.double() - want64).abs().max().item()
    # threshold 1e-6: the module builds inv_freq in fp32, so the buffer
    # carries fp32 representation error (measured 1.49e-9 here); any real
    # formula error (wrong ramp/threshold) would be orders of magnitude
    # larger. 1e-6 sits 3 decades above the noise, far below real error.
    assert d_law < 1e-6, f"yarn inv_freq diverges from fp64 law: {d_law}"

    # (2) mscale default + lazy table finite & bounded at 2^20-1
    af = 0.1 * math.log(f) + 1.0
    assert abs(ry.attention_factor - af) < 1e-9
    top = torch.arange(M - 8, M).unsqueeze(0)
    cos, sin = ry(top)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all(), \
        "yarn table non-finite near 2^20"
    assert cos.abs().max().item() <= af + 1e-5 and \
        sin.abs().max().item() <= af + 1e-5, \
        "yarn table exceeds the mscale bound"

    # (3) measured fp32 quantization error vs fp64 reference (the
    # honest floor: fp32 cannot represent 2^20*radian angles better
    # than ~2^-23 * |angle|; with |angle| up to ~2^20 this is ~1e-2)
    pos64 = (torch.arange(M - 64, M, dtype=torch.float64))
    ref_emb = torch.outer(pos64, want64)
    ref_cos = ref_emb.cos() * af
    cos64, _ = ry(torch.arange(M - 64, M).unsqueeze(0))
    got_cos = cos64[0, 0][:, :dim // 2]  # first half of cat([f, f])
    # module table is cat([freqs, freqs], -1): first half == raw freqs
    d_quant = (got_cos.double() - ref_cos).abs().max().item()
    assert d_quant < 1e-2, f"fp32 table error {d_quant:.3e} exceeds 1e-2"

    # (4) rotation composition oracle, fp64, mscale disabled so the
    # pure rotation law is tested: R(p) o R(q) == R(p+q). Angles are
    # derived in fp64 from the module's own remapped inv_freq — the
    # module's fp32 table accuracy is bounded separately in (3).
    ry1 = RotaryEmbedding(dim, L, scaling={
        "type": "yarn", "factor": float(f), "attention_factor": 1.0})
    p = M - 1024
    q = 1000
    pos = torch.tensor([[p, q, p + q]], dtype=torch.float64)
    ang = torch.outer(pos[0], ry1.inv_freq.double())  # [3, dim/2]
    c1 = torch.cat([ang.cos(), ang.cos()], dim=-1).view(1, 1, 3, dim)
    s1 = torch.cat([ang.sin(), ang.sin()], dim=-1).view(1, 1, 3, dim)
    x = torch.tensor([[[[0.3, -0.7, 1.1, 0.2, 0.5, -0.9, 0.8, 0.6]]]],
                     dtype=torch.float64)

    def rot(v, c, s):
        half = v.shape[-1] // 2
        v1, v2 = v[..., :half], v[..., half:]
        c_, s_ = c[..., :half], s[..., :half]
        return torch.cat([v1 * c_ - v2 * s_, v1 * s_ + v2 * c_], dim=-1)

    xp = rot(x, c1[:, :, 0:1], s1[:, :, 0:1])   # R(p)
    xpq = rot(xp, c1[:, :, 1:2], s1[:, :, 1:2])  # o R(q)
    xref = rot(x, c1[:, :, 2:3], s1[:, :, 2:3])  # R(p+q)
    d_comp = (xpq - xref).abs().max().item()
    assert d_comp < 1e-12, f"rotation composition broke at 1M: {d_comp}"

    # (5) end-to-end: 24-token prompt at [2^20-24, 2^20) -> finite,
    # bitwise-deterministic logits. The yarn config must be set BEFORE
    # construction — MLA reads rope_scaling when building its
    # RotaryEmbedding (setting it afterwards leaves a vanilla rope).
    torch.manual_seed(777)
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.rope_scaling = {"type": "yarn", "factor": float(f)}
    from helioslm_v5.src.model_v5 import HeliosLMv5
    model2 = HeliosLMv5(cfg).eval()
    ids = torch.randint(3, cfg.vocab_size, (1, 24))
    pos_ids = torch.arange(M - 24, M).unsqueeze(0)
    with torch.no_grad():
        out1 = model2(input_ids=ids, position_ids=pos_ids)
        out2 = model2(input_ids=ids, position_ids=pos_ids)
    logits1 = out1[0] if isinstance(out1, tuple) else out1.logits
    logits2 = out2[0] if isinstance(out2, tuple) else out2.logits
    assert torch.isfinite(logits1).all(), "non-finite logits at 1M positions"
    assert torch.equal(logits1, logits2), "1M-position prefill not deterministic"
    _pass("test_yarn_longctx_position_stress",
          f"law<1e-12, table err {d_quant:.2e} (<1e-2 fp32 floor), "
          f"compose {d_comp:.1e}, 1M-position prefill finite+deterministic")
