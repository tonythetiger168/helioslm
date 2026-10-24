"""HeliosLM v5 test cases — hybrid linear-attention stack.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch
import torch.nn as nn

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_expect_raises, _pass)


# ----------------------------------------------------------------------
# v5.5 (F1): GatedDeltaAttention — shapes, recurrent == one-shot, decay
# gate range, StateTensor marker, masked tokens never write state
# ----------------------------------------------------------------------
def test_linear_attention():
    from evastar.src.attention.linear_attention import (
        GatedDeltaAttention, StateTensor, is_recurrent_state)

    config = EvastarV5Config(size="lite")
    torch.manual_seed(140)
    attn = GatedDeltaAttention(config).eval()
    B, L = 2, 7
    h = torch.randn(B, L, config.hidden_size)
    H = config.hybrid_attention.linear_num_heads
    D = config.hybrid_attention.linear_head_dim

    with torch.no_grad():
        full_out, present = attn(h, use_cache=True)
        assert full_out.shape == (B, L, config.hidden_size)
        assert torch.isfinite(full_out).all()
        # Cache: single fixed-size state tensor carrying the marker.
        assert len(present) == 1
        state = present[0]
        assert isinstance(state, StateTensor) and is_recurrent_state(state)
        assert state.shape == (B, H, D, D)

        # Token-by-token decode must match the one-shot forward (same
        # sequential op order in both paths).
        past, outs = None, []
        for i in range(L):
            o, past = attn(h[:, i:i + 1], past_key_value=past, use_cache=True)
            outs.append(o)
        step_out = torch.cat(outs, dim=1)
        diff = (full_out - step_out).abs().max().item()
        assert diff < 1e-5, f"recurrent decode != one-shot forward: {diff:.3e}"
        sdiff = (state - past[0]).abs().max().item()
        assert sdiff < 1e-6, f"final state mismatch: {sdiff:.3e}"

        # Decay gate: strictly inside (0, 1), and the +4 bias init keeps it
        # near 1 (slow decay) at initialization.
        g = attn.decay_gate(h)
        assert g.shape == (B, L, H)
        assert bool((g > 0).all()) and bool((g < 1).all()), \
            f"decay gate out of (0,1): [{g.min()}, {g.max()}]"
        fresh = attn.decay_gate(torch.zeros(1, 1, config.hidden_size))
        assert float(fresh.min()) > 0.9, \
            f"decay at init should be near 1, got {float(fresh.min()):.4f}"

        # Masked (pad) tokens must not write into the state: masking the
        # tail of a sequence == running the truncated sequence.
        mask = torch.ones(B, L)
        mask[0, 5:] = 0
        _, present_m = attn(h, attention_mask=mask, use_cache=True)
        _, present_t = attn(h[0:1, :5], use_cache=True)
        mdiff = (present_m[0][0] - present_t[0][0]).abs().max().item()
        assert mdiff < 1e-6, f"pad tokens polluted the state: {mdiff:.3e}"

    # Bad cache arity / shape fails loudly.
    bad = (torch.zeros(B, H, D, D), torch.zeros(B, H, D, D))
    _expect_raises(ValueError,
                   lambda: attn(h[:, :1], past_key_value=bad, use_cache=True),
                   "2-tuple cache for a linear-attention layer")
    _pass("test_linear_attention",
          f"decode==one-shot max|diff|={diff:.2e}, state match {sdiff:.2e}, "
          f"decay in (0,1) (init {float(fresh.min()):.3f}), pad no-write, "
          f"state {tuple(state.shape)} marked")


# ----------------------------------------------------------------------
# v5.5 (F2): hybrid interleave — layer pattern, shapes, cached decode ==
# full forward, generate, full-size config defaults, engine compatibility
# ----------------------------------------------------------------------
def test_hybrid_model():
    from evastar.src.model_v5 import EvastarV5
    from evastar.src.attention.mla import MLA
    from evastar.src.attention.linear_attention import (
        GatedDeltaAttention, is_recurrent_state)

    # every=3, 4 layers -> MLA iff i == 0 or i % 3 == 2:
    # [MLA, GDA, MLA, GDA]; layer 0 always MLA.
    torch.manual_seed(141)
    config = EvastarV5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    model = EvastarV5(config).eval()

    pattern = [type(l.attention) for l in model.layers]
    assert pattern == [MLA, GatedDeltaAttention, MLA, GatedDeltaAttention], \
        f"unexpected interleave pattern: {[t.__name__ for t in pattern]}"

    ids = torch.randint(3, config.vocab_size, (2, 8))
    with torch.no_grad():
        logits, hidden, past = model(ids, use_cache=True)
        assert logits.shape == (2, 8, config.vocab_size)
        assert hidden.shape == (2, 8, config.hidden_size)
        assert len(past) == 4
        # MLA layers: dim-2-sequence caches; GDA layers: marked state.
        assert past[0][0].shape[2] == 8 and not is_recurrent_state(past[0][0])
        assert is_recurrent_state(past[1][0]) and past[1][0].dim() == 4
        assert past[2][0].shape[2] == 8 and not is_recurrent_state(past[2][0])
        assert is_recurrent_state(past[3][0])

        # Cached token-by-token decode == full forward (whole hybrid stack).
        full = logits
        past_i, outs = None, []
        for i in range(8):
            o, _, past_i = model(ids[:, i:i + 1], past_key_values=past_i,
                                 use_cache=True)
            outs.append(o)
        dec = torch.cat(outs, dim=1)
        diff = (full - dec).abs().max().item()
        assert diff < 1e-4, f"hybrid cached decode diverged: {diff:.3e}"

    g1 = model.generate(ids, max_new_tokens=5, temperature=0)
    g2 = model.generate(ids, max_new_tokens=5, temperature=0)
    assert g1.shape == (2, 13) and torch.equal(g1, g2)

    # Engine: unequal prompts run unpadded (state cannot absorb pad
    # prefixes) and still match per-request greedy exactly.
    from evastar.src.inference.vllm_engine import VLLMEngine
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    assert engine._has_recurrent_state
    prompts = [[5, 100, 200, 7], [42, 900]]
    rids = [engine.add_request(p, max_new_tokens=5, temperature=0.0)
            for p in prompts]
    results = engine.run()
    assert all(engine_r.prompt_pad == 0
               for engine_r in engine.finished_requests), \
        "hybrid engine must not use the pad-prefix watermark"
    for p, rid in zip(prompts, rids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=5,
                             temperature=0)[0, len(p):].tolist()
        got = results[rid]
        assert ref[:len(got)] == got, \
            f"hybrid engine request {rid}: {got} != greedy {ref[:len(got)]}"

    # Config: validation of full_attention_every, and size-based defaults.
    def _bad_every():
        c = EvastarV5Config(size="lite")
        c.hybrid_attention.enabled = True
        c.hybrid_attention.full_attention_every = 1
        c._validate()
    _expect_raises(ValueError, _bad_every, "full_attention_every=1")

    full = EvastarV5Config(size="full")  # validation only (no model build)
    assert full.hybrid_attention.enabled is True
    assert full.moe.latent_dim == 1024
    assert full.use_attention_residuals is True
    lite = EvastarV5Config(size="lite")
    assert lite.hybrid_attention.enabled is False
    assert lite.moe.latent_dim is None
    assert lite.use_attention_residuals is False
    _pass("test_hybrid_model",
          f"pattern [MLA,GDA,MLA,GDA], cached-decode max|diff|={diff:.2e}, "
          f"generate OK, engine hybrid==greedy (unpadded), size defaults OK")


# ----------------------------------------------------------------------
# v5.5 (F5a): attention residuals — on/off changes output, per-layer gate
# exists, is learnable and receives gradient (layers >= 1)
# ----------------------------------------------------------------------
def test_attention_residuals():
    from evastar.src.model_v5 import EvastarV5

    torch.manual_seed(145)
    cfg_on = EvastarV5Config(size="lite")
    cfg_on.use_attention_residuals = True
    model_on = EvastarV5(cfg_on)
    torch.manual_seed(145)
    cfg_off = EvastarV5Config(size="lite")  # v5.4 default: off
    model_off = EvastarV5(cfg_off)

    for i, layer in enumerate(model_on.layers):
        assert isinstance(layer.attn_res_gate, torch.nn.Parameter)
        assert layer.attn_res_gate.requires_grad
        assert abs(float(layer.attn_res_gate.detach()) - 0.1) < 1e-6
    assert model_off.layers[0].attn_res_gate is None

    ids = torch.randint(3, cfg_on.vocab_size, (2, 6))
    logits_on, _, _ = model_on(ids)
    logits_off, _, _ = model_off(ids)
    delta = (logits_on - logits_off).abs().max().item()
    assert delta > 1e-4, \
        f"attention residuals had no effect on the output ({delta:.3e})"

    model_on.zero_grad(set_to_none=True)
    model_on(ids)[0].sum().backward()
    g0 = model_on.layers[0].attn_res_gate.grad
    g1 = model_on.layers[1].attn_res_gate.grad
    assert g0 is not None and g1 is not None
    # Layer 0 injects gate * zeros (empty accumulator) -> zero grad by
    # construction; deeper layers must receive a real gradient.
    assert g1.abs().item() > 0, "attention-residual gate got no gradient"
    _pass("test_attention_residuals",
          f"on/off output max|delta|={delta:.3e}, gates init 0.1, "
          f"layer1 gate |grad|={g1.abs().item():.3e}")


# ----------------------------------------------------------------------
# v5.6: hybrid model SUPPORTS packed position_ids — the recurrent state is
# zeroed at each document boundary, giving exact cross-document isolation
# (mirrors the MLA packed-isolation test); pure-MLA models unaffected;
# decode-with-past at a position restart still raises loudly
# ----------------------------------------------------------------------
def test_hybrid_packed_positions():
    from evastar.src.model_v5 import EvastarV5

    torch.manual_seed(172)
    config = EvastarV5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    hybrid = EvastarV5(config).eval()

    pos_packed = torch.tensor([[0, 1, 2, 0, 1, 2]])  # two packed documents
    doc_a, doc_b = [5, 100, 200], [42, 900, 7]
    ids_p = torch.tensor([doc_a + doc_b])
    ids_p2 = ids_p.clone()
    ids_p2[0, :3] = torch.tensor([7, 42, 5])

    with torch.no_grad():
        # Perturbing document A must leave document B's logits unchanged
        # (state zeroed at the boundary — no cross-document leak).
        la, _, _ = hybrid(ids_p, position_ids=pos_packed)
        lb, _, _ = hybrid(ids_p2, position_ids=pos_packed)
        # Packed documents must match their solo forward exactly: the
        # boundary reset makes packing equivalent to separate sequences.
        solo_b, _, _ = hybrid(torch.tensor([doc_b]))
        packed_b = la[0, 3:]

    leak = (la[0, 3:] - lb[0, 3:]).abs().max().item()
    assert leak < 1e-5, \
        f"hybrid packed isolation failed: doc B changed by {leak:.3e}"
    solo_diff = (packed_b - solo_b).abs().max().item()
    assert solo_diff < 1e-4, \
        f"hybrid packed doc B diverges from solo forward ({solo_diff:.3e})"

    # Normal prefill (default 0..T-1) and cached decode (past_len..,
    # monotone) must NOT trip the boundary logic.
    with torch.no_grad():
        logits, _, past = hybrid(ids_p, use_cache=True)
        step, _, past = hybrid(ids_p[:, -1:], past_key_values=past,
                               use_cache=True)
    assert logits.shape == (1, 6, config.vocab_size)
    assert step.shape == (1, 1, config.vocab_size)

    # A position restart while a non-empty cache is supplied (cached decode
    # at a document boundary) is contradictory and still raises loudly.
    bad_pos = torch.tensor([[2, 0]])  # restart with past present
    _expect_raises(ValueError,
                   lambda: hybrid(torch.tensor([[11, 12]]),
                                  past_key_values=past,
                                  position_ids=bad_pos),
                   "hybrid cached decode with a position restart")

    # Pure-MLA models are unaffected: packed positions still give exact
    # cross-document isolation (v5.4 B1 behaviour preserved).
    torch.manual_seed(173)
    mla_model = EvastarV5(EvastarV5Config(size="lite")).eval()
    with torch.no_grad():
        la, _, _ = mla_model(ids_p, position_ids=pos_packed)
        lb, _, _ = mla_model(ids_p2, position_ids=pos_packed)
    leak_mla = (la[0, 3:] - lb[0, 3:]).abs().max().item()
    assert leak_mla < 1e-5, \
        f"pure-MLA packed isolation regressed: doc B changed by {leak_mla:.3e}"
    _pass("test_hybrid_packed_positions",
          f"hybrid packed isolation exact (leak {leak:.2e}), packed==solo "
          f"({solo_diff:.2e}), decode+restart raises, prefill/decode OK, "
          f"pure-MLA isolation intact (leak {leak_mla:.2e})")


# ----------------------------------------------------------------------
# v5.5 (m6/m7 regression): dtype-aware decay clamp (fp16 upper bound must
# stay strictly below 1.0) and a loud error for seq_len == 0
# ----------------------------------------------------------------------
def test_linear_attention_guards():
    from evastar.src.attention.linear_attention import GatedDeltaAttention

    config = EvastarV5Config(size="lite")
    torch.manual_seed(174)
    attn = GatedDeltaAttention(config).eval()

    # m6: deep saturation (huge pre-activations -> sigmoid ~ 1).
    big = torch.randn(2, 5, config.hidden_size) * 1e4
    with torch.no_grad():
        g32 = attn.decay_gate(big)
    # fp32 keeps the historical [1e-6, 1 - 1e-6] bounds (fp32(1e-6) is
    # marginally below the decimal value, hence the 1e-12 slack).
    assert abs(g32.max().item() - (1.0 - 1e-6)) < 1e-7, \
        f"fp32 clamp bound changed: {g32.max().item()}"
    assert g32.min().item() >= 1e-6 - 1e-12

    # fp16: a fixed 1 - 1e-6 bound would round to exactly 1.0; the
    # dtype-aware bound is 1 - eps(fp16) = 0.9990234375, strictly < 1.
    attn16 = GatedDeltaAttention(config).to(torch.float16).eval()
    with torch.no_grad():
        g16 = attn16.decay_gate(torch.randn(2, 5, config.hidden_size,
                                            dtype=torch.float16))
    eps16 = torch.finfo(torch.float16).eps
    assert bool((g16 < 1).all()), \
        f"fp16 decay gate reached 1.0 (clamp rounded away): {g16.max().item()}"
    assert g16.max().item() <= 1.0 - eps16 + 1e-7, \
        f"fp16 upper bound not dtype-aware: {g16.max().item()}"

    # m7: seq_len == 0 raises a clear ValueError, not torch.stack([])'s
    # cryptic RuntimeError.
    _expect_raises(ValueError,
                   lambda: attn(torch.zeros(1, 0, config.hidden_size)),
                   "GatedDeltaAttention with seq_len=0")
    _pass("test_linear_attention_guards",
          f"fp32 bound 1-1e-6 kept, fp16 bound {g16.max().item():.6f} "
          f"(= 1-eps16, strictly < 1), seq_len=0 raises ValueError")


# ----------------------------------------------------------------------
# v5.21: KDA-style per-channel decay gate — fine-grained eraser
# ----------------------------------------------------------------------
def test_per_channel_decay():
    from evastar.src.attention.linear_attention import (
        GatedDeltaAttention, StateTensor)

    config = EvastarV5Config(size="lite")
    config.hybrid_attention.per_channel_decay = True
    torch.manual_seed(221)
    attn = GatedDeltaAttention(config).eval()
    B, L = 2, 7
    h = torch.randn(B, L, config.hidden_size)
    H = config.hybrid_attention.linear_num_heads
    D = config.hybrid_attention.linear_head_dim

    with torch.no_grad():
        # Gate width doubles role: one sigmoid per head per key channel.
        assert attn.g_proj.out_features == H * D, \
            f"per-channel gate width {attn.g_proj.out_features} != H*D {H * D}"
        assert attn.decay_bias.shape == (H * D,)

        full_out, present = attn(h, use_cache=True)
        assert full_out.shape == (B, L, config.hidden_size)
        assert isinstance(present[0], StateTensor) \
            and present[0].shape == (B, H, D, D)

        # Gate shape [B, seq, H, Dk] and strictly inside (0, 1).
        g = attn.decay_gate(h)
        assert g.shape == (B, L, H, D), f"per-channel gate shape {g.shape}"
        assert bool((g > 0).all()) and bool((g < 1).all()), \
            f"per-channel decay out of (0,1): [{g.min()}, {g.max()}]"
        fresh = attn.decay_gate(torch.zeros(1, 1, config.hidden_size))
        assert float(fresh.min()) > 0.9, \
            f"decay at init should be near 1, got {float(fresh.min()):.4f}"

        # Token-by-token decode == one-shot forward (same op order).
        past, outs = None, []
        for i in range(L):
            o, past = attn(h[:, i:i + 1], past_key_value=past, use_cache=True)
            outs.append(o)
        step_out = torch.cat(outs, dim=1)
        diff = (full_out - step_out).abs().max().item()
        assert diff < 1e-5, f"per-channel decode != one-shot: {diff:.3e}"

        # Masked tail == truncated sequence (no write, no forgetting).
        mask = torch.ones(B, L)
        mask[0, 5:] = 0
        _, present_m = attn(h, attention_mask=mask, use_cache=True)
        _, present_t = attn(h[0:1, :5], use_cache=True)
        mdiff = (present_m[0][0] - present_t[0][0]).abs().max().item()
        assert mdiff < 1e-6, f"pad tokens polluted the state: {mdiff:.3e}"

        # Per-row forgetting independence: within every head, make half
        # the key channels forget almost instantly (bias -6 -> decay ~=
        # 0.0025) and half barely forget (bias +6 -> decay ~= 0.9975).
        # After one more token, the post-decay remainder of the previous
        # state must vanish in the fast rows and survive in the slow rows
        # of EVERY head — a per-head scalar gate cannot express that split.
        per_head = torch.tensor(
            [-6.0] * (D // 2) + [6.0] * (D - D // 2))       # [D]
        attn.decay_bias.copy_(per_head.repeat(H))           # [H*D]
        ids_a = h[:, :1]
        _, pa = attn(ids_a, use_cache=True)
        state_a = pa[0].clone()
        ids_b = h[:, 1:2]
        out_b, pb = attn(ids_b, past_key_value=pa, use_cache=True)
        state_b = pb[0]
        k_b = attn.k_proj(ids_b).view(B, 1, H, D).transpose(1, 2)
        k_b = torch.nn.functional.normalize(k_b, dim=-1)   # [B, H, 1, D]
        v_b = attn.v_proj(ids_b).view(B, 1, H, D).transpose(1, 2)
        k_b, v_b = k_b[:, :, 0], v_b[:, :, 0]              # [B, H, D]
        r_b = torch.einsum("bhk,bhkv->bhv", k_b, state_a)  # retrieval
        delta_b = v_b - r_b
        write_b = k_b.unsqueeze(-1) * delta_b.unsqueeze(-2)  # [B, H, D, D]
        remainder = state_b - write_b                      # decayed old state
        expected = attn.decay_gate(ids_b)[:, 0][:, :, :, None] * state_a
        rd = (remainder - expected).abs().max().item()
        assert rd < 1e-5, f"per-channel decay semantics wrong: {rd:.3e}"
        smax = state_a.abs().max().item()
        fast = remainder[:, :, :D // 2].abs().max().item()
        slow = remainder[:, :, D // 2:].abs().max().item()
        slow_ref = state_a[:, :, D // 2:].abs().max().item()
        # fast rows keep only decay(-6) ~= 0.25% of the old state; slow
        # rows keep decay(+6) ~= 99.8% of it. Thresholds are relative to
        # the state magnitude, not absolute (a per-head scalar gate would
        # give every row the same factor, so the fast/slow split itself
        # is the property under test).
        assert fast < 0.01 * smax + 1e-6, \
            f"fast-forget rows kept too much: {fast:.3e} (state max {smax:.3e})"
        assert slow > 0.99 * slow_ref, \
            f"slow-forget rows lost state: {slow:.3e} vs ref {slow_ref:.3e}"
        assert slow / (fast + 1e-12) > 50.0, \
            f"per-channel split too weak: slow/fast = {slow / fast:.1f}"

        # Packed document boundary still zeroes the state with the
        # per-channel gate (position restart mid-sequence).
        pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
        _, present_p = attn(h[0:1], position_ids=pos, use_cache=True)
        # Recompute the tail document alone and compare final states.
        tail_start = 4
        _, present_tail = attn(h[0:1, tail_start:], use_cache=True)
        pdiff = (present_p[0][0] - present_tail[0][0]).abs().max().item()
        assert pdiff < 1e-6, \
            f"packed boundary reset broken with per-channel decay: {pdiff:.3e}"

        # bf16: the dtype-aware clamp keeps decay strictly below 1 even
        # with the wider per-channel gate (same m6 margin as the scalar
        # gate; a fixed 1 - 1e-6 bound would round to 1.0 in fp16).
        attn16 = GatedDeltaAttention(config).to(torch.float16).eval()
        g16 = attn16.decay_gate(torch.randn(2, 5, config.hidden_size,
                                            dtype=torch.float16))
        assert g16.shape == (2, 5, H, D)
        assert bool((g16 < 1).all()), \
            f"bf16 per-channel decay reached 1.0: {g16.max().item()}"

    _pass("test_per_channel_decay",
          f"gate [B,L,H,D] in (0,1), decode==one-shot {diff:.2e}, "
          f"pad no-write, per-row forget split "
          f"(fast {fast:.1e} / slow {slow:.1e} of state {smax:.1e}), "
          f"packed reset {pdiff:.2e}, bf16 bound ok")
