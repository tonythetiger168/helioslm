"""HeliosLM v5 test cases — full lite model + MTP speculative decoding.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_expect_raises, _lite_model, _pass)


# ----------------------------------------------------------------------
# Full lite model: forward triple, greedy determinism, MTP path, validation
# ----------------------------------------------------------------------
def test_v5_model():
    from helioslm_v5.src.model_v5 import HeliosLMv5

    # Config validation happens eagerly in __post_init__.
    _expect_raises(ValueError, lambda: HeliosLMv5Config(size="huge"),
                   "unknown size")
    _expect_raises(ValueError,
                   lambda: HeliosLMv5Config(size="full", hidden_size=100),
                   "hidden_size not divisible by num_attention_heads")

    model, config = _lite_model(seed=103)
    a = config.attention
    ids = torch.randint(3, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits, hidden, past = model(ids)
        assert logits.shape == (2, 8, config.vocab_size)
        assert hidden.shape == (2, 8, config.hidden_size)
        assert past is None
        assert torch.isfinite(logits).all()

        _, _, past = model(ids, use_cache=True)
        assert len(past) == config.num_hidden_layers
        if a.use_absorption:
            c_kv, k_rope = past[0]
            assert c_kv.shape == (2, 1, 8, a.kv_latent_dim)
            assert k_rope.shape == (2, 1, 8, a.rope_head_dim)
        else:
            k_nope, k_rope, v = past[0]
            assert k_nope.shape == (2, a.num_attention_heads, 8, a.no_rope_head_dim)
            assert k_rope.shape == (2, 1, 8, a.rope_head_dim)
            assert v.shape == (2, a.num_attention_heads, 8, a.v_head_dim)

    # Batched greedy generation is deterministic.
    g1 = model.generate(ids, max_new_tokens=5, temperature=0)
    g2 = model.generate(ids, max_new_tokens=5, temperature=0)
    assert torch.equal(g1, g2), "greedy batch=2 generation is not deterministic"

    # use_mtp=True must produce exactly the plain greedy sequence (mechanism
    # correctness; acceptance rate of an untrained model is NOT asserted).
    ids1 = ids[:1]
    plain = model.generate(ids1, max_new_tokens=6, temperature=0)
    mtp_seq = model.generate(ids1, max_new_tokens=6, temperature=0, use_mtp=True)
    assert isinstance(mtp_seq, torch.Tensor), \
        f"generate(use_mtp=True) must return a token tensor, got {type(mtp_seq)}"
    n = mtp_seq.shape[1]
    assert n <= plain.shape[1] and torch.equal(plain[:, :n], mtp_seq), \
        "MTP speculative decode diverged from plain greedy"

    # use_mtp with batch > 1 (v5.2): supported, and must match per-row plain
    # greedy exactly. Rows that stop early (EOS) are right-padded with
    # pad_token_id, so compare up to each row's committed length.
    mtp_batch = model.generate(ids, max_new_tokens=6, temperature=0,
                               use_mtp=True)
    plain_batch = model.generate(ids, max_new_tokens=6, temperature=0)
    assert mtp_batch.shape[0] == ids.shape[0] == plain_batch.shape[0]
    prompt_len = ids.shape[1]
    for r in range(ids.shape[0]):
        row = mtp_batch[r]
        # committed length: prompt + tokens up to and including EOS
        # (trailing pad_token_id entries are the early-stop padding).
        real = prompt_len + 6
        eos_hits = (row[prompt_len:] == config.eos_token_id).nonzero()
        if len(eos_hits) > 0:
            real = prompt_len + int(eos_hits[0].item()) + 1
        assert torch.equal(row[:real], plain_batch[r, :real]), \
            f"batched MTP row {r} diverged from plain greedy"
        # Every position after the committed length (past the first EOS)
        # must be exactly pad_token_id — checked element by element.
        tail = row[real:]
        assert bool((tail == config.pad_token_id).all()), \
            f"row {r}: non-pad tokens after EOS: {tail.tolist()}"

    _pass("test_v5_model",
          f"forward triple OK, greedy deterministic, mtp==greedy (len {n}), "
          f"config validation raises, mtp batch>1 == per-row greedy")


# ----------------------------------------------------------------------
# MTP: weight sharing, causality, MTPDecoder.generate plumbing
# ----------------------------------------------------------------------
def test_mtp():
    from helioslm_v5.src.inference.mtp import MTPDecoder

    model, config = _lite_model(seed=104)
    mtp_mod = model.mtp_modules[0]

    # Weight sharing: MTP modules must hold the main model's own modules.
    assert mtp_mod.embed_tokens is model.embed_tokens, "embed_tokens not shared"
    assert mtp_mod.lm_head is model.lm_head, "lm_head not shared"

    # Causality (training path, module_index=0 -> d=1): position t fuses
    # hidden[t] with ids[t+1]; changing ids at positions >= 5 may only affect
    # outputs at positions t >= 4.
    L = 10
    hidden = torch.randn(1, L, config.hidden_size)
    ids_a = torch.randint(3, config.vocab_size, (1, L))
    ids_b = ids_a.clone()
    ids_b[0, 5:] = (ids_b[0, 5:] + 1) % config.vocab_size  # differs from pos 5 on
    with torch.no_grad():
        la = mtp_mod(hidden, ids_a)
        lb = mtp_mod(hidden, ids_b)
    assert la.shape == (1, L - 2, config.vocab_size), \
        f"training-path logits shape {la.shape}, expected (1, {L - 2}, vocab)"
    assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-6), \
        "future token ids leaked into earlier-position outputs"

    # MTPDecoder.generate: MTPGenerateResult plumbing + correct length
    # (acceptance of an untrained model may be 0 — only the range is checked).
    decoder = MTPDecoder(model, model.mtp_modules, config)
    ids = torch.randint(3, config.vocab_size, (1, 6))
    res = decoder.generate(ids, max_new_tokens=8, temperature=0)
    total = res.sequences.shape[1]
    assert 6 < total <= 6 + 8, f"generated length {total} out of bounds"
    if total < 14:  # stopped early -> must have emitted EOS
        assert res.sequences[0, -1].item() == config.eos_token_id
    assert 0.0 <= res.acceptance_rate <= 1.0
    assert res.num_rounds >= 1
    assert res.num_accepted <= res.num_drafted
    _pass("test_mtp",
          f"weights shared, causal, generate len={total} "
          f"acceptance={res.acceptance_rate:.3f} ({res.num_accepted}/"
          f"{res.num_drafted}, {res.num_rounds} rounds)")


# ----------------------------------------------------------------------
# v5.5 (F1+F2): MTP speculative decoding on a HYBRID model — recurrent
# state clone/rollback restores the exact pre-speculation state
# ----------------------------------------------------------------------
def test_mtp_hybrid_rollback():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.inference.mtp import MTPDecoder
    from helioslm_v5.src.attention.linear_attention import is_recurrent_state

    torch.manual_seed(147)
    config = HeliosLMv5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    model = HeliosLMv5(config).eval()

    # --- unit level: clone -> speculate -> restore -> replay is exact ---
    ids = torch.randint(3, config.vocab_size, (1, 7))
    with torch.no_grad():
        _, _, p5 = model(ids[:, :5], use_cache=True)
        p5_state = p5[1][0].clone()  # linear layer state after 5 tokens
        # Speculative +2 tokens from p5 (must NOT mutate p5).
        _, _, p7 = model(ids[:, 5:7], past_key_values=p5, use_cache=True)
        assert torch.equal(p5[1][0], p5_state), \
            "speculative forward mutated the input state cache in place"
        # _row_past with a truncation length: MLA cache truncated, state NOT.
        rp = MTPDecoder._row_past(p7, 0, length=6)
        assert rp[0][0].shape[2] == 6, "MLA cache not truncated to 6"
        assert rp[1][0].shape == p7[1][0].shape \
            and torch.equal(rp[1][0], p7[1][0]), \
            "state cache must never be dim-2 truncated"
        # Reject the speculation: restore the clone of p5 and replay the
        # committed token ids[:, 5].
        clone = [tuple(t.clone() for t in lp) for lp in p5]
        _, _, p6r = model(ids[:, 5:6], past_key_values=clone, use_cache=True)
        # Reference: a fresh sequential decode to 6 tokens.
        _, _, p6d = model(ids[:, :6], use_cache=True)
        s_diff = (p6r[1][0] - p6d[1][0]).abs().max().item()
        k_diff = (p6r[0][0] - p6d[0][0]).abs().max().item()
        assert s_diff < 1e-6, f"restored+replayed state diverges: {s_diff:.3e}"
        # The MLA cache threshold is 1e-5, not 1e-6: the restored lower-layer
        # GDA states differ from the fresh forward at float32 reassociation
        # noise (one-shot vs stepwise recurrence), and that noise propagates
        # into the MLA layers' c_kv projections (~1.5e-6 observed). The
        # suite's own hybrid decode test allows 1e-4 for the same reason.
        assert k_diff < 1e-5, f"restored+replayed MLA cache diverges: {k_diff:.3e}"

    # --- end to end: MTP speculative decode == plain greedy on hybrid ---
    decoder = MTPDecoder(model, model.mtp_modules, config)
    ids = torch.randint(3, config.vocab_size, (1, 6))
    res = decoder.generate(ids, max_new_tokens=8, temperature=0)
    plain = model.generate(ids, max_new_tokens=8, temperature=0)
    n = res.sequences.shape[1]
    assert torch.equal(plain[:, :n], res.sequences), \
        "hybrid MTP decode diverged from plain greedy"
    assert res.num_drafted > 0
    assert res.num_accepted < res.num_drafted, \
        "test expected at least one rejection (rollback path) to occur"
    assert decoder._past_has_state(
        [tuple(t.clone() for t in lp) for lp in p5])
    _pass("test_mtp_hybrid_rollback",
          f"replay-exactness state {s_diff:.2e}/mla {k_diff:.2e}, "
          f"MTP==greedy len {n}, acceptance {res.acceptance_rate:.3f} "
          f"({res.num_accepted}/{res.num_drafted} drafted, rejections "
          "rolled back)")


# ----------------------------------------------------------------------
# v5.21: MTP speculative rollback stays exact with the per-channel gate
# ----------------------------------------------------------------------
def test_mtp_per_channel_rollback():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.inference.mtp import MTPDecoder

    torch.manual_seed(222)
    config = HeliosLMv5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.hybrid_attention.per_channel_decay = True
    config.num_hidden_layers = 4
    model = HeliosLMv5(config).eval()

    # Unit level: speculate from a cloned state, reject, restore+replay —
    # the per-channel gate must not break the restore contract.
    ids = torch.randint(3, config.vocab_size, (1, 7))
    with torch.no_grad():
        _, _, p5 = model(ids[:, :5], use_cache=True)
        p5_state = p5[1][0].clone()
        _, _, p7 = model(ids[:, 5:7], past_key_values=p5, use_cache=True)
        assert torch.equal(p5[1][0], p5_state), \
            "speculative forward mutated the input state cache in place"
        clone = [tuple(t.clone() for t in lp) for lp in p5]
        _, _, p6r = model(ids[:, 5:6], past_key_values=clone, use_cache=True)
        _, _, p6d = model(ids[:, :6], use_cache=True)
        s_diff = (p6r[1][0] - p6d[1][0]).abs().max().item()
        assert s_diff < 1e-6, \
            f"per-channel restored+replayed state diverges: {s_diff:.3e}"

    # End to end: MTP speculative decode == plain greedy on a hybrid model
    # whose linear layers run the per-channel gate.
    decoder = MTPDecoder(model, model.mtp_modules, config)
    ids = torch.randint(3, config.vocab_size, (1, 6))
    res = decoder.generate(ids, max_new_tokens=8, temperature=0)
    plain = model.generate(ids, max_new_tokens=8, temperature=0)
    n = res.sequences.shape[1]
    assert torch.equal(plain[:, :n], res.sequences), \
        "per-channel hybrid MTP decode diverged from plain greedy"
    assert res.num_drafted > 0
    _pass("test_mtp_per_channel_rollback",
          f"replay state {s_diff:.2e}, MTP==greedy len {n}, "
          f"acceptance {res.acceptance_rate:.3f} "
          f"({res.num_accepted}/{res.num_drafted})")


# ----------------------------------------------------------------------
# v5.3: QuantizationManager re-binds MTP's shared lm_head/embed_tokens
# ----------------------------------------------------------------------
def test_mtp_rebind_after_quantization():
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, QuantizationManager)

    model, config = _lite_model(seed=133)
    QuantizationManager(method="awq").quantize_model(model, group_size=32)
    assert isinstance(model.lm_head, AWQLinear), \
        f"lm_head should be AWQLinear after quantization, got {type(model.lm_head)}"
    for i, mtp in enumerate(model.mtp_modules):
        assert mtp.lm_head is model.lm_head, \
            f"mtp_modules[{i}].lm_head still points at the pre-quantization module"
        assert mtp.embed_tokens is model.embed_tokens, \
            f"mtp_modules[{i}].embed_tokens not re-bound"
    ids = torch.randint(3, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _, _ = model(ids)
    assert torch.isfinite(logits).all(), \
        "quantized model forward produced NaN/Inf"
    _pass("test_mtp_rebind_after_quantization",
          "lm_head/embed_tokens re-bound to quantized modules, forward finite")
