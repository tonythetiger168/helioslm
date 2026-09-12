"""HeliosLM v5.4 test suite (CPU, lite config).

Rewritten for v5.1 after the full code review and extended since (v5.2: true
GPTQ calibration, audio sliding window; v5.3: AWQ calibration repair;
v5.4: packed-position cross-document isolation, input-validation coverage):
every test exercises the real module with numerical assertions (not just
shapes), and any failure makes the process exit non-zero.

Usage (from the repository root, i.e. the directory that CONTAINS the
``helioslm_v5`` package):

    cd /mnt/agents/output
    python -m helioslm_v5.tests.test_v5

The project root is derived from this file's location, so direct execution
(``python helioslm_v5/tests/test_v5.py``) works as well. Exit code is 0 iff
all tests pass; failures print a FAIL line plus the full traceback and the
final exit code is 1.
"""
import math
import sys
import traceback
from pathlib import Path

# Derive the repo root from this file: <root>/helioslm_v5/tests/test_v5.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config

RESULTS = []


def _pass(name, detail=""):
    RESULTS.append((name, True))
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name):
    RESULTS.append((name, False))
    print(f"[FAIL] {name}")
    traceback.print_exc()


def _expect_raises(exc_type, fn, what):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} from {what}")


def _lite_model(seed):
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(seed)
    config = HeliosLMv5Config(size="lite")
    return HeliosLMv5(config).eval(), config


# ----------------------------------------------------------------------
# MLA: cached decode == full forward; cache growth; truthful size numbers
# (runs for BOTH cache modes: expanded and absorbed)
# ----------------------------------------------------------------------
def _check_mla_mode(use_absorption):
    from helioslm_v5.src.attention.mla import MLA
    config = HeliosLMv5Config(size="lite")
    config.attention.use_absorption = use_absorption
    torch.manual_seed(101)
    mla = MLA(config).eval()
    L = 10
    h = torch.randn(1, L, config.hidden_size)

    with torch.no_grad():
        full_out, _ = mla(h, use_cache=False)
        past = None
        step_outs, growth = [], []
        for i in range(L):
            out, past = mla(h[:, i:i + 1], past_key_value=past, use_cache=True)
            step_outs.append(out)
            growth.append(past[0].shape[2])  # seq dim (dim 2 in both layouts)
        step_out = torch.cat(step_outs, dim=1)

    diff = (full_out - step_out).abs().max().item()
    assert diff < 1e-4, f"cached decode diverges from full forward: {diff:.3e}"
    assert growth == list(range(1, L + 1)), f"cache did not grow by 1/step: {growth}"

    # get_kv_cache_size must match the actual cache tensors (per token).
    # Generic over both layouts: every cache tensor is [B, n, L, d].
    per_token = sum(t.shape[1] * t.shape[3] for t in past)
    info = mla.get_kv_cache_size(L)
    assert info["mla"] == L * per_token, \
        f"get_kv_cache_size reports {info['mla']}, actual cache holds {L * per_token}"
    assert info["mla_absorbed"] < info["mla_expanded"], \
        "latent (absorbed) cache must be smaller than the expanded cache"
    return diff, per_token


def test_mla():
    from helioslm_v5.src.attention.mla import MLA
    config = HeliosLMv5Config(size="lite")

    diff_e, tok_e = _check_mla_mode(use_absorption=False)
    diff_a, tok_a = _check_mla_mode(use_absorption=True)
    assert tok_a < tok_e, "absorbed per-token cache must beat expanded"
    _pass("test_mla",
          f"expanded: max|diff|={diff_e:.2e} tok={tok_e}; "
          f"absorbed: max|diff|={diff_a:.2e} tok={tok_a} "
          f"(GQA/MHA baseline {2 * config.attention.num_key_value_heads * config.attention.head_dim})")


# ----------------------------------------------------------------------
# MLA weight absorption: absorbed == non-absorbed numerically (same weights)
# ----------------------------------------------------------------------
def test_mla_absorption():
    from helioslm_v5.src.attention.mla import MLA
    config = HeliosLMv5Config(size="lite")
    torch.manual_seed(115)
    mla = MLA(config).eval()  # built in one mode, then flipped in-place:
    # use_absorption only selects the cache layout / matmul grouping, all
    # weights are shared, so flipping the flag gives an identical model.
    assert config.attention.use_absorption  # lite default must be ON
    L = 8
    h = torch.randn(2, L, config.hidden_size)
    toks = [torch.randn(2, 1, config.hidden_size) for _ in range(3)]

    def run(mode):
        mla.use_absorption = mode
        with torch.no_grad():
            out, past = mla(h, use_cache=True)
            for tok in toks:  # 3 decode steps (same tokens for both modes)
                out_step, past = mla(tok, past_key_value=past, use_cache=True)
                out = torch.cat([out, out_step], dim=1)
        return out, past

    out_abs, past_abs = run(True)
    out_exp, past_exp = run(False)
    diff = (out_abs - out_exp).abs().max().item()
    assert diff < 1e-4, f"absorbed vs expanded diverge: {diff:.3e}"

    # MTP-rollback contract: truncating every cache tensor along dim 2 must
    # be legal and stay consistent across modes.
    keep = L + 1  # drop the last 2 decode steps
    trunc_a = tuple(t[:, :, :keep] for t in past_abs)
    trunc_e = tuple(t[:, :, :keep] for t in past_exp)
    tok = torch.randn(2, 1, config.hidden_size)
    with torch.no_grad():
        mla.use_absorption = True
        out_ta, past_ta = mla(tok, past_key_value=trunc_a, use_cache=True)
        mla.use_absorption = False
        out_te, _ = mla(tok, past_key_value=trunc_e, use_cache=True)
    tdiff = (out_ta - out_te).abs().max().item()
    assert tdiff < 1e-4, f"decode after dim-2 truncation diverges: {tdiff:.3e}"
    assert past_ta[0].shape[2] == keep + 1 and torch.isfinite(out_ta).all()

    # Cache-size numbers: absorbed must show a large saving vs MHA.
    mla.use_absorption = True
    info = mla.get_kv_cache_size(1024)
    assert info["mode"] == "absorbed"
    assert info["mla"] == info["mla_absorbed"]
    assert info["reduction_vs_mha_pct"] > 50.0, \
        f"absorbed saving vs MHA only {info['reduction_vs_mha_pct']:.1f}%"
    _pass("test_mla_absorption",
          f"absorbed-vs-expanded max|diff|={diff:.2e}, post-truncation "
          f"diff={tdiff:.2e}, cache saving vs MHA={info['reduction_vs_mha_pct']:.1f}%")


# ----------------------------------------------------------------------
# C1 regression (v5.3): absorbed mode + left-padding must not produce
# NaN (fully-masked pad query rows) nor pollute real-token outputs
# ----------------------------------------------------------------------
def test_mla_absorbed_left_padding():
    from helioslm_v5.src.attention.mla import MLA
    config = HeliosLMv5Config(size="lite")
    torch.manual_seed(130)
    mla = MLA(config).eval()
    L = 8
    h = torch.randn(2, L, config.hidden_size)
    # Row 0 is left-padded by 3 tokens; row 1 has no padding. The pad
    # query rows of row 0 are FULLY masked (causal + padding), which is
    # exactly the C1 NaN scenario in the absorbed softmax.
    mask = torch.ones(2, L)
    mask[0, :3] = 0

    nxt = torch.randn(2, 1, config.hidden_size)  # shared across both modes
    step_mask = torch.cat([mask, torch.ones(2, 1)], dim=1)

    def run(mode):
        mla.use_absorption = mode
        with torch.no_grad():
            out, past = mla(h, attention_mask=mask, use_cache=True)
            # One decode step on top; mask covers past + the new token.
            out_step, past = mla(nxt, attention_mask=step_mask,
                                 past_key_value=past, use_cache=True)
        return torch.cat([out, out_step], dim=1)

    out_abs = run(True)
    assert torch.isfinite(out_abs).all(), \
        "absorbed mode produced NaN/Inf with left-padding (C1 regression)"
    out_exp = run(False)
    assert torch.isfinite(out_exp).all(), \
        "expanded mode produced NaN/Inf with left-padding"

    # Real (non-pad) query rows must agree between the two modes; pad
    # rows are don't-care values but must be finite (checked above).
    real_rows = torch.cat([out_abs[0, 3:], out_abs[1]])
    ref_rows = torch.cat([out_exp[0, 3:], out_exp[1]])
    diff = (real_rows - ref_rows).abs().max().item()
    assert diff < 1e-4, \
        f"left-padded absorbed vs expanded diverge on real tokens: {diff:.3e}"
    _pass("test_mla_absorbed_left_padding",
          f"no NaN under left-padding, absorbed==expanded on real tokens "
          f"(max|diff|={diff:.2e})")


# ----------------------------------------------------------------------
# v5.4 (B1): packed position_ids segment attention per document —
# perturbing document A must not change document B's logits
# ----------------------------------------------------------------------
def test_packed_positions():
    model, config = _lite_model(seed=134)
    doc_a = [5, 100, 200]
    doc_b = [42, 900, 7]
    ids = torch.tensor([doc_a + doc_b])          # [1, 6], two packed docs
    pos = torch.tensor([[0, 1, 2, 0, 1, 2]])     # per-doc position restarts

    # Perturb EVERY token of document A (distinct replacement tokens).
    ids2 = ids.clone()
    ids2[0, :3] = torch.tensor([7, 42, 5])
    assert not torch.equal(ids2, ids)

    worst = 0.0
    for mode in (True, False):  # absorbed + expanded paths share the mask
        for layer in model.layers:
            layer.attention.use_absorption = mode
        with torch.no_grad():
            logits_a, _, _ = model(ids, position_ids=pos)
            logits_b, _, _ = model(ids2, position_ids=pos)
        assert logits_a.shape == (1, 6, config.vocab_size)
        assert torch.isfinite(logits_a).all()
        # No cross-document leakage through attention: the attention block
        # itself is bit-exact here (verified: masked softmax probabilities
        # are exactly 0 across the doc boundary). The only residual is
        # float-reassociation noise (~1e-7) from the MoE's expert-grouped
        # matmuls, whose batching changes when doc A tokens change — the
        # same noise floor as running a token solo vs batched. The sanity
        # check below shows real leakage is ~1e-3+, 4 orders larger.
        leak = (logits_a[0, 3:] - logits_b[0, 3:]).abs().max().item()
        worst = max(worst, leak)
        assert leak < 1e-5, \
            f"mode={'absorbed' if mode else 'expanded'}: doc A perturbation " \
            f"leaked into packed doc B logits ({leak:.3e})"

        # Packed doc B must equal doc B run alone with default positions
        # (same document, same per-doc positions => same output).
        with torch.no_grad():
            solo_b, _, _ = model(ids[:, 3:])
        solo_diff = (logits_a[0, 3:] - solo_b[0]).abs().max().item()
        assert solo_diff < 1e-5, \
            f"mode={'absorbed' if mode else 'expanded'}: packed doc B " \
            f"diverges from its solo forward ({solo_diff:.3e})"

    # Sanity (the test is not vacuous): with DEFAULT contiguous positions
    # the same perturbation DOES leak across the boundary.
    for layer in model.layers:
        layer.attention.use_absorption = True
    with torch.no_grad():
        plain_a, _, _ = model(ids)
        plain_b, _, _ = model(ids2)
    leaked = (plain_a[0, 3:] - plain_b[0, 3:]).abs().max().item()
    assert leaked > 1e-3, \
        f"sanity: expected cross-doc leakage under default positions, got {leaked:.3e}"
    _pass("test_packed_positions",
          f"packed [0,1,2,0,1,2]: doc B logits unchanged under doc A "
          f"perturbation (both cache modes, max {worst:.2e} = float-noise "
          f"floor), packed == solo doc forward, default-position leak "
          f"sanity={leaked:.2e}")



def test_sigmoid_moe():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
    config = HeliosLMv5Config(size="lite")
    torch.manual_seed(102)
    moe = DeviceLimitedMoE(config)
    x = torch.randn(2, 6, config.hidden_size)
    N = 2 * 6

    # 1) Selection uses bias; gating weights are bias-free.
    flat = x.reshape(-1, config.hidden_size)
    idx0, gates0 = moe._route(flat)
    with torch.no_grad():
        moe.route_bias.add_(torch.randn(config.moe.num_experts) * 0.05)
    idx1, gates1 = moe._route(flat)
    same = idx0 == idx1
    assert same.any(), "sanity: expected some unchanged selections"
    assert torch.equal(gates0[same], gates1[same]), \
        "gating weights changed when only the selection bias changed"

    # 2) A large bias forces selection of a targeted expert (bias affects selection).
    target = 3
    with torch.no_grad():
        moe.route_bias.zero_()
        moe.route_bias[target] = 100.0
    moe.expert_load.zero_()
    out = moe(x)
    assert out.shape == x.shape, f"output shape {out.shape} != input {x.shape}"
    assert moe.expert_load[target].item() == N, \
        f"biased expert selected {moe.expert_load[target].item()}/{N} times"

    # 3) update_bias: overloaded expert's bias decreases, underloaded increases,
    #    and the accumulated statistics are consumed.
    load = moe.expert_load.clone()
    before = moe.route_bias.clone()
    moe.update_bias()
    delta = moe.route_bias - before
    over, under = load > load.mean(), load < load.mean()
    assert load[target] > load.mean(), "sanity: target expert must be overloaded"
    assert (delta[over] < 0).all(), f"overloaded experts not penalized: {delta[over]}"
    assert (delta[under] > 0).all(), f"underloaded experts not boosted: {delta[under]}"
    assert moe.expert_load.sum().item() == 0, "update_bias must consume expert_load"

    # 4) Gradient flows to the router but NOT to route_bias (aux-free).
    moe.zero_grad(set_to_none=True)
    with torch.no_grad():
        moe.route_bias.zero_()
    moe(x).sum().backward()
    assert moe.router.weight.grad is not None \
        and moe.router.weight.grad.abs().sum() > 0, "router received no gradient"
    assert moe.route_bias.grad is None, "route_bias must not be gradient-trained"
    assert not moe.route_bias.requires_grad
    _pass("test_sigmoid_moe",
          f"bias selects-only, update_bias sign correct, router grad "
          f"norm={moe.router.weight.grad.norm():.3e}, route_bias.grad=None")


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
# PagedAttention: fp32 write/read across blocks, CoW fork, no leaks
# ----------------------------------------------------------------------
def test_paged_attention():
    from helioslm_v5.src.inference.paged_attention import (
        BlockManager, PagedAttention)

    config = HeliosLMv5Config(size="lite")
    H = config.attention.num_attention_heads
    D = config.hidden_size // H  # PagedAttention head_dim = hidden / heads

    # 1) Cross-block write -> read back value-for-value.
    bm = BlockManager(block_size=4, num_blocks=16, device="cpu",
                      dtype=torch.float32)
    bm.allocate(0, 3, H, D)
    bm.append_tokens(0, 5)  # total 8 tokens = 2 full blocks (crosses boundary)
    torch.manual_seed(105)
    ks, vs = torch.randn(8, H, D), torch.randn(8, H, D)
    for i in range(8):
        bm.write_kv(0, i, ks[i], vs[i])
    k_g, v_g = bm.gather_kv(0)
    assert torch.equal(k_g, ks) and torch.equal(v_g, vs), \
        "paged K/V read-back mismatch across block boundary"

    # 2) Copy-on-write fork: writing the child must not touch the parent.
    bm.fork(0, 1)
    parent_block = bm.get_block_table(0)[0]
    assert bm.refcounts[parent_block] == 2, "fork must share blocks (refcount 2)"
    new_k, new_v = torch.randn(H, D), torch.randn(H, D)
    bm.write_kv(1, 1, new_k, new_v)  # write inside the shared block
    assert bm.get_block_table(0)[0] == parent_block, "parent block table changed"
    assert bm.get_block_table(1)[0] != parent_block, "CoW did not copy the block"
    assert bm.refcounts[parent_block] == 1, "parent refcount wrong after CoW"
    k_p, _ = bm.gather_kv(0)
    k_c, _ = bm.gather_kv(1)
    assert torch.equal(k_p[1], ks[1]), "parent data corrupted by child write"
    assert torch.equal(k_c[1], new_k) and torch.equal(k_c[0], ks[0]), \
        "child did not see its own write / lost shared prefix"

    # 3) Freeing both sequences returns every block, no leaks.
    bm.free(0)
    bm.free(1)
    assert bm.num_free_blocks() == 16 and not bm.block_tables and not bm.refcounts, \
        "block leak after free"

    # 4) PagedAttention module: fp32 end-to-end, cache dtype follows input,
    #    and output matches a manual causal-attention reference.
    torch.manual_seed(106)
    pa = PagedAttention(config).eval()
    bm2 = BlockManager(block_size=4, num_blocks=16, device="cpu")  # dtype=None
    bm2.allocate(0, 0, pa.num_heads, pa.head_dim)
    h = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        out = pa(h, bm2, [0])
    assert bm2.k_cache is not None and bm2.k_cache.dtype == torch.float32, \
        "cache dtype must follow fp32 input"
    assert out.shape == h.shape and torch.isfinite(out).all()

    # Manual reference: single-shot causal attention over the same projections.
    with torch.no_grad():
        q = pa.q_proj(h).view(1, 5, H, D).transpose(1, 2)
        k = pa.k_proj(h).view(1, 5, H, D).transpose(1, 2)
        v = pa.v_proj(h).view(1, 5, H, D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        ref = pa.o_proj((torch.softmax(scores, dim=-1) @ v)
                        .transpose(1, 2).reshape(1, 5, H * D))
    ref_diff = (out - ref).abs().max().item()
    assert ref_diff < 1e-5, f"PagedAttention vs manual causal ref: {ref_diff:.3e}"
    _pass("test_paged_attention",
          f"cross-block readback exact, CoW isolated, no leaks, "
          f"vs manual ref max|diff|={ref_diff:.2e}")


# ----------------------------------------------------------------------
# VLLMEngine: continuous batching == per-request greedy; no block leaks
# ----------------------------------------------------------------------
def test_vllm_engine():
    from helioslm_v5.src.inference.vllm_engine import VLLMEngine

    model, config = _lite_model(seed=107)
    prompts = [[5, 100, 200, 7], [42, 900]]  # unequal lengths
    max_new = 6
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    req_ids = [engine.add_request(p, max_new_tokens=max_new, temperature=0.0)
               for p in prompts]
    results = engine.run()

    for p, rid in zip(prompts, req_ids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=max_new,
                             temperature=0)
        ref_new = ref[0, len(p):].tolist()
        got = results[rid]
        # The engine stops at EOS while generate() freezes with pad; the
        # engine output must be an exact prefix of the greedy reference.
        assert ref_new[:len(got)] == got, \
            f"request {rid}: engine {got} != greedy reference {ref_new[:len(got)]}"
        assert len(got) >= 1

    bm = engine.block_manager
    assert len(bm.block_tables) == 0 and len(bm.refcounts) == 0 \
        and bm.num_free_blocks() == 64, \
        f"block leak: free={bm.num_free_blocks()}/64 tables={len(bm.block_tables)}"
    _pass("test_vllm_engine",
          "2 unequal-length requests match per-request greedy; blocks 64/64 free")


# ----------------------------------------------------------------------
# DualPipe: run_dual gradients == direct forward+backward (mean over MBs)
# ----------------------------------------------------------------------
def test_dualpipe():
    from helioslm_v5.src.training.dualpipe import DualPipeScheduler, DualPipeStage

    model, config = _lite_model(seed=108)

    class LayerWrap(nn.Module):
        """Adapt HeliosLMv5Layer ((hidden, kv) output) to tensor->tensor."""
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            h, _ = self.layer(x, use_cache=False)
            return h

    stage = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0]),
                                         LayerWrap(model.layers[1])]))
    torch.manual_seed(109)
    inputs = [torch.randn(2, 5, config.hidden_size) for _ in range(4)]
    loss_fn = lambda out: out.pow(2).mean()
    M = len(inputs)

    sched = DualPipeScheduler([stage], num_micro_batches=M)
    loss_pipe = sched.run_dual(inputs, loss_fn)
    grads_pipe = {n: p.grad.clone()
                  for n, p in model.named_parameters() if p.grad is not None}
    assert grads_pipe, "run_dual produced no parameter gradients"

    # Reference: direct forward+backward with the same 1/M scaling.
    model.zero_grad(set_to_none=True)
    loss_ref = 0.0
    for x in inputs:
        loss = loss_fn(stage(x)) / M
        loss.backward()
        loss_ref += loss.item()

    assert math.isfinite(loss_pipe) and math.isfinite(loss_ref)
    assert abs(loss_pipe - loss_ref) < 1e-5, \
        f"loss mismatch: pipe={loss_pipe:.6f} ref={loss_ref:.6f}"
    max_diff = 0.0
    for n, p in model.named_parameters():
        if n in grads_pipe:
            assert p.grad is not None, f"{n}: grad missing in reference"
            d = (grads_pipe[n] - p.grad).abs().max().item()
            max_diff = max(max_diff, d)
            assert torch.allclose(grads_pipe[n], p.grad, atol=1e-5, rtol=1e-4), \
                f"{n}: pipe grad != direct grad (max diff {d:.3e})"

    # All attention parameters of both layers must have received a gradient
    # (recompute backward reaches every stage parameter on the path).
    attn_params = [p for layer in (model.layers[0], model.layers[1])
                   for p in layer.attention.parameters()]
    assert all(p.grad is not None for p in attn_params)
    _pass("test_dualpipe",
          f"loss={loss_pipe:.5f} (ref {loss_ref:.5f}), "
          f"grads match (max diff {max_diff:.2e}, atol 1e-5)")


# ----------------------------------------------------------------------
# FP8Trainer: finite losses, no grad accumulation, real quantization
# ----------------------------------------------------------------------
def test_fp8_trainer():
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer, FP8Linear

    model, config = _lite_model(seed=110)
    trainer = FP8Trainer(model, config)

    # Quantization is real (not an identity pass-through).
    fp8_layer = next(m for m in model.modules() if isinstance(m, FP8Linear))
    w = fp8_layer.weight.detach()
    wq = fp8_layer._quantize_to_fp8(w, fp8_layer.weight_scale.clamp(min=1e-8))
    q_diff = (w - wq).abs().max().item()
    q_rel = ((w - wq).norm() / w.norm().clamp(min=1e-12)).item()
    assert q_diff > 0, "FP8 quantization is an identity — not real quantization"
    assert q_rel < 0.2, f"FP8 quantization error implausibly large: {q_rel:.3f}"

    ids = torch.randint(3, config.vocab_size, (2, 10))
    batch = {"input_ids": ids}
    probe = model.embed_tokens.weight  # not a Linear -> survives FP8 conversion
    losses, grad_norms = [], []
    for _ in range(3):
        losses.append(trainer.train_step(batch))
        grad_norms.append(probe.grad.norm().item())

    assert all(math.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    assert grad_norms[0] > 0, "no gradient reached embed_tokens"
    # Same batch each step: with per-step zero_grad the grad norm stays ~flat
    # (weights drift only slightly); accumulation would roughly triple it.
    ratio = grad_norms[-1] / max(grad_norms[0], 1e-12)
    assert ratio < 1.10, \
        f"grad norm ratio {ratio:.2f} over 3 identical steps suggests accumulation"
    _pass("test_fp8_trainer",
          f"losses={[f'{l:.4f}' for l in losses]}, "
          f"grad-norm ratio={ratio:.3f}, fp8 rel-err={q_rel:.4f}")


# ----------------------------------------------------------------------
# GRPO: one real train_step; reward/answer alignment; non-negative KL
# ----------------------------------------------------------------------
def test_grpo():
    from helioslm_v5.src.training.grpo import GRPOTrainer

    model, config = _lite_model(seed=111)
    ref_model, _ = _lite_model(seed=211)  # different init -> non-trivial KL
    G = 3
    config.grpo.group_size = G
    config.grpo.max_new_tokens = 4
    trainer = GRPOTrainer(model, ref_model, config)

    # Spy reward function: records the answers it receives and returns
    # varied deterministic rewards (non-zero advantage variance per group).
    seen = {}

    def spy_reward(responses, answers):
        seen["answers"] = list(answers)
        seen["n_responses"] = len(responses)
        return torch.tensor([0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
                            dtype=torch.float32, device=trainer.device)

    trainer.compute_rewards = spy_reward
    out = trainer.train_step(["q1?", "q2?"], ["a1", "a2"])

    # batch=2, G=3 -> answers must be repeat-interleaved per question.
    assert seen["answers"] == ["a1"] * G + ["a2"] * G, \
        f"reward answers misaligned: {seen['answers']}"
    assert seen["n_responses"] == 2 * G

    assert isinstance(out["loss"], float) and math.isfinite(out["loss"])
    assert out["kl_penalty"] >= -1e-6, \
        f"k3 KL estimator must be non-negative, got {out['kl_penalty']}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), \
        "no (finite) parameter gradients after train_step"
    assert model.embed_tokens.weight.grad is not None, \
        "embed_tokens must receive gradient via the grad-carrying logprob forward"
    _pass("test_grpo",
          f"loss={out['loss']:.4f} policy={out['policy_loss']:.4f} "
          f"kl={out['kl_penalty']:.6f}, answers aligned, "
          f"params_with_grad={len(grads)}")


# ----------------------------------------------------------------------
# Streaming audio: chunked == one-shot; causality; process_stream chunking
# ----------------------------------------------------------------------
def test_streaming_audio():
    from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder

    config = HeliosLMv5Config(size="lite")
    config.multimodal.audio_hidden_size = 128
    config.multimodal.audio_num_layers = 2
    torch.manual_seed(112)
    enc = StreamingAudioEncoder(config).eval()
    n_mels = config.multimodal.audio_n_mels
    mel = torch.randn(1, n_mels, 60)

    with torch.no_grad():
        enc.reset_state()
        full = enc(mel)
        enc.reset_state()
        chunks = [enc(mel[:, :, i:i + 20]) for i in range(0, 60, 20)]
        streamed = torch.cat(chunks, dim=1)
    diff = (full - streamed).abs().max().item()
    assert full.shape == streamed.shape == (1, 60, 128)
    assert diff < 1e-3, f"streaming != one-shot: max|diff|={diff:.3e}"

    # Causality: changing frames >= 40 must not change outputs < 40.
    mel2 = mel.clone()
    mel2[:, :, 40:] = torch.randn(1, n_mels, 20)
    with torch.no_grad():
        enc.reset_state()
        full2 = enc(mel2)
    assert torch.allclose(full[:, :40], full2[:, :40], atol=1e-6), \
        "future audio frames leaked into past outputs"

    # process_stream: 500 ms chunks at 16 kHz / hop 160 -> 50 frames per chunk.
    mel3 = torch.randn(1, n_mels, 120)
    with torch.no_grad():
        outs = list(enc.process_stream(mel3, chunk_ms=500))
    assert [o.shape[1] for o in outs] == [50, 50, 20], \
        f"chunk frame counts {[o.shape[1] for o in outs]} != [50, 50, 20]"
    _pass("test_streaming_audio",
          f"3-chunk max|diff|={diff:.2e}, causal, 500ms->50 frames/chunk")


# ----------------------------------------------------------------------
# NaViT: 224x224 works, over-grid raises, forward_packed mixed sizes
# ----------------------------------------------------------------------
def test_navit():
    from helioslm_v5.src.vision.navit import NaViTEncoder

    config = HeliosLMv5Config(size="lite")
    config.multimodal.vision_hidden_size = 128
    config.multimodal.vision_num_layers = 2
    config.multimodal.vision_num_heads = 4
    torch.manual_seed(113)
    enc = NaViTEncoder(config).eval()
    assert enc.max_grid == 32  # lite default
    # Factorized 2D position embedding: separate row/col tables.
    assert enc.row_embed.weight.shape == (32, 128)
    assert enc.col_embed.weight.shape == (32, 128)

    img = torch.randn(1, 3, 224, 224)  # 16x16 = 256 patches, within grid
    with torch.no_grad():
        out = enc(img)
    assert out.shape == (1, 256, 128) and torch.isfinite(out).all()

    # 512x512 -> 36x36 grid > max_grid=32 -> ValueError.
    big = torch.randn(1, 3, 512, 512)
    _expect_raises(ValueError, lambda: enc(big), "over-grid image")

    # forward_packed: mixed sizes padded to the longest sequence with a
    # boolean mask; the un-padded row must match a solo forward.
    imgs = [torch.randn(3, 224, 224), torch.randn(3, 112, 112)]  # 256 / 64 patches
    with torch.no_grad():
        feats, mask = enc.forward_packed(imgs)
    assert feats.shape == (2, 256, 128)
    assert mask.dtype == torch.bool
    assert int(mask[0].sum()) == 256 and bool(mask[0].all())
    assert int(mask[1].sum()) == 64 and not bool(mask[1, 64:].any())
    with torch.no_grad():
        solo = enc(imgs[0].unsqueeze(0))
    pack_diff = (feats[0][mask[0]] - solo[0]).abs().max().item()
    assert pack_diff < 1e-4, \
        f"padded batching changed unpadded row outputs: {pack_diff:.3e}"
    _pass("test_navit",
          f"224px OK (256 patches), 512px@grid32 raises, packed mask exact, "
          f"packed-vs-solo max|diff|={pack_diff:.2e}")


# ----------------------------------------------------------------------
# v5.4 (LOW): NaViT rejects degenerate inputs with a clear ValueError
# ----------------------------------------------------------------------
def test_navit_input_validation():
    from helioslm_v5.src.vision.navit import NaViTEncoder

    config = HeliosLMv5Config(size="lite")
    config.multimodal.vision_hidden_size = 128
    config.multimodal.vision_num_layers = 2
    config.multimodal.vision_num_heads = 4
    torch.manual_seed(135)
    enc = NaViTEncoder(config).eval()
    p = enc.patch_size  # 14

    # Image smaller than the patch size produces zero patches; this must
    # raise a clear ValueError (not a cryptic Conv2d shape error).
    _expect_raises(ValueError, lambda: enc(torch.randn(1, 3, p - 1, 224)),
                   "forward with H < patch_size")
    _expect_raises(ValueError, lambda: enc(torch.randn(1, 3, 224, p - 1)),
                   "forward with W < patch_size")

    # forward_packed: empty list and sub-patch member images raise ValueError
    # (previously: IndexError on all_patches[0] / cryptic Conv2d error).
    _expect_raises(ValueError, lambda: enc.forward_packed([]),
                   "forward_packed([])")
    _expect_raises(ValueError,
                   lambda: enc.forward_packed(
                       [torch.randn(3, 224, 224), torch.randn(3, p - 1, 224)]),
                   "forward_packed with a sub-patch image")
    _pass("test_navit_input_validation",
          "small image raises in forward/forward_packed, empty list raises")


# ----------------------------------------------------------------------
# Quantization: AWQ/GPTQ/FP8 + QuantizationManager dispatch
# ----------------------------------------------------------------------
def test_quantization():
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, FP8Linear, QuantizationManager)

    torch.manual_seed(114)
    lin = nn.Linear(64, 48, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(48, 64) * 0.05)
        lin.bias.copy_(torch.randn(48) * 0.01)
    W_ref = lin.weight.detach()
    x = torch.randn(7, 64)

    # AWQ: 4-bit reconstruction error near the theoretical RTN floor (<10%).
    awq = AWQLinear.from_linear(lin, group_size=32)
    rel_awq = ((awq._dequantize() - W_ref).norm() / W_ref.norm()).item()
    assert rel_awq < 0.10, f"AWQ relative weight error {rel_awq:.3%} >= 10%"
    assert awq.bias is not None and torch.equal(awq.bias, lin.bias), \
        "AWQ must preserve the bias"
    out_awq = awq(x)
    rel_out = ((out_awq - lin(x)).norm() / lin(x).norm().clamp(min=1e-12)).item()
    assert out_awq.shape == (7, 48) and rel_out < 0.15

    # GPTQ without calibration data takes the documented RTN fallback
    # (the true Hessian-compensated path is covered by
    # test_gptq_calibration below): deterministic forward (weights rebuilt
    # from packed buffers), bias preserved, reconstruction error < 10%.
    gptq = GPTQLinear.from_linear(lin, group_size=32)  # no calibration -> RTN
    o1, o2 = gptq(x), gptq(x)
    assert torch.equal(o1, o2), "GPTQ forward is not deterministic"
    rel_gptq = ((gptq._dequantize() - W_ref).norm() / W_ref.norm()).item()
    assert rel_gptq < 0.10, f"GPTQ relative weight error {rel_gptq:.3%} >= 10%"
    assert gptq.bias is not None and torch.equal(gptq.bias, lin.bias)

    # M-Q1 regression (v5.3): fp16/bf16 inputs must not crash — the bias
    # is cast to the input dtype, like AWQ/FP8 already did.
    gptq_c = GPTQLinear.from_linear(lin, group_size=32,
                                    calibration_data=torch.randn(128, 64))
    for dt in (torch.float16, torch.bfloat16):
        for m, tag in ((gptq, "rtn"), (gptq_c, "gptq")):
            out_dt = m(x.to(dt))
            assert out_dt.dtype == dt and torch.isfinite(out_dt).all(), \
                f"GPTQ({tag}) forward broke for {dt}"

    # QuantizationManager: dispatch + unknown method.
    tiny = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 16))
    QuantizationManager(method="awq").quantize_model(tiny, group_size=16)
    assert isinstance(tiny[0], AWQLinear) and isinstance(tiny[2], AWQLinear)
    y = tiny(torch.randn(3, 32))
    assert torch.isfinite(y).all()
    _expect_raises(ValueError,
                   lambda: QuantizationManager(method="mx4").quantize_model(tiny),
                   "unknown quantization method")

    # FP8: real float8 storage when the torch build supports it, otherwise a
    # documented NotImplementedError.
    if FP8Linear is not None:
        f8 = FP8Linear.from_linear(lin)
        assert f8.weight_fp8.dtype == torch.float8_e4m3fn, \
            f"FP8 weights must be stored as float8_e4m3fn, got {f8.weight_fp8.dtype}"
        out_f8 = f8(x)
        assert out_f8.shape == (7, 48) and torch.isfinite(out_f8).all()
        fp8_note = "native float8_e4m3fn storage, forward OK"
    else:
        _expect_raises(NotImplementedError,
                       lambda: QuantizationManager(method="fp8").quantize_model(tiny),
                       "fp8 without float8 support")
        fp8_note = "no float8 in this torch -> NotImplementedError (as documented)"

    _pass("test_quantization",
          f"AWQ rel-err={rel_awq:.2%}, GPTQ rel-err={rel_gptq:.2%} & "
          f"deterministic, bias preserved, unknown method raises, fp8: {fp8_note}")


# ----------------------------------------------------------------------
# v5.2: true GPTQ Hessian error compensation beats RTN on layer OUTPUT
# ----------------------------------------------------------------------
def test_gptq_calibration():
    from helioslm_v5.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(120)
    lin = nn.Linear(256, 256, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(256, 256) * 0.05)
        lin.bias.copy_(torch.randn(256) * 0.01)

    # Correlated calibration activations (low-rank + small noise): the
    # Hessian H = X^T X is strongly anisotropic, which is exactly where
    # GPTQ's OBS error compensation wins over plain RTN.
    rank = 48
    basis = torch.randn(rank, 256)

    def make_data(n, seed):
        g = torch.Generator().manual_seed(seed)
        coeff = torch.randn(n, rank, generator=g)
        return coeff @ basis + 0.05 * torch.randn(n, 256, generator=g)

    calib = make_data(512, seed=700)

    rtn = GPTQLinear.from_linear(lin, group_size=128, bits=4)  # no calibration
    gptq = GPTQLinear.from_linear(lin, group_size=128, bits=4,
                                  calibration_data=calib)

    # Held-out data from the same distribution: GPTQ must reconstruct the
    # layer OUTPUT significantly better than RTN (measured ratio ~0.33;
    # threshold 0.80 leaves ample margin).
    eval_x = make_data(1024, seed=701)
    ref = lin(eval_x)
    err_rtn = ((rtn(eval_x) - ref).norm() / ref.norm()).item()
    err_gptq = ((gptq(eval_x) - ref).norm() / ref.norm()).item()
    assert err_gptq < 0.8 * err_rtn, \
        f"GPTQ output rel-err {err_gptq:.4f} not < 80% of RTN {err_rtn:.4f}"

    # Deterministic forward in both modes (weights rebuilt from packed
    # buffers every call).
    assert torch.equal(gptq(eval_x), gptq(eval_x)), "GPTQ forward not deterministic"
    assert torch.equal(rtn(eval_x), rtn(eval_x)), "RTN forward not deterministic"

    # Calibration data whose last dim != in_features raises ValueError.
    _expect_raises(ValueError,
                   lambda: GPTQLinear.from_linear(
                       lin, group_size=128,
                       calibration_data=torch.randn(64, 128)),
                   "calibration_data with wrong last dim")
    _pass("test_gptq_calibration",
          f"output rel-err GPTQ {err_gptq:.2%} vs RTN {err_rtn:.2%} "
          f"(ratio {err_gptq / err_rtn:.2f} < 0.80), deterministic, "
          f"bad calibration shape raises")


# ----------------------------------------------------------------------
# v5.3 (M-Q3): AWQ calibration path is never significantly worse than
# plain RTN, including under peaky/outlier activations
# ----------------------------------------------------------------------
def test_awq_calibration():
    from helioslm_v5.src.quantization.standard_quant import AWQLinear

    g = torch.Generator().manual_seed(42)
    lin = nn.Linear(128, 96, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(96, 128, generator=g) * 0.05)
        lin.bias.copy_(torch.randn(96, generator=g) * 0.01)

    def make(kind, n, gen):
        if kind == "gauss":
            return torch.randn(n, 128, generator=gen)
        if kind == "lognorm2":
            return (torch.exp(2.0 * torch.randn(n, 128, generator=gen))
                    * torch.where(torch.rand(n, 128, generator=gen) > 0.5,
                                  1., -1.))
        if kind == "peaky":  # a few outlier channels with ~100x magnitude
            xx = torch.randn(n, 128, generator=gen)
            idx = torch.randperm(128, generator=gen)[:5]
            xx[:, idx] *= 100.0
            return xx
        raise AssertionError(kind)

    def rel_err(mod, xx):
        ref = lin(xx)
        return ((mod(xx) - ref).norm() / ref.norm().clamp(min=1e-12)).item()

    detail = []
    for kind in ("gauss", "lognorm2", "peaky"):
        calib = make(kind, 512, torch.Generator().manual_seed(42))
        eval_x = make(kind, 1024, torch.Generator().manual_seed(43))
        rtn = AWQLinear.from_linear(lin, group_size=32)
        cal = AWQLinear.from_linear(lin, group_size=32, activations=calib)
        assert cal.act_scale is not None, "calibrated AWQ must store act_scale"
        e_rtn, e_cal = rel_err(rtn, eval_x), rel_err(cal, eval_x)
        detail.append(f"{kind} {e_cal:.2%}/RTN {e_rtn:.2%}")
        assert e_cal <= 1.3 * e_rtn + 1e-3, \
            f"AWQ calibrated path on {kind}: output rel-err {e_cal:.4f} " \
            f"significantly worse than RTN {e_rtn:.4f}"
    _pass("test_awq_calibration", "; ".join(detail) + " (all <= 1.3x RTN)")


# ----------------------------------------------------------------------
# v5.3: quantized .weight property == forward's effective weight matrix
# ----------------------------------------------------------------------
def test_quant_weight_property():
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, FP8Linear)

    torch.manual_seed(131)
    lin = nn.Linear(48, 40, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(40, 48) * 0.05)
        lin.bias.copy_(torch.randn(40) * 0.01)
    x = torch.randn(6, 48)
    calib = torch.randn(256, 48)

    mods = [
        ("awq-rtn", AWQLinear.from_linear(lin, group_size=16)),
        ("awq-calib", AWQLinear.from_linear(lin, group_size=16,
                                            activations=calib)),
        ("gptq-rtn", GPTQLinear.from_linear(lin, group_size=16)),
        ("gptq-calib", GPTQLinear.from_linear(lin, group_size=16,
                                              calibration_data=calib)),
    ]
    if FP8Linear is not None:
        mods.append(("fp8", FP8Linear.from_linear(lin)))

    for tag, m in mods:
        w = m.weight
        assert w.shape == (lin.out_features, lin.in_features), \
            f"{tag}: .weight shape {w.shape}"
        assert torch.equal(m.weight, m.weight), \
            f"{tag}: .weight is not deterministic across accesses"
        ref = nn.functional.linear(x, w.to(x.dtype), m.bias.to(x.dtype))
        d = (m(x) - ref).abs().max().item()
        assert d < 1e-5, \
            f"{tag}: forward disagrees with .weight property ({d:.3e})"
    _pass("test_quant_weight_property",
          f"{len(mods)} module variants: .weight == forward effective weight")


# ----------------------------------------------------------------------
# v5.4 (LOW): quantization input validation fails loudly
# ----------------------------------------------------------------------
def test_quant_input_validation():
    import warnings as _warnings
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, QuantizationManager)

    torch.manual_seed(136)
    lin = nn.Linear(16, 8)

    # AWQ calibration activations whose last dim != in_features must raise
    # ValueError (aligned with the GPTQ path) instead of silently
    # reshape-reinterpreting the data.
    _expect_raises(ValueError,
                   lambda: AWQLinear.from_linear(
                       lin, activations=torch.randn(10, 32)),
                   "AWQ activations with wrong last dim")
    # bits != 4 raises ValueError (no bare `assert`) on both packers.
    _expect_raises(ValueError,
                   lambda: AWQLinear.from_linear(lin, bits=8),
                   "AWQ bits=8")
    _expect_raises(ValueError,
                   lambda: GPTQLinear.from_linear(lin, bits=8),
                   "GPTQ bits=8")

    # quantize_model on a bare nn.Linear cannot replace it in place; it must
    # warn and leave the layer unquantized (previously a silent skip).
    bare = nn.Linear(16, 8)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        QuantizationManager(method="awq").quantize_model(bare, group_size=8)
    assert isinstance(bare, nn.Linear), \
        "bare nn.Linear should be left unquantized"
    assert any("bare nn.Linear" in str(w.message) for w in caught), \
        f"expected a bare-nn.Linear skip warning, got {[str(w.message) for w in caught]}"
    _pass("test_quant_input_validation",
          "AWQ bad calibration shape raises, bits!=4 raises (AWQ+GPTQ), "
          "bare nn.Linear skip warns")


# ----------------------------------------------------------------------
# v5.4: GPTQ odd in/out dims pack/unpack exactly (forward == .weight)
# ----------------------------------------------------------------------
def test_gptq_odd_dims():
    from helioslm_v5.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(137)
    # Odd in AND out features stress the pad-to-8 packing of qweight (rows)
    # and qzeros (cols).
    lin = nn.Linear(35, 27, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(27, 35) * 0.05)
        lin.bias.copy_(torch.randn(27) * 0.01)
    x = torch.randn(5, 35)

    for tag, mod in (
        ("rtn", GPTQLinear.from_linear(lin, group_size=16)),
        ("gptq", GPTQLinear.from_linear(lin, group_size=16,
                                        calibration_data=torch.randn(128, 35))),
    ):
        w = mod.weight
        assert w.shape == (27, 35), f"{tag}: .weight shape {w.shape}"
        out = mod(x)
        assert out.shape == (5, 27) and torch.isfinite(out).all()
        ref = nn.functional.linear(x, w.to(x.dtype), mod.bias.to(x.dtype))
        d = (out - ref).abs().max().item()
        assert d < 1e-5, \
            f"{tag}: forward disagrees with .weight on odd dims ({d:.3e})"
        rel = ((w - lin.weight).norm() / lin.weight.norm()).item()
        assert rel < 0.15, f"{tag}: odd-dim reconstruction error {rel:.3%}"
    _pass("test_gptq_odd_dims",
          "35x27 (odd in+out): forward == .weight property, RTN + calibrated")


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


# ----------------------------------------------------------------------
# v5.2: streaming audio sliding-window memory cap
# ----------------------------------------------------------------------
def test_audio_sliding_window():
    from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder

    config = HeliosLMv5Config(size="lite")
    config.multimodal.audio_hidden_size = 128
    config.multimodal.audio_num_layers = 2
    config.multimodal.audio_max_memory_frames = 90
    torch.manual_seed(122)
    enc = StreamingAudioEncoder(config).eval()
    assert enc.max_memory_frames == 90
    n_mels = config.multimodal.audio_n_mels
    chunk = 20

    # 1) Within the window (4 chunks x 20 = 80 frames <= 90): streaming is
    #    identical to a one-shot forward and memory is NOT truncated.
    mel = torch.randn(1, n_mels, 4 * chunk)
    with torch.no_grad():
        enc.reset_state()
        full = enc(mel)
        enc.reset_state()
        streamed = torch.cat(
            [enc(mel[:, :, i:i + chunk]) for i in range(0, 80, chunk)], dim=1)
        in_window_lens = [m.shape[1] for m in enc._memory]
    diff = (full - streamed).abs().max().item()
    assert diff < 1e-4, f"in-window streaming != one-shot: max|diff|={diff:.3e}"
    assert all(l == 80 for l in in_window_lens), \
        f"in-window memory must be uncapped (80), got {in_window_lens}"

    # 2) Beyond the window: 10 chunks x 20 = 200 frames; every layer's
    #    memory must stay capped at 90 (oldest frames dropped).
    mel_long = torch.randn(1, n_mels, 10 * chunk)
    with torch.no_grad():
        enc.reset_state()
        for i in range(0, 200, chunk):
            out = enc(mel_long[:, :, i:i + chunk])
            lens = [m.shape[1] for m in enc._memory]
            assert all(l <= 90 for l in lens), \
                f"memory exceeded window at frame {i + chunk}: {lens}"
        assert out.shape == (1, chunk, 128) and torch.isfinite(out).all()
        assert all(m.shape[1] == 90 for m in enc._memory), \
            f"memory should be pinned at the window: " \
            f"{[m.shape[1] for m in enc._memory]}"

    # 3) reset_state restarts cleanly: a fresh forward reproduces the
    #    pre-overflow one-shot output.
    enc.reset_state()
    assert enc._memory is None, "reset_state must clear the memory prefix"
    with torch.no_grad():
        again = enc(mel)
    restart_diff = (again - full).abs().max().item()
    assert restart_diff < 1e-4, \
        f"post-reset forward diverged: {restart_diff:.3e}"
    _pass("test_audio_sliding_window",
          f"memory capped at 90 over 200 frames, in-window "
          f"stream-vs-oneshot max|diff|={diff:.2e}, reset_state OK")


TESTS = [
    test_mla,
    test_mla_absorption,
    test_mla_absorbed_left_padding,
    test_packed_positions,
    test_sigmoid_moe,
    test_v5_model,
    test_mtp,
    test_paged_attention,
    test_vllm_engine,
    test_dualpipe,
    test_fp8_trainer,
    test_grpo,
    test_streaming_audio,
    test_navit,
    test_navit_input_validation,
    test_quantization,
    test_gptq_calibration,
    test_awq_calibration,
    test_quant_weight_property,
    test_quant_input_validation,
    test_gptq_odd_dims,
    test_mtp_rebind_after_quantization,
    test_audio_sliding_window,
]


def main():
    print("HeliosLM v5.4 Test Suite (lite config, CPU)")
    print("=" * 72)
    for t in TESTS:
        try:
            t()
        except Exception:
            _fail(t.__name__)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("=" * 72)
    print(f"SUMMARY: {n_ok}/{len(RESULTS)} PASS")
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_ok == len(RESULTS) == len(TESTS) else 1)


if __name__ == "__main__":
    main()
