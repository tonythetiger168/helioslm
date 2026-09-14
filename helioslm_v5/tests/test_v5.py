"""HeliosLM v5.8 test suite (CPU, lite config).

Rewritten for v5.1 after the full code review and extended since (v5.2: true
GPTQ calibration, audio sliding window; v5.3: AWQ calibration repair;
v5.4: packed-position cross-document isolation, input-validation coverage;
v5.5: hybrid GatedDeltaAttention + state-cache rollback, LatentMoE,
quantile balancing, attention residuals, SiTU-GLU; v5.7: RoPE scaling,
FP8 KV cache, Hyper-Connections, QAT; v5.8: YaRN RoPE scaling, DSA sparse
top-k attention, per-head Muon, GPTQ act-order):
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
    from helioslm_v5.src.training.dualpipe import (
        DualPipeScheduler, DualPipeStage, LayerWrap)

    model, config = _lite_model(seed=108)

    # LayerWrap (dualpipe) adapts HeliosLMv5Layer ((hidden, kv, attn_res)
    # output) to the tensor->tensor stage contract; with attention
    # residuals off (lite default) it threads nothing (v5.4 behaviour).
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
# v5.6: Muon optimizer — Newton-Schulz orthogonalization quality, quadratic
# convergence vs SGD, AdamW fallback for non-matrix params, muon=False
# group routing
# ----------------------------------------------------------------------
def test_muon():
    from helioslm_v5.src.training.muon import (
        Muon, zeropower_via_newtonschulz5)

    torch.manual_seed(601)
    # 1) Orthogonalization quality: 5-step NS pushes every singular value
    #    into the documented [~0.3, ~1.35] band around 1 (approximate
    #    orthogonalization — the standard Muon recipe; exactness is NOT
    #    expected at 5 steps) with mean singular value ~= 1 (energy kept).
    A = torch.randn(24, 24)
    O = zeropower_via_newtonschulz5(A)
    sv = torch.linalg.svdvals(O)
    assert 0.25 < sv.min().item() and sv.max().item() < 1.4, \
        f"NS singular values out of band: [{sv.min():.3f}, {sv.max():.3f}]"
    mean_sv = sv.mean().item()
    assert 0.8 < mean_sv < 1.2, f"NS mean singular value {mean_sv:.3f} != 1"
    eye_err = (O @ O.T - torch.eye(24)).abs().max().item()
    assert eye_err < 0.6, f"NS far from orthogonal: |OO^T - I| max {eye_err:.3e}"
    # Condition number is dramatically improved vs the raw matrix.
    assert sv.max() / sv.min() < torch.linalg.svdvals(A).max() / \
        torch.linalg.svdvals(A).min(), "NS did not improve conditioning"
    # Tall/wide matrices are handled via the transposed iteration.
    for shape in [(32, 12), (12, 32)]:
        B = torch.randn(*shape)
        OB = zeropower_via_newtonschulz5(B)
        gram = OB @ OB.T if shape[0] <= shape[1] else OB.T @ OB
        e = (gram - torch.eye(min(shape))).abs().max().item()
        assert e < 0.6, f"NS({shape}) gram error {e:.3e}"

    # 2) Convergence: Muon vs plain SGD on a least-squares problem. Muon's
    #    update norm is ~constant (orthogonalized momentum), so like all
    #    fixed-step methods it needs a decaying schedule to reach a tight
    #    optimum; use a linear decay (standard practice in the Muon recipe).
    X = torch.randn(64, 16)
    W_true = torch.randn(16, 20)
    Y = X @ W_true

    def run(opt_cls, steps=300, decay=False, **kw):
        W = torch.zeros(16, 20, requires_grad=True)
        opt = opt_cls([W], **kw)
        for t in range(steps):
            if decay:
                for gparam in opt.param_groups:
                    gparam["lr"] = kw.get("lr", 0.02) * (1 - t / steps)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(X @ W, Y)
            loss.backward()
            opt.step()
        return loss.item()

    sgd_final = run(torch.optim.SGD, lr=0.05, momentum=0.9)
    muon_final = run(Muon, lr=0.1, decay=True)
    assert muon_final < sgd_final, \
        f"Muon ({muon_final:.3e}) should beat SGD ({sgd_final:.3e}) here"

    # 3) Mixed model: 2-D params on the Muon path, 1-D on AdamW fallback.
    lin = nn.Linear(16, 20)
    norm = nn.LayerNorm(20)
    opt = Muon(list(lin.parameters()) + list(norm.parameters()), lr=0.01)
    x = torch.randn(8, 16)
    named = [("lin." + n, p) for n, p in lin.named_parameters()] + \
            [("norm." + n, p) for n, p in norm.named_parameters()]
    before = {n: p.detach().clone() for n, p in named}
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(norm(lin(x)), torch.randn(8, 20))
        loss.backward()
        opt.step()
    after = {n: p for n, p in named}
    changed = {n: (after[n] - before[n]).abs().max().item() for n in after}
    assert all(torch.isfinite(p).all() for p in after.values())
    assert changed["lin.weight"] > 0, "Muon path did not update the Linear weight"
    assert changed["lin.bias"] > 0, "AdamW fallback did not update the bias"
    assert changed["norm.weight"] > 0, "AdamW fallback did not update the norm gain"

    # 4) muon=False routes a 2-D param to the AdamW path.
    w = torch.nn.Parameter(torch.randn(8, 8))
    opt2 = Muon([{"params": [w], "muon": False}], lr=0.01)
    for _ in range(2):
        opt2.zero_grad()
        (w ** 2).sum().backward()
        opt2.step()
    assert torch.isfinite(w).all() and opt2.state[w]["step"] == 2, \
        "muon=False group did not take the AdamW path"

    _pass("test_muon",
          f"NS orthog err {eye_err:.2e}, quadratic: muon {muon_final:.2e} "
          f"vs sgd {sgd_final:.2e}, fallback + routing OK")


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
# v5.6: MXFP4 microscaling FP4 (E2M1 codes + E8M0 per-block power-of-two
# scales, MX block 32) — format validity, error bound, odd dims, manager
# integration, MTP rebind
# ----------------------------------------------------------------------
def test_mxfp4():
    from helioslm_v5.src.quantization.standard_quant import (
        MXFP4Linear, QuantizationManager)

    torch.manual_seed(501)
    # Odd in/out exercises the pad path both directions.
    lin = nn.Linear(35, 27, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(27, 35) * 0.05)
        lin.bias.copy_(torch.randn(27) * 0.01)
    x = torch.randn(5, 35)

    mod = MXFP4Linear.from_linear(lin, block_size=32)
    w = mod.weight
    assert w.shape == (27, 35), f".weight shape {w.shape}"
    # E8M0: every stored scale is a power of two.
    lg = torch.log2(mod.scales)
    assert torch.equal(lg, lg.round()), "scales are not all powers of two"
    # Every dequantized magnitude must be a valid E2M1 code times its scale.
    pad = (-35) % 32
    w_blocks = torch.cat([w, w.new_zeros(27, pad)], 1).view(27, -1, 32)
    ratio = w_blocks / mod.scales.unsqueeze(-1)
    mags = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    for v in ratio.flatten().tolist():
        assert abs(v) in mags or abs(abs(v) - round(abs(v), 6)) < 1e-9 and \
            min(abs(abs(v) - m) for m in mags) < 1e-6, \
            f"dequantized value {v} is not an E2M1 code x scale"
    # Reconstruction error: FP4 is coarse (8 magnitudes) but must be bounded.
    rel = ((w - lin.weight).norm() / lin.weight.norm()).item()
    assert rel < 0.35, f"MXFP4 reconstruction error {rel:.3%} too high"
    # forward == F.linear with .weight; deterministic; bias preserved.
    out = mod(x)
    ref = nn.functional.linear(x, w.to(x.dtype), mod.bias.to(x.dtype))
    d = (out - ref).abs().max().item()
    assert d < 1e-5, f"forward disagrees with .weight ({d:.3e})"
    mod2 = MXFP4Linear.from_linear(lin, block_size=32)
    assert torch.equal(mod.qweight, mod2.qweight) \
        and torch.equal(mod.scales, mod2.scales), "not deterministic"

    # QuantizationManager integration: whole-model quantize + MTP rebind.
    model, config = _lite_model(seed=502)
    QuantizationManager(method="mxfp4").quantize_model(model)
    assert isinstance(model.lm_head, MXFP4Linear), \
        f"lm_head should be MXFP4Linear, got {type(model.lm_head)}"
    for i, mtp in enumerate(model.mtp_modules):
        assert mtp.lm_head is model.lm_head, f"mtp[{i}] lm_head not re-bound"
    ids = torch.randint(3, config.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _, _ = model(ids)
    assert torch.isfinite(logits).all(), "mxfp4 model produced non-finite logits"
    assert logits.shape == (1, 6, config.vocab_size)
    _pass("test_mxfp4",
          f"E2M1+E8M0 valid (rel-err {rel:.1%}), odd dims exact, deterministic, "
          "manager + MTP rebind OK")


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


# ----------------------------------------------------------------------
# v5.5 (F1): GatedDeltaAttention — shapes, recurrent == one-shot, decay
# gate range, StateTensor marker, masked tokens never write state
# ----------------------------------------------------------------------
def test_linear_attention():
    from helioslm_v5.src.attention.linear_attention import (
        GatedDeltaAttention, StateTensor, is_recurrent_state)

    config = HeliosLMv5Config(size="lite")
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
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.attention.mla import MLA
    from helioslm_v5.src.attention.linear_attention import (
        GatedDeltaAttention, is_recurrent_state)

    # every=3, 4 layers -> MLA iff i == 0 or i % 3 == 2:
    # [MLA, GDA, MLA, GDA]; layer 0 always MLA.
    torch.manual_seed(141)
    config = HeliosLMv5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    model = HeliosLMv5(config).eval()

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
    from helioslm_v5.src.inference.vllm_engine import VLLMEngine
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
        c = HeliosLMv5Config(size="lite")
        c.hybrid_attention.enabled = True
        c.hybrid_attention.full_attention_every = 1
        c._validate()
    _expect_raises(ValueError, _bad_every, "full_attention_every=1")

    full = HeliosLMv5Config(size="full")  # validation only (no model build)
    assert full.hybrid_attention.enabled is True
    assert full.moe.latent_dim == 1024
    assert full.use_attention_residuals is True
    lite = HeliosLMv5Config(size="lite")
    assert lite.hybrid_attention.enabled is False
    assert lite.moe.latent_dim is None
    assert lite.use_attention_residuals is False
    _pass("test_hybrid_model",
          f"pattern [MLA,GDA,MLA,GDA], cached-decode max|diff|={diff:.2e}, "
          f"generate OK, engine hybrid==greedy (unpadded), size defaults OK")


# ----------------------------------------------------------------------
# v5.5 (F3): LatentMoE — latent dims wired through, output shape, grads
# flow in both latent and full-width variants
# ----------------------------------------------------------------------
def test_latent_moe():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

    config = HeliosLMv5Config(size="lite")
    config.moe.latent_dim = 64
    torch.manual_seed(142)
    moe = DeviceLimitedMoE(config)
    H = config.hidden_size

    # Latent wiring: router + routed experts live in latent space; the
    # shared expert stays full-width.
    assert moe.down_proj.weight.shape == (64, H)
    assert moe.up_proj.weight.shape == (H, 64)
    assert moe.router.weight.shape == (config.moe.num_experts, 64)
    assert moe.experts[0].w13.in_features == 64
    assert moe.experts[0].w2.out_features == 64
    assert moe.shared_experts[0].w13.in_features == H

    x = torch.randn(2, 6, H, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    moe.zero_grad(set_to_none=True)
    out.sum().backward()
    for name, p in (("down_proj", moe.down_proj.weight),
                    ("up_proj", moe.up_proj.weight),
                    ("router", moe.router.weight),
                    ("expert0.w13", moe.experts[0].w13.weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"no gradient through latent MoE: {name}"

    # Full-width variant (v5.4 default) still backprops end-to-end.
    config2 = HeliosLMv5Config(size="lite")
    torch.manual_seed(142)
    moe2 = DeviceLimitedMoE(config2)
    assert moe2.down_proj is None and moe2.up_proj is None
    assert moe2.experts[0].w13.in_features == H
    x2 = torch.randn(2, 6, H, requires_grad=True)
    moe2(x2).sum().backward()
    assert moe2.router.weight.grad is not None \
        and moe2.router.weight.grad.abs().sum() > 0

    # Output differs between latent and full-width (real mechanism, and
    # LatentMoE has fewer routed-expert parameters).
    p_lat = sum(p.numel() for p in moe.experts.parameters())
    p_full = sum(p.numel() for p in moe2.experts.parameters())
    assert p_lat < p_full, "latent experts should shrink parameter count"
    _pass("test_latent_moe",
          f"latent=64 wiring OK, grads OK both variants, expert params "
          f"{p_lat} < full-width {p_full}")


# ----------------------------------------------------------------------
# v5.5 (F4): quantile balancing — drives load toward uniform at least as
# well as the heuristic update; bias is actually updated
# ----------------------------------------------------------------------
def test_quantile_balancing():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

    E = HeliosLMv5Config(size="lite").moe.num_experts

    def make(strategy, seed):
        config = HeliosLMv5Config(size="lite")
        config.moe.balance_strategy = strategy
        torch.manual_seed(seed)
        return DeviceLimitedMoE(config).train()

    moe_h = make("heuristic", 143)
    moe_q = make("quantile", 143)
    # Identical initial weights -> any divergence comes from the strategy.
    moe_q.load_state_dict(moe_h.state_dict(), strict=False)

    # Strongly skewed input: a dominant fixed direction makes the initial
    # ranking nearly input-independent -> a few experts are always picked.
    g = torch.Generator().manual_seed(144)
    direction = torch.randn(1, moe_h.hidden_size, generator=g)
    direction = direction / direction.norm()

    def batch(i):
        noise = 0.3 * torch.randn(64, moe_h.hidden_size,
                                  generator=torch.Generator().manual_seed(1000 + i))
        return (5.0 * direction + noise).unsqueeze(0)  # [1, 64, H] tokens

    def run(moe, steps=25):
        for i in range(steps):
            moe(batch(i))  # training mode: stats + margins accumulate
            moe.update_bias()
        # Measure the post-update load distribution over fresh batches.
        moe.expert_load.zero_()
        for i in range(4):
            moe(batch(100 + i))
        load = moe.expert_load.clone()
        moe.expert_load.zero_()
        return load

    load_h = run(moe_h)
    load_q = run(moe_q)
    cv_h = (load_h.std() / load_h.mean().clamp(min=1e-12)).item()
    cv_q = (load_q.std() / load_q.mean().clamp(min=1e-12)).item()
    assert load_h.mean() > 0 and load_q.mean() > 0
    assert cv_q <= cv_h + 1e-6, \
        f"quantile balancing worse than heuristic: CV {cv_q:.3f} vs {cv_h:.3f}"
    assert moe_q.route_bias.abs().sum() > 0, "quantile update never moved bias"
    # Sanity: the skew really is hard for the heuristic at this step size
    # (otherwise the comparison above is vacuous).
    assert cv_h > 0.5, f"test is vacuous: heuristic already balanced (CV {cv_h:.3f})"
    _pass("test_quantile_balancing",
          f"load CV: quantile {cv_q:.3f} <= heuristic {cv_h:.3f}, "
          f"|bias| sum={moe_q.route_bias.abs().sum():.3f}")

    # Config validation of the strategy enum.
    def _bad_strategy():
        c = HeliosLMv5Config(size="lite")
        c.moe.balance_strategy = "softmax"
        c._validate()
    _expect_raises(ValueError, _bad_strategy, "unknown balance_strategy")


# ----------------------------------------------------------------------
# v5.5 (F5a): attention residuals — on/off changes output, per-layer gate
# exists, is learnable and receives gradient (layers >= 1)
# ----------------------------------------------------------------------
def test_attention_residuals():
    from helioslm_v5.src.model_v5 import HeliosLMv5

    torch.manual_seed(145)
    cfg_on = HeliosLMv5Config(size="lite")
    cfg_on.use_attention_residuals = True
    model_on = HeliosLMv5(cfg_on)
    torch.manual_seed(145)
    cfg_off = HeliosLMv5Config(size="lite")  # v5.4 default: off
    model_off = HeliosLMv5(cfg_off)

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
# v5.5 (F5b): SiTU-GLU — soft cap is bounded, expert output finite under
# huge inputs, gradients flow
# ----------------------------------------------------------------------
def test_situ_glu():
    from helioslm_v5.src.moe.sigmoid_moe import (
        DeviceLimitedMoE, SiTUExpert, SwiGLUExpert, _situ_cap, build_expert)

    softcap = 3.0
    x_big = torch.randn(4, 32) * 1000.0
    t = _situ_cap(x_big, softcap)
    # |t(x)| < softcap mathematically; fp32 tanh saturates to exactly 1.0
    # for large |x|, hence <= with a rounding allowance.
    assert bool((t.abs() <= softcap + 1e-6).all()), \
        f"soft cap violated: |t(x)| max {t.abs().max():.3e} > {softcap}"
    # Identity near zero (t(x) ~= x for |x| << softcap).
    x_small = torch.randn(8, 16) * 0.01
    assert ((_situ_cap(x_small, softcap) - x_small).abs().max()
            < 1e-4), "soft cap should be ~identity near zero"

    config = HeliosLMv5Config(size="lite")
    config.moe.activation = "situ"
    config.moe.situ_softcap = softcap
    torch.manual_seed(146)
    expert = build_expert(config, 32, 32)
    assert isinstance(expert, SiTUExpert)
    x = (torch.randn(4, 32) * 100.0).requires_grad_(True)
    out = expert(x)
    assert out.shape == (4, 32) and torch.isfinite(out).all(), \
        "SiTU expert output not finite under huge inputs"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() \
        and x.grad.abs().sum() > 0, "no gradient through SiTU-GLU"
    assert expert.w13.weight.grad is not None \
        and expert.norm.weight.grad is not None

    # MoE-level: "situ" experts are wired everywhere, "swiglu" is default.
    moe = DeviceLimitedMoE(config)
    assert all(isinstance(e, SiTUExpert) for e in moe.experts)
    assert all(isinstance(e, SiTUExpert) for e in moe.shared_experts)
    h = torch.randn(2, 5, config.hidden_size)
    out_m = moe(h)
    assert out_m.shape == h.shape and torch.isfinite(out_m).all()

    config_sw = HeliosLMv5Config(size="lite")
    assert isinstance(build_expert(config_sw, 32, 32), SwiGLUExpert)

    def _bad_act():
        c = HeliosLMv5Config(size="lite")
        c.moe.activation = "gelu"
        c._validate()
    _expect_raises(ValueError, _bad_act, "unknown activation")
    _pass("test_situ_glu",
          f"|t(x)| < {softcap} (max {t.abs().max():.3f} @|x|=1000), "
          f"finite output, grads OK, MoE wiring OK, bad activation raises")


# ----------------------------------------------------------------------
# v5.5 (M1/M2 regression): quantile margin boundary — margin > 0 iff the
# expert is actually selected (P(margin>0 | selected) == 1.0), and the
# quantile update keeps route_bias bounded over 300 updates (fixed point,
# not the unbounded linear drift of the off-by-one boundary)
# ----------------------------------------------------------------------
def test_quantile_margin_boundary():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

    config = HeliosLMv5Config(size="lite")
    config.moe.balance_strategy = "quantile"
    torch.manual_seed(160)
    moe = DeviceLimitedMoE(config).train()
    E, K = moe.num_experts, moe.top_k

    # Known selection: a huge bias on expert 3 forces it into every token's
    # top-k; the rest of the ranking is data-dependent.
    with torch.no_grad():
        moe.route_bias[3] = 100.0
    flat = torch.randn(12, config.hidden_size)
    topk_indices, _ = moe._route(flat)  # train mode + grad on: records margins
    N = flat.shape[0]
    n = int(moe.margin_count)
    assert n == N, f"expected {N} recorded margins, got {n}"
    window = moe.margin_buffer[:, moe._MARGIN_BUF - n:]
    selected = torch.zeros(E, N, dtype=torch.bool)
    selected.scatter_(0, topk_indices.t(), True)
    margin_pos = window > 0
    p_pos_given_sel = margin_pos[selected].float().mean().item()
    assert p_pos_given_sel == 1.0, \
        f"P(margin>0 | selected) = {p_pos_given_sel:.3f} != 1.0 " \
        "(boundary off-by-one regression)"
    p_pos_given_unsel = margin_pos[~selected].float().mean().item()
    assert p_pos_given_unsel == 0.0, \
        f"P(margin>0 | not selected) = {p_pos_given_unsel:.3f} != 0.0"
    assert bool(margin_pos[3].all()), "forced expert must have margin > 0"

    # Boundedness: 300 forward+update cycles on a strongly skewed input
    # distribution. With the fixed boundary the bias converges to a fixed
    # point (selection frequency -> K/E); the off-by-one boundary made the
    # K/E target unreachable and the bias drifted linearly without bound
    # (mean |bias| ~3.65 after 300 steps in the buggy version).
    torch.manual_seed(161)
    moe2 = DeviceLimitedMoE(config).train()
    g = torch.Generator().manual_seed(162)
    direction = torch.randn(1, config.hidden_size, generator=g)
    direction = direction / direction.norm()

    def batch(seed):
        noise = 0.3 * torch.randn(64, config.hidden_size,
                                  generator=torch.Generator().manual_seed(seed))
        return (5.0 * direction + noise).unsqueeze(0)

    hist = {}
    for i in range(300):
        moe2(batch(2000 + i % 50))
        moe2.update_bias()
        if i in (250, 299):
            hist[i] = moe2.route_bias.detach().clone()
    max_abs = moe2.route_bias.abs().max().item()
    assert torch.isfinite(moe2.route_bias).all()
    assert max_abs < 2.0, \
        f"route_bias unbounded after 300 updates: max|bias|={max_abs:.3f}"
    late_drift = (hist[299] - hist[250]).abs().max().item()
    assert late_drift < 0.5, \
        f"bias still drifting late in training (fixed point not reached): " \
        f"{late_drift:.3f}"
    # The K/E target is now reachable: post-convergence load is near uniform.
    moe2.expert_load.zero_()
    for i in range(4):
        moe2(batch(3000 + i))
    load = moe2.expert_load.clone()
    moe2.expert_load.zero_()
    cv = (load.std() / load.mean().clamp(min=1e-12)).item()
    assert cv < 0.3, f"post-convergence load CV {cv:.3f} not near-uniform"
    _pass("test_quantile_margin_boundary",
          f"P(margin>0|selected)=1.0 exactly, 300 updates: max|bias|="
          f"{max_abs:.3f} (bounded), late drift {late_drift:.3f}, "
          f"load CV {cv:.3f}")


# ----------------------------------------------------------------------
# v5.6: hybrid model SUPPORTS packed position_ids — the recurrent state is
# zeroed at each document boundary, giving exact cross-document isolation
# (mirrors the MLA packed-isolation test); pure-MLA models unaffected;
# decode-with-past at a position restart still raises loudly
# ----------------------------------------------------------------------
def test_hybrid_packed_positions():
    from helioslm_v5.src.model_v5 import HeliosLMv5

    torch.manual_seed(172)
    config = HeliosLMv5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    hybrid = HeliosLMv5(config).eval()

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
    mla_model = HeliosLMv5(HeliosLMv5Config(size="lite")).eval()
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
# v5.5 (M4/M5 regression): attention-residual accumulator is threaded
# explicitly through the DualPipe path — attn_res_gate receives gradient
# under run_dual, and pipeline gradients match the direct forward+backward
# reference bitwise (single stage AND a two-stage split where the
# accumulator crosses the stage boundary)
# ----------------------------------------------------------------------
def test_attention_residuals_dualpipe():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.training.dualpipe import (
        DualPipeScheduler, DualPipeStage, LayerWrap)

    torch.manual_seed(170)
    config = HeliosLMv5Config(size="lite")
    config.use_attention_residuals = True
    model = HeliosLMv5(config)
    assert all(l.attn_res_gate is not None for l in model.layers)

    torch.manual_seed(171)
    inputs = [torch.randn(2, 5, config.hidden_size) for _ in range(4)]
    loss_fn = lambda out: out.pow(2).mean()
    M = len(inputs)

    def run_pair(stages, reference):
        """run_dual on `stages` vs a direct forward+backward `reference`;
        return (loss_pipe, loss_ref, max_grad_diff, grads_pipe)."""
        model.zero_grad(set_to_none=True)
        sched = DualPipeScheduler(stages, num_micro_batches=M)
        loss_pipe = sched.run_dual(inputs, loss_fn)
        grads_pipe = {n: p.grad.clone()
                      for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        loss_ref = 0.0
        for x in inputs:
            loss = loss_fn(reference(x)) / M
            loss.backward()
            loss_ref += loss.item()
        assert abs(loss_pipe - loss_ref) < 1e-6, \
            f"loss mismatch: pipe={loss_pipe:.8f} ref={loss_ref:.8f}"
        max_d = 0.0
        for n, p in model.named_parameters():
            if n in grads_pipe:
                assert p.grad is not None, f"{n}: grad missing in reference"
                d = (grads_pipe[n] - p.grad).abs().max().item()
                max_d = max(max_d, d)
        return loss_pipe, loss_ref, max_d, grads_pipe

    # --- 1) single stage holding both layers ---
    stage = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0]),
                                         LayerWrap(model.layers[1])]))
    assert stage.threads_attn_res
    _, _, d1, grads1 = run_pair([stage], lambda x: stage(x)[0])
    g1 = grads1.get("layers.1.attn_res_gate")
    assert g1 is not None and g1.abs().item() > 0, \
        "attn_res_gate got no gradient through DualPipe (M4 regression)"
    assert d1 < 1e-6, f"1-stage pipe vs direct grad diff {d1:.3e}"

    # --- 2) two-stage split: the accumulator must cross the stage
    #    boundary through the scheduler's detached-carry leaves ---
    s0 = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0])]))
    s1 = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[1])]))

    def ref2(x):
        h, ares = s0(x)
        h, _ = s1(h, ares)
        return h

    _, _, d2, grads2 = run_pair([s0, s1], ref2)
    g1b = grads2.get("layers.1.attn_res_gate")
    assert g1b is not None and g1b.abs().item() > 0, \
        "attn_res_gate grad missing when the accumulator crosses stages"
    assert d2 < 1e-6, f"2-stage pipe vs direct grad diff {d2:.3e}"

    # --- 3) residuals OFF: LayerWrap/stage degrade to the plain v5.4
    #    tensor->tensor contract (no carry threaded) ---
    model_off, _ = _lite_model(seed=175)
    stage_off = DualPipeStage(nn.ModuleList([LayerWrap(model_off.layers[0]),
                                             LayerWrap(model_off.layers[1])]))
    assert not stage_off.threads_attn_res
    assert torch.is_tensor(stage_off(torch.randn(1, 3, config.hidden_size)))
    _pass("test_attention_residuals_dualpipe",
          f"gate grads flow under DualPipe (|g1|={g1.abs().item():.3e}, "
          f"2-stage |g1|={g1b.abs().item():.3e}), pipe==direct bitwise "
          f"(max diff {max(d1, d2):.1e}), residuals-off path unchanged")


# ----------------------------------------------------------------------
# v5.5 (m6/m7 regression): dtype-aware decay clamp (fp16 upper bound must
# stay strictly below 1.0) and a loud error for seq_len == 0
# ----------------------------------------------------------------------
def test_linear_attention_guards():
    from helioslm_v5.src.attention.linear_attention import GatedDeltaAttention

    config = HeliosLMv5Config(size="lite")
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


def test_qat():
    import torch.nn.functional as F
    from helioslm_v5.src.quantization.qat import FakeQuantLinear, apply_qat
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, MXFP4Linear)

    torch.manual_seed(11)
    lin = nn.Linear(64, 32)
    fq = FakeQuantLinear(lin, method="mxfp4")
    x = torch.randn(5, 64)
    out = fq(x)
    # Forward runs on the quantization grid (exactly the qdq matmul).
    ref_w = fq.fake_quant_weight()
    ref = F.linear(x, ref_w, fq.bias)
    assert (out - ref).abs().max().item() == 0.0, (
        "QAT forward must equal the quantize-dequantize matmul exactly")

    # STE backward: the weight gradient equals the DENSE gradient evaluated
    # at the quantized point (identity through the staircase).
    ref_leaf = ref_w.clone().requires_grad_(True)
    bias_leaf = fq.bias.detach().clone().requires_grad_(True)
    F.linear(x, ref_leaf, bias_leaf).pow(2).sum().backward()
    fq.zero_grad()
    out.pow(2).sum().backward()
    assert (fq.weight.grad - ref_leaf.grad).abs().max().item() == 0.0, (
        "STE gradient != dense gradient at the quantized point")

    # QAT training actually converges on a toy least-squares task (the
    # optimizer moves the full-precision weight behind the fake quantizer).
    torch.manual_seed(12)
    w_target = torch.randn(16, 16)
    fq2 = FakeQuantLinear(nn.Linear(16, 16), method="awq", group_size=8)
    opt = torch.optim.Adam(fq2.parameters(), lr=0.05)
    xt = torch.randn(64, 16)
    yt = xt @ w_target.t()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (fq2(xt) - yt).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (
        f"QAT did not converge: {losses[0]:.3f} -> {losses[-1]:.3f}")

    # apply_qat wraps every nn.Linear in place and skips quantized modules
    # (double-wrapping a reconstruction would fake-quantize noise).
    container = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    applied = apply_qat(container, method="mxfp4")
    assert len(applied) == 2 and all(
        isinstance(m, FakeQuantLinear) for m in container), "apply_qat wiring"
    q_container = nn.Sequential(
        MXFP4Linear.from_linear(nn.Linear(8, 8)), nn.Linear(8, 4))
    apply_qat(q_container, method="mxfp4")
    assert isinstance(q_container[0], MXFP4Linear), (
        "already-quantized module must not be re-wrapped")
    assert isinstance(q_container[1], FakeQuantLinear)
    _expect_raises(ValueError, lambda: FakeQuantLinear(
        nn.Linear(4, 4), method="bogus"), "unknown QAT method")
    _pass("test_qat",
          f"fwd==qdq exact, STE grad exact, toy loss "
          f"{losses[0]:.2f}->{losses[-1]:.2f}, wrap/skip OK")


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


def test_sparse_top_k_attention():
    from helioslm_v5.src.attention.mla import MLA

    config = HeliosLMv5Config(size="lite")
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
    config_d = HeliosLMv5Config(size="lite")
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
    config_big = HeliosLMv5Config(size="lite")
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
    config1 = HeliosLMv5Config(size="lite")
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
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg_s = HeliosLMv5Config(size="lite")
    cfg_s.attention.sparse_top_k = 4
    torch.manual_seed(304)
    model_s = HeliosLMv5(cfg_s).eval()
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
    model_s2 = HeliosLMv5(cfg_s).eval()
    model_s2.load_state_dict(model_s.state_dict())
    assert torch.equal(lg_s, decode(model_s2)), (
        "sparse top-k decode is not deterministic")

    cfg_big = HeliosLMv5Config(size="lite")
    cfg_big.attention.sparse_top_k = 10**6
    model_big = HeliosLMv5(cfg_big).eval()
    model_big.load_state_dict(model_s.state_dict())
    cfg_m = HeliosLMv5Config(size="lite")
    model_m = HeliosLMv5(cfg_m).eval()
    model_m.load_state_dict(model_s.state_dict())
    d_model = (decode(model_big) - decode(model_m)).abs().max().item()
    assert d_model == 0.0, (
        f"model-level sparse_top_k >= kv_len must equal dense exactly: "
        f"{d_model:.3e}")

    # Loud errors: non-absorbed cache rejects the flag; bad values fail at
    # config validation.
    bad = HeliosLMv5Config(size="lite")
    bad.attention.use_absorption = False
    bad.attention.sparse_top_k = 4
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=bad.attention), "sparse_top_k w/o absorption")
    bad2 = HeliosLMv5Config(size="lite")
    bad2.attention.sparse_top_k = 0
    _expect_raises(ValueError, lambda: HeliosLMv5Config(
        size="lite", attention=bad2.attention), "sparse_top_k = 0")
    _pass("test_sparse_top_k_attention",
          f"cache bit-equal dense, k>=L exact (unit+model), k=1 W_UV form "
          f"{d_k1:.1e}, deterministic, guards OK")


def test_per_head_muon():
    from helioslm_v5.src.training.muon import (
        Muon, zeropower_via_newtonschulz5)

    torch.manual_seed(701)
    n_heads, head_dim, cols = 4, 8, 16
    # 1) Exact per-head semantics: the optimizer update equals a manual
    #    per-head Newton-Schulz of the (nesterov) momentum.
    p1 = torch.nn.Parameter(torch.zeros(n_heads * head_dim, cols))
    g = torch.randn(n_heads * head_dim, cols)
    opt = Muon([p1], lr=0.1, momentum=0.9, per_head_dim=head_dim)
    p1.grad = g.clone()
    opt.step()
    # First step: buf = g, d = g + 0.9*g = 1.9*g.
    d = 1.9 * g
    manual = torch.stack([
        zeropower_via_newtonschulz5(d[i * head_dim:(i + 1) * head_dim])
        for i in range(n_heads)])
    scale = max(1.0, head_dim / cols) ** 0.5
    want = -0.1 * scale * manual.reshape(n_heads * head_dim, cols)
    assert torch.allclose(p1.data, want, atol=1e-6), (
        f"per-head update mismatch: {(p1.data - want).abs().max().item():.3e}")

    # 2) Per-head blocks are independently (approximately) orthogonalized —
    #    each head's singular values sit in the documented NS band.
    sv_bands = []
    for i in range(n_heads):
        sv = torch.linalg.svdvals(manual[i])
        sv_bands.append((sv.min().item(), sv.max().item()))
    assert all(0.25 < lo and hi < 1.4 for lo, hi in sv_bands), (
        f"per-head NS singular values out of band: {sv_bands}")

    # 3) Convergence with per-head routing (quadratic toward a target).
    W = torch.zeros(n_heads * head_dim, cols, requires_grad=True)
    tgt = torch.randn(n_heads * head_dim, cols)
    opt2 = Muon([W], lr=0.1, per_head_dim=head_dim)
    steps = 200
    for t in range(steps):
        for gp in opt2.param_groups:
            gp["lr"] = 0.1 * (1 - t / steps)
        opt2.zero_grad()
        loss = ((W - tgt) ** 2).sum()
        loss.backward()
        opt2.step()
    err = ((W - tgt) ** 2).sum().item() / (tgt ** 2).sum().item()
    assert err < 0.2, f"per-head Muon did not converge: rel err {err:.3f}"

    # 4) Loud errors: a row count not divisible by per_head_dim raises
    #    (silently falling back to whole-matrix NS would hide a layout
    #    bug); non-positive per_head_dim is rejected at construction.
    p2 = torch.nn.Parameter(torch.randn(10, cols))
    opt3 = Muon([p2], lr=0.01, per_head_dim=head_dim)  # 10 % 8 != 0
    p2.grad = torch.randn_like(p2)
    _expect_raises(ValueError, opt3.step,
                   "per_head_dim not dividing rows")
    _expect_raises(ValueError, lambda: Muon([p2], per_head_dim=0),
                   "per_head_dim=0")
    _pass("test_per_head_muon",
          f"manual per-head NS exact (atol 1e-6), bands OK, converge rel "
          f"err {err:.3f}, guards OK")


def test_gptq_act_order():
    from helioslm_v5.src.quantization.standard_quant import GPTQLinear

    torch.manual_seed(801)
    in_f, out_f, group_size = 32, 16, 16
    # Heterogeneous column activation scales — the regime act-order is
    # designed for (high-variance columns are quantized first so the
    # still-dense low-variance columns can absorb their error).
    col_scale = torch.logspace(-2, 2, in_f)
    X = torch.randn(512, in_f) * col_scale
    W_lin = nn.Linear(in_f, out_f, bias=False)

    def out_err(mod):
        with torch.no_grad():
            ref = X @ W_lin.weight.t()
            got = mod(X)
        return ((got - ref).norm() / ref.norm()).item()

    q_rtn = GPTQLinear.from_linear(W_lin, group_size=group_size)
    q_plain = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                     calibration_data=X, act_order=False)
    q_act = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                   calibration_data=X, act_order=True)
    e_rtn, e_plain, e_act = out_err(q_rtn), out_err(q_plain), out_err(q_act)
    assert e_act <= e_plain * 1.05, (
        f"act-order hurt output error: {e_act:.4f} vs plain {e_plain:.4f}")
    assert e_act < e_rtn, (
        f"act-order GPTQ should beat RTN here: {e_act:.4f} vs {e_rtn:.4f}")

    # The act-order packed layout is still exact: g_idx maps every
    # original column to the group it was quantized in, and forward ==
    # matmul with the .weight property.
    assert q_act.g_idx.shape == (in_f,) and q_act.g_idx.max().item() == \
        (in_f + group_size - 1) // group_size - 1, "act-order g_idx broken"
    with torch.no_grad():
        w_eff = q_act.weight
        got = torch.nn.functional.linear(X, w_eff)
        ref = q_act(X)
    assert (got - ref).abs().max().item() == 0.0, (
        ".weight property inconsistent with forward after act-order")

    # Deterministic: same inputs -> same packed buffers.
    q_act2 = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                    calibration_data=X, act_order=True)
    assert torch.equal(q_act.qweight, q_act2.qweight) and \
        torch.equal(q_act.qzeros, q_act2.qzeros), "act-order not deterministic"

    # Plain GPTQ is untouched by the new flag (default path unchanged).
    q_plain2 = GPTQLinear.from_linear(W_lin, group_size=group_size,
                                      calibration_data=X)
    assert torch.equal(q_plain.qweight, q_plain2.qweight), (
        "default GPTQ path changed")
    _pass("test_gptq_act_order",
          f"out err RTN {e_rtn:.4f} -> GPTQ {e_plain:.4f} -> act-order "
          f"{e_act:.4f}, layout exact, deterministic")


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
    test_muon,
    test_streaming_audio,
    test_navit,
    test_navit_input_validation,
    test_quantization,
    test_gptq_calibration,
    test_awq_calibration,
    test_quant_weight_property,
    test_quant_input_validation,
    test_gptq_odd_dims,
    test_mxfp4,
    test_mtp_rebind_after_quantization,
    test_audio_sliding_window,
    # v5.5
    test_linear_attention,
    test_hybrid_model,
    test_latent_moe,
    test_quantile_balancing,
    test_attention_residuals,
    test_situ_glu,
    test_mtp_hybrid_rollback,
    # v5.5 review fixes (M1-M5, m6/m7)
    test_quantile_margin_boundary,
    test_hybrid_packed_positions,
    test_attention_residuals_dualpipe,
    test_linear_attention_guards,
    # v5.7
    test_rope_scaling,
    test_fp8_kv_cache,
    test_hyper_connections,
    test_qat,
    # v5.8
    test_yarn_rope_scaling,
    test_sparse_top_k_attention,
    test_per_head_muon,
    test_gptq_act_order,
]


def main():
    print("HeliosLM v5.8 Test Suite (lite config, CPU)")
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
