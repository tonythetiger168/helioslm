"""HeliosLM v5 test cases — MLA core (both cache modes) + packed positions.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_lite_model, _pass)


# ----------------------------------------------------------------------
# MLA: cached decode == full forward; cache growth; truthful size numbers
# (runs for BOTH cache modes: expanded and absorbed)
# ----------------------------------------------------------------------
def _check_mla_mode(use_absorption):
    from evastar.src.attention.mla import MLA
    config = EvastarV5Config(size="lite")
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
    from evastar.src.attention.mla import MLA
    config = EvastarV5Config(size="lite")

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
    from evastar.src.attention.mla import MLA
    config = EvastarV5Config(size="lite")
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
    from evastar.src.attention.mla import MLA
    config = EvastarV5Config(size="lite")
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
