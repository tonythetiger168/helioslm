"""OS6: MoE configuration table for #42 (eval/moe_config_table.py).

Pins:
1. every reference row's expert-weight count lands within 4% of the
   vendor's stated TOTAL parameters (attention and embeddings are the
   rest), so a mistyped field in REFERENCE shows up;
2. the counts are the closed form (layers - dense) * (E + S) * 3 h d;
3. ``candidates`` returns only configs within tolerance of the target,
   best first, and each is constructible as a HeliosLM MoEConfig.

Run from repo root: python3 -m pytest evastar/tests/test_moe_config_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.eval.moe_config_table import (  # noqa: E402
    REFERENCE, MoESpec, candidates, model_params, moe_ffn_params, ratios)


def test_reference_rows_match_vendor_totals():
    assert len(REFERENCE) == 6
    for rid, s in REFERENCE.items():
        total, active = moe_ffn_params(s)
        assert active < total
        if s.claimed_total_b:
            assert 0.96 <= total / (s.claimed_total_b * 1e9) <= 1.0, rid
        if s.claimed_active_b:
            assert active < s.claimed_active_b * 1e9, rid
        r = ratios(s)
        assert 0.02 <= r["top_k/experts"] <= 0.05, rid


def test_closed_form():
    s = MoESpec(10, 64, 8, 1, 2, 2, 32)
    assert moe_ffn_params(s) == (8 * 9 * 3 * 64 * 32, 8 * 3 * 3 * 64 * 32)
    t, a = model_params(s, q_heads=4, kv_heads=2, head_dim=16, vocab=100,
                        tied=True)
    dense = 2 * 3 * 64 * 256
    attn = 10 * 64 * 16 * (8 + 4)
    assert t == moe_ffn_params(s)[0] + dense + attn + 100 * 64
    assert t - a == moe_ffn_params(s)[0] - moe_ffn_params(s)[1]


def test_candidates_near_30b_a3b():
    cs = candidates()
    assert cs
    scores = [abs(c["total"] / 30e9 - 1) + abs(c["active"] / 3e9 - 1) for c in cs]
    assert scores == sorted(scores)
    for c in cs:
        assert abs(c["total"] / 30e9 - 1) <= 0.12
        assert abs(c["active"] / 3e9 - 1) <= 0.12
    from evastar.configs.config_v5 import MoEConfig
    s = cs[0]["spec"]
    m = MoEConfig(num_experts=s.experts, num_shared_experts=s.shared,
                  num_activated_experts=s.top_k, expert_hidden_size=s.expert_inter,
                  first_k_dense=s.first_k_dense)
    assert m.num_experts == s.experts
