"""T155 — no host-sync graph breaks in RoPE or the MoE load count, v5.165.

The fourth L4 round (torch 2.8) ran the MoE pilot with ``--compile``
but dynamo hit its recompile limit on the resume function after
``RotaryEmbedding.forward``'s host-synced checks (``.any()`` and
``.item()`` on the position ids). Under compilation the table is now
sized from the static sequence length, and the training-time expert
load count uses ``scatter_add`` instead of ``bincount`` (data-dependent
output length).

Pins: RoPE and the dense pilot compile without graph breaks; the MoE
pilot breaks only at its ``torch.compiler.disable`` dispatch; compiled
RoPE matches eager; the load count equals ``bincount``.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_compile_graph_breaks.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.src.attention.mla import RotaryEmbedding  # noqa: E402
from helioslm_v5.src.training.pretrain import HeliosLMv5, pilot_config  # noqa: E402


def _breaks(fn, *args):
    torch._dynamo.reset()
    return torch._dynamo.explain(fn)(*args).break_reasons


def test_rope_compiles_without_breaks():
    rope = RotaryEmbedding(16, 4096)
    assert _breaks(rope, torch.arange(32)) == []
    torch._dynamo.reset()
    cos_c, sin_c = torch.compile(rope, backend="eager", fullgraph=True)(torch.arange(32))
    cos_e, sin_e = rope(torch.arange(32))
    assert torch.equal(cos_c, cos_e) and torch.equal(sin_c, sin_e)


def test_rope_eager_checks_kept():
    rope = RotaryEmbedding(16, 64)
    try:
        rope(torch.tensor([0, -1]))
    except ValueError:
        pass
    else:
        raise AssertionError("negative position ids must still raise in eager")
    cos, _ = rope(torch.arange(100))  # eager still grows the table
    assert cos.shape[2] == 100 and rope.max_seq_len >= 100


def _model(arch):
    cfg = pilot_config(arch, vocab_size=512, hidden_size=128, num_layers=2,
                       intermediate_size=256, max_seq=128)
    return HeliosLMv5(cfg)


def test_dense_pilot_has_no_breaks():
    assert _breaks(_model("dense"), torch.randint(0, 512, (2, 64))) == []


def test_moe_pilot_breaks_only_at_dispatch():
    reasons = _breaks(_model("moe"), torch.randint(0, 512, (2, 64)))
    assert reasons
    for r in reasons:
        assert "disable" in str(r.reason), str(r.reason).splitlines()[0]


def test_load_count_matches_bincount():
    m = _model("moe").train()
    moe = next(x for x in m.modules() if hasattr(x, "expert_load"))
    moe.expert_load.zero_()
    seen = []
    orig = moe._route

    def spy(x):
        idx, gates = orig(x)
        seen.append(idx.detach().reshape(-1))
        return idx, gates

    moe._route = spy
    m(torch.randint(0, 512, (2, 64)))
    want = torch.bincount(torch.cat(seen), minlength=moe.num_experts)
    assert torch.equal(moe.expert_load.to(want.dtype), want)
