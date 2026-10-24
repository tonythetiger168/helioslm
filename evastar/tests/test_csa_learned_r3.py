"""T95 — roadmap Phase B: learned compressor round 3 (v5.61).

Pins:

1. the committed round-3 weights (4x, hidden 64, 9600 steps, gelu)
   re-evaluate to the recorded accuracy (one slice), which beats the
   v5.60 artifact on the same slice;
2. the squared-ReLU compressor option: zero-initialised it is bitwise
   mean pooling like the gelu one, its activation is relu(x)**2, and an
   unknown activation is refused;
3. the artifact is registered.

Run from repo root: python3 evastar/tests/test_csa_learned_r3.py
"""
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

import csa_learned_compressor as script  # noqa: E402
from hca_ab_passkey import probe_accuracy  # noqa: E402
from evastar.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from evastar.src.attention.kv_compress import (  # noqa: E402
    CompressedSparseAttention)

_NAME = "csa_learned_r3_2026-10-15.json"
_ART = json.loads((_REPO / "benchmarks" / _NAME).read_text())
_P = _ART["params"]


class _Args:
    dim, heads, hidden, act = _P["dim"], _P["heads"], _P["hidden"], _P["act"]


def test_round3_slice_reevaluates_and_beats_v560():
    assert (_P["hidden"], _P["steps"], _P["act"], _P["ratios"]) == \
        (64, 9600, "gelu", [4])
    row = _ART["results"][0]
    state = script._from_json(row["compressor_state"])
    acc = probe_accuracy(lambda: script.make_csa(_Args, 4, 8, state), 1024,
                         _P["pairs"], _P["dim"], _P["amp"], _P["boost"],
                         _P["trials"], _P["eval_seed"])
    assert acc == row["eval"]["L=1024/k8"]["accuracy"], acc
    prev = json.loads((_REPO / "benchmarks" /
                       "csa_learned_h64_2026-10-15.json").read_text())
    prev_acc = next(r for r in prev["results"]
                    if r["ratio"] == 4)["eval"]["L=1024/k8"]["accuracy"]
    assert acc > prev_acc, (acc, prev_acc)
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_NAME, reg) == "measured"
    print(f"PASS test_round3_slice_reevaluates_and_beats_v560 "
          f"(L=1024: {prev_acc} -> {acc})")


def test_relu2_option():
    torch.manual_seed(1)
    base = CompressedSparseAttention(32, 2, ratio=4, topk=3,
                                     compressor="learned").eval()
    sq = CompressedSparseAttention(32, 2, ratio=4, topk=3,
                                   compressor="learned",
                                   compressor_act="relu2").eval()
    sq.load_state_dict(base.state_dict())
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        assert torch.equal(base(x), sq(x))       # both start as mean pooling
        t = torch.randn(10)
        assert torch.equal(sq.compressor_mlp[1](t), torch.relu(t) ** 2)
    try:
        CompressedSparseAttention(32, 2, compressor="learned",
                                  compressor_act="tanh")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown compressor_act accepted")
    print("PASS test_relu2_option")


if __name__ == "__main__":
    test_round3_slice_reevaluates_and_beats_v560()
    test_relu2_option()
