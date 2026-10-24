"""T117 — roadmap Phase B: 16x CSA compressor by curriculum (v5.78).

Pins:

1. transfer_state carries the ratio-4 MLP over unchanged and restarts the
   per-position bias at zero for the new chunk size; training from it
   starts from those weights (not from the zero init);
2. the committed 16x weights re-evaluate to the recorded accuracy (one
   slice), the zero-shot row (ratio-4 weights at 16x, no 16x training)
   re-evaluates too, and the order is v5.60 < zero-shot < fine-tuned <=
   oracle ceiling at both lengths;
3. the artifact records its curriculum source and is registered.

Run from repo root: python3 evastar/tests/test_csa_learned_16x.py
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

_NAME = "csa_learned_16x_2026-10-21.json"
_ART = json.loads((_REPO / "benchmarks" / _NAME).read_text())
_P = _ART["params"]
_SRC = json.loads((_REPO / _P["init_from"]).read_text())


class _Args:
    dim, heads, hidden, act = _P["dim"], _P["heads"], _P["hidden"], _P["act"]


def _src_state():
    return script._from_json(next(r for r in _SRC["results"]
                                  if r["ratio"] == 4)["compressor_state"])


def test_transfer_state():
    src = _src_state()
    st = script.transfer_state(src, 16, _P["heads"])
    assert st["compressor_pos"].shape == (16, _P["heads"])
    assert torch.equal(st["compressor_pos"], torch.zeros(16, _P["heads"]))
    for k, v in src.items():
        if k != "compressor_pos":
            assert torch.equal(st[k], v), k
    assert src["compressor_pos"].shape == (4, _P["heads"])   # untouched
    m = script.make_csa(_Args, 16, 8, st)
    got = script.compressor_state(m)
    assert all(torch.equal(got[k], st[k]) for k in st)
    print("PASS test_transfer_state")


def test_16x_slice_and_ordering():
    assert (_P["ratios"], _P["hidden"], _P["init_from"]) == \
        ([16], 64, "benchmarks/csa_learned_r3_2026-10-15.json")
    row = _ART["results"][0]
    state = script._from_json(row["compressor_state"])

    def acc(st, L):
        return probe_accuracy(lambda: script.make_csa(_Args, 16, 8, st), L,
                              _P["pairs"], _P["dim"], _P["amp"], _P["boost"],
                              _P["trials"], _P["eval_seed"])
    a = acc(state, 1024)
    assert a == row["eval"]["L=1024/k8"]["accuracy"], a
    z = acc(script.transfer_state(_src_state(), 16, _P["heads"]), 1024)
    assert z == row["zero_shot_eval"]["L=1024/k8"]["accuracy"], z
    prev = json.loads((_REPO / "benchmarks" /
                       "csa_learned_h64_2026-10-15.json").read_text())
    prev16 = next(r for r in prev["results"] if r["ratio"] == 16)["eval"]
    ceil = _ART["oracle_ceiling"]["ratio=16"]
    for L in (512, 1024):
        k = f"L={L}/k8"
        seq = (prev16[k]["accuracy"], row["zero_shot_eval"][k]["accuracy"],
               row["eval"][k]["accuracy"], ceil[f"L={L}"])
        assert seq[0] < seq[1] < seq[2] <= seq[3], (L, seq)
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_NAME, reg) == "measured"
    print(f"PASS test_16x_slice_and_ordering (L=1024: "
          f"{prev16['L=1024/k8']['accuracy']} -> zero-shot {z} -> {a}, "
          f"ceiling {ceil['L=1024']})")


if __name__ == "__main__":
    test_transfer_state()
    test_16x_slice_and_ordering()
