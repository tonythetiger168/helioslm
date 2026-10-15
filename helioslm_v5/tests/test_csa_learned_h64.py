"""T94 — roadmap Phase B: learned compressor round 2 (v5.60).

Pins:

1. the committed hidden-64 weights re-evaluate to the recorded numbers
   (one 16x slice, the headline that moved from chance);
2. the oracle-pooling ceiling recorded in the artifact recomputes;
3. ``--check`` evaluates with the ARTIFACT's own sizes: loading the
   hidden-64 weights into a module built with the CLI default
   (hidden 16) fails loudly, so a check that ignored the artifact's
   params could not silently pass.

Run from repo root: python3 helioslm_v5/tests/test_csa_learned_h64.py
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

import csa_learned_compressor as script  # noqa: E402
from hca_ab_passkey import probe_accuracy  # noqa: E402
from helioslm_v5.eval.artifact_tiers import load_registry, tier_of  # noqa: E402

_NAME = "csa_learned_h64_2026-10-15.json"
_ART = json.loads((_REPO / "benchmarks" / _NAME).read_text())
_P = _ART["params"]


class _Args:
    dim, heads, hidden = _P["dim"], _P["heads"], _P["hidden"]
    pairs, amp, boost = _P["pairs"], _P["amp"], _P["boost"]
    trials, eval_seed = _P["trials"], _P["eval_seed"]
    seq_lens, topks = _P["seq_lens"], _P["topks"]


def _state(ratio):
    row = next(r for r in _ART["results"] if r["ratio"] == ratio)
    return row, script._from_json(row["compressor_state"])


def test_committed_slice_reevaluates():
    assert _P["hidden"] == 64 and _P["steps"] == 4800
    row, state = _state(16)
    acc = probe_accuracy(lambda: script.make_csa(_Args, 16, 8, state), 512,
                         _P["pairs"], _P["dim"], _P["amp"], _P["boost"],
                         _P["trials"], _P["eval_seed"])
    assert acc == row["eval"]["L=512/k8"]["accuracy"], acc
    assert acc > 2.5 * _ART["chance"], acc       # 16x moved off chance
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_NAME, reg) == "measured"
    print(f"PASS test_committed_slice_reevaluates (16x acc={acc})")


def test_oracle_ceiling_recomputes():
    class One(_Args):
        seq_lens = [512]
    got = script.oracle_ceiling(One, 16)
    assert got["L=512"] == _ART["oracle_ceiling"]["ratio=16"]["L=512"], got
    print(f"PASS test_oracle_ceiling_recomputes (16x ceiling={got['L=512']})")


def test_check_needs_artifact_sizes():
    _, state = _state(4)

    class Default(_Args):
        hidden = 16                          # the CLI default
    try:
        script.make_csa(Default, 4, 8, state)
    except RuntimeError as e:
        assert "size mismatch" in str(e)
    else:
        raise AssertionError("hidden-64 weights loaded into a hidden-16 "
                             "compressor without error")
    script.make_csa(_Args, 4, 8, state)      # artifact sizes load cleanly
    assert set(script.EVAL_KEYS) <= set(_P)  # --check copies these over
    print("PASS test_check_needs_artifact_sizes")


if __name__ == "__main__":
    test_committed_slice_reevaluates()
    test_oracle_ceiling_recomputes()
    test_check_needs_artifact_sizes()
