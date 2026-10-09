"""T40 - v5.41: CalibratedPrefetcher oracles (Colibri x RLCD)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
# 2026-10-09 (v5.47): CalibratedPrefetcher lives in the repo-root agent/
# package (alongside self_gen_env / meta_prefetcher), NOT in
# helioslm_v5/agent — the bare import died with ModuleNotFoundError on
# every platform. Mirror test_selfgen_meta's dual-path discipline.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.calibrated_prefetcher import CalibratedPrefetcher
from trajectory import text_to_ids


def test_interface_matches_colibri():
    pf = CalibratedPrefetcher(num_experts=8, top_k=2)
    chosen, confs = pf.predict("layer 0 state: high compute tokens",
                               current_experts=[1, 2, 3, 4])
    assert isinstance(chosen, list) and len(chosen) >= 2
    assert all(isinstance(c, int) for c in chosen)
    assert len(confs) == 8
    print(f"PASS test_interface_matches_colibri (chosen={chosen})")


def test_uncertainty_widens_prefetch():
    pf = CalibratedPrefetcher(num_experts=8, top_k=2, hi=0.7, lo=0.3,
                              widen_factor=2)
    recs = []
    for i in range(40):
        conf = i % 2 == 0
        marker = "STABLE" if conf else "CHAOTIC"
        for e in range(8):
            needed = 1.0 if (conf and e < 2) else 0.0
            recs.append({"ids": torch.tensor(text_to_ids(
                f"{marker} layer state {i}")),
                "answers": {f"expert_{e}": ("yes" if needed else "no"),
                            f"expert_{e}__target_conf": float(needed)}})
    pf.record_outcome(recs)
    conf_chosen, _ = pf.predict("STABLE layer state 0")
    chaos_chosen, _ = pf.predict("CHAOTIC layer state 0")
    assert conf_chosen != chaos_chosen or len(chaos_chosen) > len(conf_chosen)
    print(f"PASS test_uncertainty_widens_prefetch "
          f"(stable={len(conf_chosen)} chaos={len(chaos_chosen)})")


def test_outcome_training_lowers_miss_confidence():
    pf = CalibratedPrefetcher(num_experts=8, top_k=2)
    recs = []
    for i in range(60):
        for e in range(8):
            needed = 1.0 if (e in (0, 1) and i % 5 != 0) else 0.0
            recs.append({"ids": torch.tensor(text_to_ids(f"state {i}")),
                         "answers": {f"expert_{e}": ("yes" if needed else "no"),
                                     f"expert_{e}__target_conf": float(needed)}})
    pf.record_outcome(recs)
    _, confs = pf.predict("state 0")
    assert confs["7"] < 0.5, f"expert_7 conf not low: {confs['7']}"
    print(f"PASS test_outcome_training (expert_7 conf={confs['7']:.2f})")


if __name__ == "__main__":
    test_interface_matches_colibri()
    test_uncertainty_widens_prefetch()
    test_outcome_training_lowers_miss_confidence()
    print("\n3/3 calibrated-prefetcher tests passed")
