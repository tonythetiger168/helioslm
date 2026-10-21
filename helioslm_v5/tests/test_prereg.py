"""Plan E5: pre-registered evaluation plans.

Pins: the lock is a stable hash of the canonical plan (key order does not
matter, any edit does); validation rejects malformed plans; check_gates
returns per-gate verdicts, refuses results filed against another plan
(e.g. gates edited after seeing scores) and results that omit a
registered task.
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.eval.prereg import PreregError, check_gates, lock, validate  # noqa: E402

PLAN = {
    "release": "v1.0", "model": "0.5B dense",
    "harness": {"name": "lm-evaluation-harness", "version": "0.4.x"},
    "seeds": [0, 1, 2],
    "tasks": [
        {"name": "arc_easy", "revision": "r1", "split": "test", "n_shot": 0,
         "metric": "acc_norm"},
        {"name": "tmmluplus", "revision": "v1.1", "split": "test", "n_shot": 5,
         "metric": "acc"},
        {"name": "wikitext", "revision": "r1", "split": "test", "n_shot": 0,
         "metric": "bits_per_byte"},
    ],
    "gates": [
        {"task": "arc_easy", "metric": "acc_norm", "op": ">=", "threshold": 0.45},
        {"task": "tmmluplus", "metric": "acc", "op": ">=", "threshold": 0.30},
        {"task": "wikitext", "metric": "bits_per_byte", "op": "<=", "threshold": 1.0},
    ],
}


def _results(arc=0.5, tw=0.31, bpb=0.9, sha=None):
    return {"prereg_sha256": sha or lock(PLAN),
            "scores": {"arc_easy": {"acc_norm": arc}, "tmmluplus": {"acc": tw},
                       "wikitext": {"bits_per_byte": bpb}}}


def test_lock_is_canonical_and_edit_sensitive():
    reordered = dict(reversed(list(copy.deepcopy(PLAN).items())))
    assert lock(reordered) == lock(PLAN)
    edited = copy.deepcopy(PLAN)
    edited["gates"][0]["threshold"] = 0.40
    assert lock(edited) != lock(PLAN)
    assert len(lock(PLAN)) == 64


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("gates"),
    lambda p: p["harness"].pop("version"),
    lambda p: p.__setitem__("seeds", [0, 0]),
    lambda p: p.__setitem__("seeds", []),
    lambda p: p["tasks"][0].pop("revision"),
    lambda p: p["tasks"].append(dict(p["tasks"][0])),
    lambda p: p["tasks"][0].__setitem__("n_shot", -1),
    lambda p: p["gates"].append({"task": "gsm8k", "metric": "acc", "op": ">=",
                                 "threshold": 0.1}),
    lambda p: p["gates"][0].__setitem__("metric", "acc"),
    lambda p: p["gates"][0].__setitem__("op", ">"),
    lambda p: p["gates"][0].__setitem__("threshold", "0.4"),
    lambda p: p["gates"][0].__setitem__("threshold", True),
    lambda p: p.__setitem__("gates", []),
])
def test_validation_rejects(mutate):
    bad = copy.deepcopy(PLAN)
    mutate(bad)
    with pytest.raises(PreregError):
        validate(bad)


def test_gate_verdicts():
    ok = check_gates(PLAN, _results())
    assert ok["passed"] and [r["pass"] for r in ok["gates"]] == [True] * 3
    fail = check_gates(PLAN, _results(tw=0.29, bpb=1.2))
    assert not fail["passed"]
    assert [r["pass"] for r in fail["gates"]] == [True, False, False]
    edge = check_gates(PLAN, _results(arc=0.45, bpb=1.0))      # thresholds inclusive
    assert edge["passed"]


def test_results_must_cite_this_plan_and_be_complete():
    moved = copy.deepcopy(PLAN)
    moved["gates"][1]["threshold"] = 0.25          # relaxed after the fact
    with pytest.raises(PreregError, match="different"):
        check_gates(moved, _results(tw=0.27))       # results cite the original lock
    with pytest.raises(PreregError, match="different"):
        check_gates(PLAN, {"scores": _results()["scores"]})
    partial = _results()
    del partial["scores"]["tmmluplus"]
    with pytest.raises(PreregError, match="omit"):
        check_gates(PLAN, partial)
