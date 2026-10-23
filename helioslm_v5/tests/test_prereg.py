"""Plan E5: pre-registered evaluation plans.

Pins: the lock is a stable hash of the canonical plan (key order does not
matter, any edit does); validation rejects malformed plans; check_gates
returns per-gate verdicts, refuses results filed against another plan
(e.g. gates edited after seeing scores) and results that omit a
registered task.

v5.119: relative gates ("beat a registered baseline": the lower end of the
95% paired-bootstrap interval of ours - theirs is above the margin), PENDING
revisions (a plan locks but cannot be checked until pinned), and the
committed v1.0 plan (docs/prereg/v1.0.json).
"""
import json
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.eval.prereg import (  # noqa: E402
    PreregError, check_gates, lock, paired_bootstrap, pending, validate)

ROOT = Path(__file__).resolve().parents[2]

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


REL = copy.deepcopy(PLAN)
REL["baselines"] = [{"name": "breeze2-3b", "revision": "abc123"}]
REL["gates"].append({"task": "tmmluplus", "metric": "acc", "vs": "breeze2-3b",
                     "margin": 0.0})


def _rel_results(low):
    r = _results()
    r["prereg_sha256"] = lock(REL)
    r["comparisons"] = {"breeze2-3b": {"tmmluplus": {"acc": {
        "ours": 0.31, "theirs": 0.30, "ci95_low": low, "ci95_high": low + 0.02}}}}
    return r


@pytest.mark.parametrize("mutate", [
    lambda g: g.__setitem__("vs", "taide-8b"),
    lambda g: g.__setitem__("margin", "0"),
    lambda g: g.pop("margin"),
    lambda g: g.__setitem__("metric", "acc_norm"),
])
def test_relative_gate_validation(mutate):
    bad = copy.deepcopy(REL)
    mutate(bad["gates"][-1])
    with pytest.raises(PreregError):
        validate(bad)


def test_relative_gate_verdicts():
    win = check_gates(REL, _rel_results(0.004))
    assert win["passed"] and win["gates"][-1]["ours"] == 0.31
    tie = check_gates(REL, _rel_results(-0.001))   # point estimate ahead, CI is not
    assert not tie["passed"] and not tie["gates"][-1]["pass"]
    edge = check_gates(REL, _rel_results(0.0))     # strictly above the margin
    assert not edge["passed"]
    missing = _rel_results(0.01)
    del missing["comparisons"]
    with pytest.raises(PreregError, match="omit"):
        check_gates(REL, missing)


def test_pending_revisions_lock_but_do_not_check():
    plan = copy.deepcopy(REL)
    plan["baselines"][0]["revision"] = "PENDING"
    assert pending(plan) == ["breeze2-3b"] and len(lock(plan)) == 64
    r = _rel_results(0.01)
    r["prereg_sha256"] = lock(plan)
    with pytest.raises(PreregError, match="not final"):
        check_gates(plan, r)


def test_paired_bootstrap():
    import numpy as np
    rng = np.random.default_rng(0)
    base = (rng.random(2000) < 0.4).astype(int)
    same = paired_bootstrap([base, base], base, n_boot=2000)
    assert same["diff"] == 0 and same["ci95_low"] == 0 == same["ci95_high"]
    better = base.copy()
    better[np.flatnonzero(base == 0)[:200]] = 1          # +10 points
    win = paired_bootstrap([better] * 3, base, n_boot=2000)
    assert win["diff"] == pytest.approx(0.1) and win["ci95_low"] > 0.08
    assert win["seeds"] == 3 and win["n"] == 2000
    assert paired_bootstrap([better], base, seed=1) == paired_bootstrap([better], base, seed=1)
    with pytest.raises(PreregError, match="shapes"):
        paired_bootstrap([base[:10]], base)


def test_committed_v1_plan():
    plan = json.loads((ROOT / "docs" / "prereg" / "v1.0.json").read_text(encoding="utf-8"))
    validate(plan)
    assert plan["release"] == "v1.0" and plan["seeds"] == [0, 1, 2]
    gated = {(g["task"], g["vs"]) for g in plan["gates"]}
    breeze = "MediaTek-Research/Llama-Breeze2-3B-Instruct"
    assert gated == {("tmmluplus", breeze), ("tw_justice", breeze)}
    assert all(g["margin"] == 0.0 for g in plan["gates"])
    assert set(pending(plan)) >= {"tmmluplus", breeze}   # pinned on the pod
