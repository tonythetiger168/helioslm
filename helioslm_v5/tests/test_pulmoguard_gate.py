"""v5.218: PulmoGuard report bake-off pre-registration and gate
(docs/prereg/pulmoguard-report.json, eval/pulmoguard_gate.py).

Pins:
1. the committed plan validates and locks, is a draft (revisions PENDING,
   so check_gates refuses it), gates unverified_facts <= 0 and the
   physician-score bootstrap lower bound > -2 against every open model
   (incl. unmodified Gemma 4 E4B), and does not gate the reference ceiling;
2. a full gate computation on synthetic reports and ratings (pinned copy
   of the plan, three seeds, paired bootstrap): a grounded model within
   2 points passes; one fabricated number fails the grounding gate; a
   rating far below a baseline fails that baseline's gate;
3. omission rate and missing-field accuracy come from the grader;
4. missing ratings / seeds / report sets are refused.
No patient data: findings are synthetic and ratings are made up.
"""
import copy
import json
import random
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # v5.225: this file is also executed directly by the daily TESTS
    # suite (case_upstream_suites), where pytest is not installed. The
    # shim covers exactly the one pytest feature used below:
    # raises(match=...) (pytest semantics: re.search on str(exception)).
    import contextlib as _contextlib
    import re as _re

    class _PytestShim:
        @staticmethod
        @_contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match is not None and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"raised {e!r}, which does not match /{match}/"
                    ) from None
                return
            raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _PytestShim()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval.prereg import PreregError, check_gates, lock, pending, validate  # noqa: E402
from helioslm_v5.eval.pulmoguard_gate import (  # noqa: E402
    build_results, load_ratings, report_metrics)
from helioslm_v5.src.training.report_abstain import (  # noqa: E402
    remove_fields, render_reference, synth_findings)

PLAN = json.loads((ROOT / "docs" / "prereg" / "pulmoguard-report.json").read_text(encoding="utf-8"))
OPEN = ["Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "google/medgemma-27b-text-it",
        "Llama-Breeze2-8B-Instruct", "google/gemma-4-E4B"]
OURS = "HeliosLM-B1 + PulmoGuard SFT"


def test_committed_plan():
    validate(PLAN)
    assert len(lock(PLAN)) == 64
    assert pending(PLAN)
    with pytest.raises(PreregError, match="PENDING"):
        check_gates(PLAN, {})
    gates = PLAN["gates"]
    assert {"task": "report_grounding", "metric": "unverified_facts", "op": "<=",
            "threshold": 0} in gates
    rel = {g["vs"]: g["margin"] for g in gates if "vs" in g}
    assert rel == {m: -2.0 for m in OPEN}
    names = {b["name"] for b in PLAN["baselines"]}
    assert {"claude-sonnet-5-5", "HeliosLM-B1"} <= names and set(OPEN) <= names
    assert PLAN["seeds"] == [0, 1, 2]
    text = json.dumps(PLAN, ensure_ascii=False).lower()
    assert "patient" not in text.replace("no patient data", "")


def _pinned():
    p = copy.deepcopy(PLAN)
    for x in p["tasks"] + p["baselines"]:
        x["revision"] = "synthetic-test"
    return p


def _reports(n=40, seed=0):
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        f, _ = remove_fields(synth_findings(rng), rng, 0.3)
        cases.append({"report_id": f"r{i}", "findings": f,
                      "lang": rng.choice(("zh", "en"))})
    return cases


def _rows(cases, corrupt=()):
    out = []
    for c in cases:
        text = render_reference(c["findings"], c["lang"])
        if c["report_id"] in corrupt:      # fabricate: change the first diameter
            n = next(x for x in c["findings"]["nodules"] if x["diameter_mm"] is not None)
            text = text.replace(f"{n['diameter_mm']:g}", f"{n['diameter_mm'] + 5:g}", 1)
        out.append({"report_id": c["report_id"], "findings": c["findings"], "text": text})
    return out


def _ratings(cases, means, seeds=(0, 1, 2)):
    rng = random.Random(1)
    rows = []
    for model, mu in means.items():
        for c in cases:
            base = mu + rng.uniform(-0.5, 0.5)
            if model == OURS:
                for s in seeds:
                    rows += [{"model": model, "report_id": c["report_id"], "seed": s,
                              "rater": r, "score": base + rng.uniform(-0.2, 0.2)}
                             for r in ("A", "B")]
            else:
                rows.append({"model": model, "report_id": c["report_id"], "score": base})
    return rows


def _setup(ours_mean=8.0, corrupt=()):
    cases = _reports()
    reports = {OURS: {s: _rows(cases, corrupt if s == 1 else ()) for s in (0, 1, 2)}}
    means = {OURS: ours_mean, "Qwen/Qwen3-14B": 7.5, "Qwen/Qwen3-32B": 8.3,
             "google/medgemma-27b-text-it": 8.6, "Llama-Breeze2-8B-Instruct": 6.9,
             "google/gemma-4-E4B": 6.5}
    for m in OPEN:
        reports[m] = {0: _rows(cases)}
    return cases, reports, load_ratings(_ratings(cases, means))


def test_full_gate_passes():
    plan = _pinned()
    _, reports, ratings = _setup()
    res = build_results(plan, OURS, reports, ratings, n_boot=2000)
    v = check_gates(plan, res)
    assert v["passed"], v["gates"]
    assert res["scores"]["report_grounding"]["unverified_facts"] == 0
    assert res["scores"]["report_omission"]["omission_rate"] == 0.0
    assert res["scores"]["missing_field_abstention"]["accuracy"] == 1.0
    c = res["comparisons"]["google/medgemma-27b-text-it"]["physician_rating"]["score"]
    assert c["seeds"] == 3 and c["n"] == 40 and -2.0 < c["ci95_low"] < 0
    assert check_gates(plan, build_results(plan, OURS, reports, ratings, n_boot=2000)) == v


def test_one_fabrication_fails():
    plan = _pinned()
    _, reports, ratings = _setup(corrupt=("r3",))
    res = build_results(plan, OURS, reports, ratings, n_boot=500)
    # a changed diameter in a bound sentence is a contradiction: unsupported
    assert res["per_seed"]["1"]["contradictions"] == 1
    assert res["scores"]["report_grounding"]["unverified_facts"] == 1
    v = check_gates(plan, res)
    assert not v["passed"]
    assert [g["pass"] for g in v["gates"] if g["task"] == "report_grounding"] == [False]
    # an extra unbound number is an unverified fact, counted too
    reports[OURS][2][0]["text"] += " 另見 33.3 公釐。"
    res = build_results(plan, OURS, reports, ratings, n_boot=500)
    assert res["scores"]["report_grounding"]["unverified_facts"] == 2


def test_low_rating_fails_against_best_open_model():
    plan = _pinned()
    _, reports, ratings = _setup(ours_mean=5.9)
    v = check_gates(plan, build_results(plan, OURS, reports, ratings, n_boot=2000))
    failed = sorted(g["vs"] for g in v["gates"] if "vs" in g and not g["pass"])
    assert "google/medgemma-27b-text-it" in failed and "google/gemma-4-E4B" not in failed
    assert not v["passed"]


def test_report_metrics_counts_omissions_and_abstention():
    cases = _reports(10, seed=4)
    rows = _rows(cases)
    m = report_metrics(rows)
    assert m["omissions"] == 0 and m["required"] > 0 and m["missing_field_accuracy"] == 1.0
    stripped = [dict(r, text=r["text"].replace("未測量", "").replace("not measured", ""))
                for r in rows]
    if m["missing_fields"]:
        assert report_metrics(stripped)["missing_field_accuracy"] == 0.0


def test_refusals():
    plan = _pinned()
    cases, reports, ratings = _setup()
    bad = dict(reports, **{OURS: {0: reports[OURS][0], 1: reports[OURS][1]}})
    with pytest.raises(PreregError, match="seeds"):
        build_results(plan, OURS, bad, ratings)
    no_rate = {k: v for k, v in ratings.items() if k[0] != "Qwen/Qwen3-14B"}
    with pytest.raises(PreregError, match="no physician rating"):
        build_results(plan, OURS, reports, no_rate)
    short = dict(reports, **{"Qwen/Qwen3-32B": {0: _rows(cases[:-1])}})
    with pytest.raises(PreregError, match="different reports"):
        build_results(plan, OURS, short, ratings)
    missing = {k: v for k, v in reports.items() if k != "google/gemma-4-E4B"}
    with pytest.raises(PreregError, match="no reports for baseline"):
        build_results(plan, OURS, missing, ratings)
