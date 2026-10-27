"""HeliosLM v5 test cases — trust-gate calibration + decision-head audits.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math
import sys
from pathlib import Path

import torch

from helioslm_v5.tests.case_common import _pass


# ----------------------------------------------------------------------
# v5.43 (3/3): quant-drift-aware trust gate — a positive NVFP4 ECE drift
# widens the abstain band (widen-only policy), provenance is preserved,
# and the merged record passes TrustGateV2's own validation
# ----------------------------------------------------------------------
def test_quant_drift_trust_gate():
    _agent_dir = str(Path(__file__).resolve().parent.parent / "agent")
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from decision_head import DecisionHead
    from gate import Route
    from trust_gate_v2 import TrustGateV2, apply_quant_drift
    from types import SimpleNamespace

    base = {"ece": 0.05, "n": 500}

    # Positive drift widens the band by exactly the drift.
    merged = apply_quant_drift(base, 0.03)
    assert merged["ece"] == 0.08 and merged["quant_drift"] == 0.03
    assert merged["policy"] == "widen-only" and "source" in merged
    # Input record untouched (dict copied, not mutated).
    assert base == {"ece": 0.05, "n": 500}

    # WIDEN-ONLY: a negative drift (quantization helped) leaves the band
    # at the bf16 record — trusting a lucky measurement to make the gate
    # MORE aggressive is a policy choice kept out of the math.
    assert apply_quant_drift(base, -0.04)["ece"] == 0.05

    # Clamp at the TrustGateV2 validation ceiling.
    assert apply_quant_drift(base, 10.0)["ece"] == 0.5

    # Loud errors: no base record, out-of-range base, non-finite drift.
    for bad in (lambda: apply_quant_drift(None, 0.1),
                lambda: apply_quant_drift({"ece": 0.9, "n": 10}, 0.1),
                lambda: apply_quant_drift(base, float("nan")),
                lambda: apply_quant_drift(base, float("inf"))):
        try:
            bad()
            assert False, "bad drift input must fail loudly"
        except ValueError:
            pass

    # End-to-end: at p=0.84 with p*=0.8, a near-perfect bf16 record
    # (ece=0.005, band [0.795, 0.805]) DIRECTs; a +0.10 quant drift widens
    # the band to [0.65, 0.95] and the same p falls INSIDE it -> the gate
    # abstains. Widening buys escalation, never aggression.
    head = DecisionHead({"trust": ("noul", None)})
    g = TrustGateV2(head, calibration={"ece": 0.005, "n": 500})
    g.router.head.forward = lambda ids: {"trust": (0.84, None)}
    call = SimpleNamespace(name="t.call")
    assert g.decide(call, {}) == Route.DIRECT, \
        "tight bf16 band must act at p=0.84"
    g2 = TrustGateV2(DecisionHead({"trust": ("noul", None)}),
                     calibration=apply_quant_drift(base, 0.10))
    g2.router.head.forward = lambda ids: {"trust": (0.84, None)}
    rec = g2.decide_explain(call, {})
    assert rec["route"] == "ESCALATE" and \
        abs(rec["margin"] - 0.15) < 1e-9, rec
    _pass("test_quant_drift_trust_gate",
          "drift +0.03 -> ece 0.08; widen-only; same p=0.84 DIRECTs on "
          "bf16, ESCALATEs with +0.10 quant drift")


# ----------------------------------------------------------------------
# v5.44 (2/3): quant-calib probe -> TrustGate pipeline — the drift is
# recomputed from the two arms (never the report's embedded copy), the
# merged record follows the widen-only policy, and a partial report is
# a loud error
# ----------------------------------------------------------------------
def test_trust_calibration_from_report():
    _agent_dir = str(Path(__file__).resolve().parent.parent / "agent")
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from trust_gate_v2 import trust_calibration_from_report
    from helioslm_v5.eval.quant_calib import ece_drift_from_report

    report1 = {"bf16": {"tag": "m/bf16", "accuracy": 0.6, "ece": 0.05,
                        "n": 100},
               "nvfp4": {"tag": "m/nvfp4-fakequant", "accuracy": 0.55,
                         "ece": 0.08, "n": 100},
               "drift": {"ece": 999.0}}  # stale embedded copy must be ignored
    merged = trust_calibration_from_report(report1)
    assert merged["ece"] == 0.08, "positive drift widens to the nvfp4 ECE"
    assert merged["quant_drift"] == 0.03
    assert merged["policy"] == "widen-only" and merged["n"] == 100

    # WIDEN-ONLY through the pipeline: quantization that HELPED calibration
    # (nvfp4 ece 0.02 < bf16 0.05) keeps the band at the bf16 record.
    report2 = {"bf16": {"ece": 0.05, "n": 100},
               "nvfp4": {"ece": 0.02, "n": 100}}
    assert trust_calibration_from_report(report2)["ece"] == 0.05

    # The drift helper recomputes from the arms, never the embedded field.
    # (0.02 - 0.05 under IEEE754 double is -0.030000000000000002, so the
    # negative-drift side asserts with an explicit 1e-12 tolerance instead
    # of exact equality — the drift is float noise, not semantics.)
    assert abs(ece_drift_from_report(report1) - 0.03) < 1e-12
    assert abs(ece_drift_from_report(report2) - (-0.03)) < 1e-12
    # n override is honored as provenance.
    assert trust_calibration_from_report(report1, n=250)["n"] == 250

    # The merged record passes TrustGateV2's own validation and widens the
    # abstain band exactly like a hand-merged record (parity with the v5.43
    # gate test: p=0.84, p*=0.8, bf16 ece 0.05 + 0.03 drift -> band
    # [0.72, 0.88] -> ESCALATE).
    from types import SimpleNamespace
    from trust_gate_v2 import TrustGateV2
    from decision_head import DecisionHead
    from gate import Route
    gate = TrustGateV2(DecisionHead({"trust": ("noul", None)}),
                       calibration=merged)
    gate.router.head.forward = lambda ids: {"trust": (0.84, None)}
    rec = gate.decide_explain(SimpleNamespace(name="t.call"), {})
    assert rec["route"] == "ESCALATE" and \
        abs(rec["margin"] - 0.08) < 1e-9, rec

    # Loud errors: a report missing either arm (or its ece) is partial
    # evidence and must not manufacture a drift.
    for bad_report in ({}, {"bf16": {"ece": 0.05}},
                       {"nvfp4": {"ece": 0.08}},
                       {"bf16": {"n": 5}, "nvfp4": {"ece": 0.08}}):
        try:
            trust_calibration_from_report(bad_report)
            assert False, "partial report must fail loudly"
        except ValueError:
            pass
    _pass("test_trust_calibration_from_report",
          f"report ece 0.05->0.08 (drift {merged['quant_drift']}); "
          f"widen-only keeps 0.05 when quant helped; stale embedded drift "
          f"ignored; partial reports loud")


# ----------------------------------------------------------------------
# v5.47 (2/3): quant-calib -> TrustGate end-to-end example — the v5.44
# pipeline math wired into a runnable script: probe report in, calibrated
# gate decision record out, widen-only band visible in the numbers
# ----------------------------------------------------------------------
def test_quant_calib_trust_gate_example():
    from examples.quant_calib_trust_gate import run_gate_demo, run_probe_demo

    # Section 1: the real probe on the untrained lite model — mechanics
    # smoke only (the module's own honesty constraint), both arms present
    # and schema-complete.
    report = run_probe_demo(n_tasks=8, seed=0)
    for side in ("bf16", "nvfp4"):
        assert report[side]["n"] == 8
        assert 0.0 <= report[side]["ece"] <= 1.0
        assert 0.0 <= report[side]["accuracy"] <= 1.0
    assert "fakequant" in report["method"]

    # Section 2: deterministic band math on a hand-built report — the same
    # record the pipeline test (v5.44) locks, now through the example's
    # own wiring. p*=0.8, +0.03 drift widens the band to [0.72, 0.88].
    demo = {"bf16": {"tag": "x/bf16", "accuracy": 0.6, "ece": 0.05,
                     "n": 100},
            "nvfp4": {"tag": "x/nvfp4", "accuracy": 0.6, "ece": 0.08,
                      "n": 100}}
    rec = run_gate_demo(demo, p_trust=0.84)
    assert rec["calibrated"] and abs(rec["p_star"] - 0.8) < 1e-9
    assert abs(rec["margin"] - 0.08) < 1e-9, rec
    assert rec["route"] == "ESCALATE", rec
    # Far outside the widened band -> DIRECT (widening buys escalation,
    # never aggression).
    assert run_gate_demo(demo, p_trust=0.99)["route"] == "DIRECT"
    assert run_gate_demo(demo, p_trust=0.05)["route"] == "ESCALATE"

    # Loud: a report missing an arm must not produce a gate decision.
    try:
        run_gate_demo({"bf16": {"ece": 0.05, "n": 10}})
        assert False, "partial report must fail loudly"
    except ValueError:
        pass
    _pass("test_quant_calib_trust_gate_example",
          f"probe arms ok (drift {report['drift']['ece']:+.4f}, mechanics "
          f"only); demo band [0.72, 0.88] -> p=0.84 ESCALATE, p=0.99 "
          f"DIRECT; partial report loud")


# ----------------------------------------------------------------------
# v5.47 (1/3): decision_head noul-target regression — the 6d96cec "noul
# fix" duplicated the branches into an IndentationError that reached main
# undetected; the three-branch semantics (None -> 0.5, yes/no strings via
# _NOUL_IDX, raw floats) are locked here, plus the new empty-records
# loud guard
# ----------------------------------------------------------------------
def test_decision_head_noul_targets():
    _agent_dir = str(Path(__file__).resolve().parent.parent / "agent")
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from decision_head import DecisionHead

    head = DecisionHead({"t": ("noul", None), "s": ("score", None),
                         "c": ("choice", ("a", "b"))})

    def _rec(t_ans, s_ans, c_ans):
        return {"ids": torch.tensor([1, 2, 3]),
                "answers": {"t": t_ans, "s": s_ans, "c": c_ans}}

    records = [_rec("yes", 0.9, "b"), _rec("no", 0.1, "a"),
               _rec(None, 0.5, "b"), _rec(0.7, None, "a")]
    _, labels = head._targets(records)

    # noul: None fills 0.5, strings map via _NOUL_IDX, raw floats pass
    # through (the exact branches 6d96cec corrupted). Targets are float32
    # tensors, so compare at 1e-4 (float32(0.7) != 0.7 exactly).
    assert [round(v, 4) for v in labels["t"].tolist()] == \
        [1.0, 0.0, 0.5, 0.7], labels["t"]
    # score: raw floats; a missing answer fills 0.0 (documented policy).
    assert [round(v, 4) for v in labels["s"].tolist()] == \
        [0.9, 0.1, 0.5, 0.0], labels["s"]
    # choice: index into the option tuple.
    assert labels["c"].tolist() == [1, 0, 1, 0], labels["c"]

    # Loud guard: empty records fail at the boundary with the real cause
    # (previously a bare torch.stack runtime error).
    try:
        head._targets([])
        assert False, "empty records must fail loudly"
    except ValueError as e:
        assert "at least one record" in str(e), e
    _pass("test_decision_head_noul_targets",
          "noul None/str/float + score None-fill + choice idx locked; "
          "empty-records loud guard ok")


def test_decision_layer_audit():
    """v5.22: calibration metrics + routing monotonicity gate + tau evolution."""
    from helioslm_v5.eval.system_one import (MockSystemOne,
        ThresholdRouter, brier_score, evolve_threshold,
        expected_calibration_error, routing_gate)

    # 1) calibration metrics discriminate temperatures, ground truth only.
    #    The ECE surface is U-shaped in temperature (this mock is NOT
    #    calibrated at T=1: confidence exceeds accuracy) -- the audit
    #    kit's job is to FIND the optimum, the discipline a System One
    #    deployment should run in CI.
    x_base = MockSystemOne().sample_batch(2048, seed=1)

    def ece_of(t):
        m = MockSystemOne(temperature=t)
        r = m(x_base)
        return expected_calibration_error(r["probs"], r["correct"]), r

    e_sharp, r_sharp = ece_of(0.3)   # near-optimal for this mock
    e_mid, r_cal = ece_of(1.0)       # overconfident
    e_over, _ = ece_of(3.0)          # underconfident, worse
    assert e_sharp < e_mid < e_over, \
        f"U-shape expected: {e_sharp:.3f} < {e_mid:.3f} < {e_over:.3f}"
    assert e_sharp < 0.10
    assert brier_score(r_cal["probs"], r_cal["correct"]) < 0.25
    # grid search on the metric recovers the better temperature -- the
    # actionable output of a calibration audit
    grid = [(t, ece_of(t)[0]) for t in (0.2, 0.3, 0.5, 1.0, 2.0, 4.0)]
    t_best = min(grid, key=lambda p: p[1])[0]
    assert t_best <= 0.5, "grid search should prefer low temps, got " + str(t_best)

    # 2) routing monotonicity gate: correctness non-decreasing in tau
    cal = MockSystemOne(temperature=1.0)
    router = ThresholdRouter(oracle_cost=10.0, decision_cost=1.0)
    reports = routing_gate(cal, router, taus=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    assert all(r.gate_passed for r in reports), \
        "escalation path always correct => monotonicity must hold"
    cors = [r.correctness for r in reports]
    assert cors == sorted(cors), "correctness must be sorted by tau"
    costs = [r.cost for r in reports]
    assert costs == sorted(costs), "cost grows with tau"

    # 3) threshold evolution under the correctness floor
    ev = evolve_threshold(cal, router, correctness_floor=0.95,
                          n_samples=2048, seed=2)
    assert ev["best_tau"] is not None, "a feasible tau must exist"
    assert ev["correctness"] >= 0.95
    # the evolved tau must itself appear in the evaluated grid and pass gates
    row = next(r for r in ev["evaluated"] if r["tau"] == ev["best_tau"])
    assert row["gate"] and row["correctness"] >= 0.95
    _pass("test_decision_layer_audit",
          f"ECE t0.3={e_sharp:.3f} t1={e_mid:.3f} t3={e_over:.3f} "
          f"grid-best-t={t_best}; "
          f"tau*={ev['best_tau']} cost={ev['best_cost']:.2f} "
          f"correct={ev['correctness']:.3f}")
