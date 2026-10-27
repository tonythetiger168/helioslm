"""v5.42 — TrustGate v2 oracles (Phase 3.2).

P(yes) is injected by patching the head's forward — these oracles test the
GATE MATH (thresholds, bands, calibration tags), not the head. Head
quality is the head's own oracle suite.

Run from repo root: python helioslm_v5/tests/test_trust_gate_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from types import SimpleNamespace

from decision_head import DecisionHead
from gate import FixedGate, Route
from trust_gate_v2 import TrustGateV2


def _gate(p, cost_wrong=1.0, cost_escalate=0.2, calibration=None):
    head = DecisionHead({"trust": ("noul", None)})
    g = TrustGateV2(head, cost_wrong=cost_wrong,
                    cost_escalate=cost_escalate, calibration=calibration)
    g.router.head.forward = lambda ids: {"trust": (p, None)}
    call = SimpleNamespace(name="t.call")
    return g, call, {}


def test_expected_loss_threshold():
    cal = {"ece": 0.01, "n": 100}
    g, call, ctx = _gate(0.95, cost_escalate=0.2, calibration=cal)
    assert abs(g.threshold - 0.8) < 1e-9
    assert g.decide(call, ctx) == Route.DIRECT
    g, call, ctx = _gate(0.10, cost_escalate=0.2, calibration=cal)
    assert g.decide(call, ctx) == Route.ESCALATE
    rec = g.decide_explain(call, ctx)
    assert abs(rec["expected_loss"]["direct"]
               - 0.9) < 1e-6 and abs(rec["expected_loss"]["escalate"]
                                     - 0.2) < 1e-6
    print("PASS test_expected_loss_threshold p*=0.8, direct/escalate split")


def test_calibration_band_blocks_sharp_calls():
    cal = {"ece": 0.05, "n": 500}
    # p=0.82: above p*=0.8 by 0.02 — sharp calibration would DIRECT,
    # ece=0.05 widens the band to [0.75, 0.85] and the gate abstains.
    g, call, ctx = _gate(0.82, calibration=cal)
    rec = g.decide_explain(call, ctx)
    assert rec["route"] == "ESCALATE", rec
    assert rec["calibrated"] is True and rec["margin"] == 0.05
    # same p with near-perfect calibration: outside band -> DIRECT
    g2, call2, ctx2 = _gate(0.82, calibration={"ece": 0.005, "n": 500})
    assert g2.decide(call2, ctx2) == Route.DIRECT
    print("PASS test_calibration_band_blocks_sharp_calls "
          "ece .05 abstains, ece .005 acts at same p")


def test_uncalibrated_loud_default():
    g, call, ctx = _gate(0.85)
    rec = g.decide_explain(call, ctx)
    assert rec["calibrated"] is False
    assert rec["margin"] == 0.25
    # and the uncertainty is recorded in the log for auditing
    assert g.log[-1]["calibrated"] is False
    print("PASS test_uncalibrated_loud_default margin 0.25 + log tag")


def test_costs_shift_threshold():
    cal = {"ece": 0.01, "n": 100}
    p = 0.85
    g_cheap_esc, _, _ = _gate(p, cost_escalate=0.1, calibration=cal)  # p*=.9
    g_pricey_esc, _, _ = _gate(p, cost_escalate=0.4, calibration=cal)  # p*=.6
    assert g_cheap_esc.decide(SimpleNamespace(name="c"), {}) == Route.ESCALATE
    assert g_pricey_esc.decide(SimpleNamespace(name="c"),
                               {}) == Route.DIRECT
    print("PASS test_costs_shift_threshold same p=0.85: escalate when "
          "escalation cheap, direct when expensive")


def test_loud_errors_and_escalate_contract():
    head = DecisionHead({"trust": ("noul", None)})
    for bad in (dict(cost_escalate=1.0),            # >= cost_wrong
                dict(cost_escalate=-0.1),
                dict(cost_wrong=0.0),
                dict(calibration={"ece": 0.9, "n": 10})):  # ece out of range
        try:
            TrustGateV2(head, **bad)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"bad config accepted: {bad}")
    g, call, ctx = _gate(0.1)
    try:
        g.escalate(call, ctx)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("escalate without inner must fail loudly")
    g3 = TrustGateV2(DecisionHead({"trust": ("noul", None)}),
                      inner=FixedGate(Route.ESCALATE))
    g3.router.head.forward = lambda ids: {"trust": (0.1, None)}
    assert g3.decide(call, ctx) == Route.ESCALATE
    print("PASS test_loud_errors_and_escalate_contract bad costs/ece "
          "raise; abstain without inner raises; inner fallback works")


if __name__ == "__main__":
    test_expected_loss_threshold()
    test_calibration_band_blocks_sharp_calls()
    test_uncalibrated_loud_default()
    test_costs_shift_threshold()
    test_loud_errors_and_escalate_contract()
    print("\n5/5 TrustGate v2 tests passed")
