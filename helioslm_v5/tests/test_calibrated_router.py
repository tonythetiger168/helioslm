"""T30 - v5.32 P3: CalibratedRouter oracles.

Covers: head-backed gate decisions, pi-warden preset answered in ONE
forward pass (counted, not assumed), escalate discipline (never
fabricates oracles), unregistered-question loud failure, and the
sub-floor certainty escape hatch demonstrating the v5.32.1 floor finding.
Run from repo root: python3 helioslm_v5/tests/test_calibrated_router.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from calibrated_router import CalibratedRouter
from decision import (Choice, Noul, PI_WARDEN_QUESTIONS, Score)
from decision_head import DecisionHead
from gate import Route
from schema import ToolCall
from trajectory import text_to_ids

QUESTIONS = {
    "route": ("choice", ("DIRECT", "ESCALATE")),
    "is_irreversible": ("noul", None),
    "off_task": ("noul", None),
    "mutates": ("noul", None),
    "out_of_scope": ("noul", None),
    "trust": ("noul", None),
}


def _records(n=160, seed=7):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        if i % 2 == 0:
            t = f"simple read only task {i}"
            recs.append({"ids": torch.tensor(text_to_ids(t + " [calc]")),
                         "answers": {"route": "DIRECT",
                                     "is_irreversible": "no",
                                     "off_task": "no",
                                     "mutates": "no",
                                     "out_of_scope": "no",
                                     "trust": "no"}})
        else:
            t = f"RISKY delete wipe op {i}"
            recs.append({"ids": torch.tensor(text_to_ids(t + " [calc]")),
                         "answers": {"route": "ESCALATE",
                                     "is_irreversible": "yes",
                                     "off_task": "yes",
                                     "mutates": "yes",
                                     "out_of_scope": "yes",
                                     "trust": "yes"}})
    return recs


def _mk_head():
    torch.manual_seed(3)
    head = DecisionHead(QUESTIONS, brier_lambda=2.0)
    head.fit(_records(), epochs=400, lr=5e-2)
    head.eval()
    return head


def test_head_serves_gate_decisions():
    router = CalibratedRouter(_mk_head())
    call = ToolCall("calc", {"expr": "1"})
    clean = router.decide(call, {"task": "simple read only task 9"})
    risky = router.decide(call, {"task": "RISKY delete wipe op 9"})
    assert clean == Route.DIRECT and risky == Route.ESCALATE, (clean, risky)
    print("PASS test_head_serves_gate_decisions")


def test_pi_warden_one_forward():
    head = _mk_head()
    forwards = {"n": 0}
    orig = head.forward

    def counted(ids):
        forwards["n"] += 1
        return orig(ids)

    head.forward = counted
    router = CalibratedRouter(head)
    call = ToolCall("calc", {"expr": "1"})
    out = router.ask_batch(call, {"task": "RISKY delete wipe op 5"},
                           PI_WARDEN_QUESTIONS)
    assert forwards["n"] == 1, f"expected ONE forward, got {forwards['n']}"
    assert isinstance(out["route"], Choice)
    assert isinstance(out["is_irreversible"], Noul)
    assert out["route"].answer == "ESCALATE"
    assert out["is_irreversible"].answer == "yes"
    print("PASS test_pi_warden_one_forward")


def test_escalate_and_unregistered_loud():
    from contextlib import contextmanager

    @contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__}")

    router = CalibratedRouter(_mk_head())
    call = ToolCall("calc", {"expr": "1"})
    with raises(NotImplementedError):
        router.escalate(call, {})
    with raises(KeyError):
        router.ask_batch(call, {"task": "x"},
                         [("never_trained", "noul", "q?")])
    print("PASS test_escalate_and_unregistered_loud")


def test_continuous_uncertainty_readout():
    """v5.36: continuous-Noul uncertainty() readout replaces the retired
    certainty-Score escape hatch."""
    router = CalibratedRouter(_mk_head())
    call = ToolCall("calc", {"expr": "1"})
    _, route_conf = router.head.decide(
        "route", torch.tensor(text_to_ids("RISKY delete wipe op 3 [calc]")))
    p_risky = router.uncertainty(call, {"task": "RISKY delete wipe op 3"}, "trust")
    p_clean = router.uncertainty(call, {"task": "simple read only task 3"}, "trust")
    assert route_conf >= 0.5 and p_risky < 0.5 < p_clean, (route_conf, p_risky, p_clean)
    print(f"PASS test_continuous_uncertainty_readout "
          f"(route floor {route_conf:.2f}; trust P(yes) risky {p_risky:.2f} "
          f"clean {p_clean:.2f})")

if __name__ == "__main__":
    test_head_serves_gate_decisions()
    test_pi_warden_one_forward()
    test_escalate_and_unregistered_loud()
    test_continuous_uncertainty_readout()
    print("\n4/4 calibrated-router tests passed")
