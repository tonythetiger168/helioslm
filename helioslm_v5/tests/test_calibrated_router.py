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
    "certainty": ("score", None),
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
                                     "certainty": 1.0}})
        else:
            t = f"RISKY delete wipe op {i}"
            recs.append({"ids": torch.tensor(text_to_ids(t + " [calc]")),
                         "answers": {"route": "ESCALATE",
                                     "is_irreversible": "yes",
                                     "off_task": "yes",
                                     "mutates": "yes",
                                     "out_of_scope": "yes",
                                     "certainty": 0.0}})
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


def test_certainty_escapes_conf_floor():
    """v5.32.1 finding made actionable: route Choice confidence floors at
    0.5 on the risky class; the certainty Score drops below it."""
    router = CalibratedRouter(_mk_head())
    call = ToolCall("calc", {"expr": "1"})
    _, route_conf = router.head.decide(
        "route", torch.tensor(text_to_ids("RISKY delete wipe op 3 [calc]")))
    cert, _ = router.certainty(call, {"task": "RISKY delete wipe op 3"})
    assert route_conf >= 0.5, "sanity: Choice floor"
    assert cert < 0.5, f"certainty should escape the floor: {cert}"
    clean_cert, _ = router.certainty(call, {"task": "simple read only task 3"})
    assert clean_cert > 0.5
    print(f"PASS test_certainty_escapes_conf_floor "
          f"(route_conf={route_conf:.2f} risky_cert={cert:.2f} "
          f"clean_cert={clean_cert:.2f})")


if __name__ == "__main__":
    test_head_serves_gate_decisions()
    test_pi_warden_one_forward()
    test_escalate_and_unregistered_loud()
    test_certainty_escapes_conf_floor()
    print("\n4/4 calibrated-router tests passed")
