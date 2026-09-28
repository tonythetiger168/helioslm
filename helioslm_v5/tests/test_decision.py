"""T27 - v5.32: typed decision primitives + gate.ask oracles.

Covers: schema enforcement (typed output space makes invalid answers
impossible), gate.ask typed answers, pi-warden batch (serial default =
ask_batch override point), and the additive contract (T11-T19 gate
behavior unchanged).
Run from repo root: python3 helioslm_v5/tests/test_decision.py
"""
import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision import (Choice, DecisionError, Noul, PI_WARDEN_QUESTIONS,
                      Score, choice_from_route, pi_warden_batch)
from envs import make_envs
from gate import FixedGate, Gate, Route
from schema import ToolCall, ToolCallError
from tools import build_default_registry


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def reg_impls():
    return build_default_registry("/tmp/helioslm_decision_test")


def test_primitives_schema_enforced():
    c = Choice("route", ("DIRECT", "ESCALATE"), "DIRECT", 0.9)
    assert c.answer == "DIRECT" and c.confidence == 0.9
    with raises(DecisionError):
        Choice("route", ("DIRECT", "ESCALATE"), "MAYBE", 0.9)
    with raises(DecisionError):
        Choice("route", ("DIRECT",), "DIRECT", 1.5)
    with raises(DecisionError):
        Score("sev", 1.2, 0.5)
    with raises(DecisionError):
        Noul("q", "perhaps", 0.5)
    with raises(DecisionError):
        Noul("q", "yes", -0.1)
    print("PASS test_primitives_schema_enforced")


def test_choice_from_route():
    ch = choice_from_route(Route.ESCALATE, 0.31)
    assert ch.answer == "ESCALATE" and abs(ch.confidence - 0.31) < 1e-9
    ch = choice_from_route(Route.DIRECT, None)
    assert ch.answer == "DIRECT" and ch.confidence == 0.5  # default recorded
    print("PASS test_choice_from_route")


def test_gate_ask_route_and_noul():
    reg, _ = reg_impls()
    call = ToolCall("calc", {"expr": "1/0"})
    g_direct = FixedGate(Route.DIRECT)
    g_esc = FixedGate(Route.ESCALATE)
    ctx = {"confidence": 0.77}
    d = g_direct.ask(call, ctx, "choice", "route")
    assert isinstance(d, Choice) and d.answer == "DIRECT" \
        and d.confidence == 0.77
    n = g_esc.ask(call, ctx, "noul", "is_irreversible")
    assert isinstance(n, Noul) and n.answer == "yes"   # ESCALATE -> scrutiny
    n2 = g_direct.ask(call, ctx, "noul", "is_irreversible")
    assert n2.answer == "no"
    s = g_direct.ask(call, ctx, "score", "severity")
    assert isinstance(s, Score) and s.value == 0.5
    with raises(NotImplementedError):
        g_direct.ask(call, ctx, "oracle", "x")
    print("PASS test_gate_ask_route_and_noul")


def test_pi_warden_batch_serial_default():
    reg, _ = reg_impls()
    call = ToolCall("calc", {"expr": "1/0"})
    g = FixedGate(Route.ESCALATE)
    out = pi_warden_batch(call, {"confidence": 0.6}, g)
    assert set(out) == {n for n, _, _ in PI_WARDEN_QUESTIONS}
    assert out["route"].answer == "ESCALATE"
    assert out["is_irreversible"].answer == "yes"
    # ask_batch serial == per-question ask (override point is behavior-
    # preserving by contract)
    one_by_one = {n: g.ask(call, {"confidence": 0.6}, k, t)
                  for n, k, t in PI_WARDEN_QUESTIONS}
    assert out == one_by_one
    print("PASS test_pi_warden_batch_serial_default")


def test_additive_contract_gate_unchanged():
    # T11-T19 contract: decide/escalate behave exactly as before
    g = FixedGate(Route.ESCALATE)
    with raises(NotImplementedError):
        g.escalate(ToolCall("calc", {"expr": "1"}), {})
    tg_call = ToolCall("calc", {"expr": "1"})
    from gate import ThresholdGate
    tg = ThresholdGate(0.5, oracle=lambda c, x: "x")
    assert tg.decide(tg_call, {"confidence": 0.7}) == Route.DIRECT
    assert tg.decide(tg_call, {"confidence": 0.3}) == Route.ESCALATE
    with raises(ToolCallError):
        tg.decide(tg_call, {})
    print("PASS test_additive_contract_gate_unchanged")


if __name__ == "__main__":
    test_primitives_schema_enforced()
    test_choice_from_route()
    test_gate_ask_route_and_noul()
    test_pi_warden_batch_serial_default()
    test_additive_contract_gate_unchanged()
    print("\n5/5 decision tests passed")
