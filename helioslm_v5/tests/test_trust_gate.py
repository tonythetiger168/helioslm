"""T38 - v5.37: TrustGate oracles (calibrated abstention).

Covers: tier behavior (high/medium/low -> DIRECT/DIRECT/ESCALATE),
abstention discipline (no oracle -> loud NotImplementedError, with
oracle -> delegated), the log as audit evidence, and the end-to-end
doctrine: the model is DENIED action on states the head distrusts --
the therapy-side counterpart of grounding (which fixes actions the
model does take).
Run from repo root: python3 helioslm_v5/tests/test_trust_gate.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision_head import DecisionHead
from gate import FixedGate, Route
from schema import ToolCall
from trajectory import text_to_ids
from trust_gate import TrustGate


def _head(seed=3):
    torch.manual_seed(seed)
    head = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE")),
                         "trust": ("noul", None)}, brier_lambda=2.0)
    recs = []
    for i in range(80):
        ok = i % 2 == 0
        marker = "CALM" if ok else "HAZARD"
        recs.append({"ids": torch.tensor(text_to_ids(f"{marker} job {i}")),
                     "answers": {"route": "DIRECT" if ok else "ESCALATE",
                                 "trust": "yes" if ok else "no"}})
    head.fit(recs, epochs=800, lr=1e-1)
    head.eval()
    return head


def test_tiers_route_by_trust():
    g = TrustGate(_head(), hi=0.7, lo=0.3)
    call = ToolCall("calc", {"expr": "1"})
    p_calm = g._trust(call, {"task": "CALM job 3"})
    p_haz = g._trust(call, {"task": "HAZARD job 3"})
    assert p_calm > p_haz, (p_calm, p_haz)
    r_calm = g.decide(call, {"task": "CALM job 3"})
    r_haz = g.decide(call, {"task": "HAZARD job 3"})
    assert r_calm == Route.DIRECT and r_haz == Route.ESCALATE
    tiers = [e["tier"] for e in g.log]
    assert "high" in tiers or "medium" in tiers, tiers
    assert "low" in tiers, tiers
    print(f"PASS test_tiers_route_by_trust "
          f"(P(calm)={p_calm:.2f} -> DIRECT, P(hazard)={p_haz:.2f} -> ESCALATE)")


def test_low_trust_abstains_loud_without_oracle():
    from contextlib import contextmanager

    @contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__}")

    g = TrustGate(_head(), lo=0.3)
    call = ToolCall("calc", {"expr": "1"})
    assert g.decide(call, {"task": "HAZARD job 3"}) == Route.ESCALATE
    with raises(NotImplementedError):
        g.escalate(call, {"task": "HAZARD job 3"})
    print("PASS test_low_trust_abstains_loud_without_oracle")


def test_abstain_delegates_to_oracle():
    g = TrustGate(_head(), lo=0.3, inner=FixedGate(Route.ESCALATE))
    # FixedGate.escalate raises NotImplementedError by design; use a
    # recording oracle instead
    class Oracle:
        def escalate(self, call, context):
            return "ORACLE_REVIEW"
    g.inner = Oracle()
    call = ToolCall("calc", {"expr": "1"})
    assert g.decide(call, {"task": "HAZARD job 3"}) == Route.ESCALATE
    assert g.escalate(call, {"task": "HAZARD job 3"}) == "ORACLE_REVIEW"
    print("PASS test_abstain_delegates_to_oracle")


def test_log_is_audit_evidence():
    g = TrustGate(_head(), lo=0.3)
    call = ToolCall("calc", {"expr": "1"})
    g.decide(call, {"task": "CALM job 1"})
    g.decide(call, {"task": "HAZARD job 1"})
    assert len(g.log) == 2
    assert all("p_trust" in e and "tier" in e and "route" in e for e in g.log)
    print("PASS test_log_is_audit_evidence")


if __name__ == "__main__":
    test_tiers_route_by_trust()
    test_low_trust_abstains_loud_without_oracle()
    test_abstain_delegates_to_oracle()
    test_log_is_audit_evidence()
    print("\n4/4 trust-gate tests passed")
