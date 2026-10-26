"""T86 — M3: TrustGate CLARIFY three-state oracles.

Pins the v5.53 behavior contract:

1. missing required parameters route to CLARIFY and never execute;
2. completing the parameter restores the wrapped gate's DIRECT decision;
3. complete-but-distrusted and schema-invalid calls remain ESCALATE;
4. AgentLoop/ChatSession only expose partial calls to the gate in the
   explicit defer-validation mode, and CLARIFY is terminal for that turn;
5. the committed constructive artifact regenerates byte-for-byte and is
   registered as oracle-tested.

Run from repo root: python3 helioslm_v5/tests/test_trust_gate_clarify.py
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.agent.chat import ChatSession  # noqa: E402
from helioslm_v5.agent.gate import FixedGate, OracleGate, Route  # noqa: E402
from helioslm_v5.agent.loop import AgentLoop  # noqa: E402
from helioslm_v5.agent.schema import ToolCall, render_tool_call  # noqa: E402
from helioslm_v5.agent.tools import build_default_registry  # noqa: E402
from helioslm_v5.agent.trust_gate_clarify import (  # noqa: E402
    SCHEMA_ID, ClarifySchemaError, TrustGateClarify, inspect_call)
from helioslm_v5.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from helioslm_v5.eval.clarify_missing_param import build_report  # noqa: E402
from helioslm_v5.eval.cost_per_task import canonical_sha256  # noqa: E402

_ARTIFACT = _REPO / "benchmarks" / "clarify_missing_param_2026-10-13.json"


def _scripted(call):
    text = render_tool_call([call])

    def model_fn(prompt, seed, step):
        return text

    return model_fn


def test_missing_params_route_to_clarify_not_execute():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        registry, impls = build_default_registry(root)
        gate = TrustGateClarify(FixedGate(Route.DIRECT), registry)
        call = ToolCall("file_write", {"path": "note.txt"})
        info = inspect_call(call, registry)
        assert info["status"] == "missing" and info["missing"] == ["content"]
        assert gate.decide(call, {"task": "save it"}) == Route.CLARIFY
        question = gate.clarify(call, {})
        assert question.startswith("CLARIFY:") and "content: str" in question
        assert not (root / "note.txt").exists(), "CLARIFY must not execute"
        assert gate.log[-1]["route"] == "CLARIFY"
    print("PASS test_missing_params_route_to_clarify_not_execute")


def test_complete_and_low_trust_controls_delegate():
    with tempfile.TemporaryDirectory() as td:
        registry, impls = build_default_registry(td)
        complete = ToolCall("str_op", {"s": "ab", "op": "repeat", "n": 2})
        direct = TrustGateClarify(FixedGate(Route.DIRECT), registry)
        assert direct.decide(complete, {}) == Route.DIRECT

        oracle = OracleGate(lambda call, ctx: "ORACLE_REVIEW")
        escalate = TrustGateClarify(oracle, registry)
        rec = escalate.decide_explain(complete, {})
        assert rec["status"] == "complete" and rec["route"] == "ESCALATE"
        assert escalate.escalate(complete, {}) == "ORACLE_REVIEW"
    print("PASS test_complete_and_low_trust_controls_delegate")


def test_invalid_schema_escalates_and_clarify_is_loud():
    with tempfile.TemporaryDirectory() as td:
        registry, _ = build_default_registry(td)
        gate = TrustGateClarify(OracleGate(lambda c, x: "REVIEW"), registry)
        extra = ToolCall("str_op", {"s": "ab", "op": "repeat", "n": 2,
                                    "bogus": "x"})
        wrong_type = ToolCall("str_op", {"s": "ab", "op": "repeat", "n": "2"})
        assert gate.decide(extra, {}) == Route.ESCALATE
        assert gate.decide(wrong_type, {}) == Route.ESCALATE
        try:
            gate.clarify(extra, {})
        except ClarifySchemaError:
            pass
        else:
            raise AssertionError("clarify accepted an invalid-schema call")
    print("PASS test_invalid_schema_escalates_and_clarify_is_loud")


def test_deferred_loop_and_chat_make_clarify_terminal():
    with tempfile.TemporaryDirectory() as td:
        registry, impls = build_default_registry(td)
        call = ToolCall("str_op", {"s": "ab", "op": "repeat"})

        # Historical compatibility: without defer_validation this remains a
        # parser error, preserving older evals' semantics.
        old_gate = TrustGateClarify(FixedGate(Route.DIRECT), registry)
        old = AgentLoop(_scripted(call), registry, impls, old_gate,
                        max_steps=2, defer_validation=False)
        old_traj = old.run("repeat ab", seed=0)
        assert old_traj.steps[0].parse_error is not None
        assert old_traj.steps[0].route == "-"

        gate = TrustGateClarify(FixedGate(Route.DIRECT), registry)
        loop = AgentLoop(_scripted(call), registry, impls, gate,
                         max_steps=2, defer_validation=True)
        traj = loop.run("repeat ab", seed=0)
        assert len(traj.steps) == 1
        assert traj.steps[0].route == "CLARIFY"
        assert traj.steps[0].observation.startswith("CLARIFY:")
        assert traj.final_answer is None

        chat_gate = TrustGateClarify(FixedGate(Route.DIRECT), registry)
        chat = ChatSession(_scripted(call), registry, impls, chat_gate,
                           max_steps=2, defer_validation=True)
        reply = chat.send("repeat ab", seed=0)
        assert reply.startswith("CLARIFY:") and "n: int" in reply
        assert chat.steps[-1].route == "CLARIFY"
    print("PASS test_deferred_loop_and_chat_make_clarify_terminal")


def test_committed_artifact_regenerates_and_registry():
    artifact = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema"] == SCHEMA_ID
    assert artifact["verification_tier"] == "oracle-tested"
    summary = artifact["results"]["summary"]
    assert summary == {
        "missing_tasks": 6,
        "clarified": 6,
        "clarify_no_final_answer": 6,
        "complete_controls": 6,
        "direct_after_completion": 6,
        "low_trust_controls": 1,
        "escalated_low_trust": 1,
        "invalid_schema_controls": 1,
        "escalated_invalid_schema": 1,
    }
    signed = {k: v for k, v in artifact.items() if k != "results_sha256"}
    assert canonical_sha256(signed) == artifact["results_sha256"]
    assert build_report() == artifact, "constructive artifact must regenerate"

    registry = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_ARTIFACT.name, registry) == "oracle-tested"
    print("PASS test_committed_artifact_regenerates_and_registry")


def main():
    test_missing_params_route_to_clarify_not_execute()
    test_complete_and_low_trust_controls_delegate()
    test_invalid_schema_escalates_and_clarify_is_loud()
    test_deferred_loop_and_chat_make_clarify_terminal()
    test_committed_artifact_regenerates_and_registry()
    print("T86 PASS: 5/5")


if __name__ == "__main__":
    main()
