"""T87 — M4: handover-resume faithfulness and goal-drift oracles.

Pins the v5.54 behavior contract:

1. a handoff is accepted only for a bitwise-replayed strict oracle-plan
   prefix with matching goal/state/trajectory hashes;
2. a fresh AgentLoop resumes from the rendered packet, executes exactly the
   remaining plan, repeats zero completed work, and reaches the oracle final;
3. stale sandbox state, goal tampering, and truncated plans abort loudly;
4. a restarting policy is measured as goal drift and receives no credit;
5. the committed constructive artifact regenerates byte-for-byte and is
   registered as oracle-tested.

Run from repo root: python3 helioslm_v5/tests/test_handover_resume.py
"""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.agent.envs.file_env import scripted_file_policy  # noqa: E402
from helioslm_v5.agent.gate import FixedGate, Route  # noqa: E402
from helioslm_v5.agent.handover import (  # noqa: E402
    SCHEMA_ID, HandoffPacket, HandoverError, build_handoff,
    canonical_sha256, resume_metrics, run_handoff_resume, state_sha256,
    validate_handoff)
from helioslm_v5.agent.loop import AgentLoop  # noqa: E402
from helioslm_v5.agent.schema import ToolCall, render_tool_call  # noqa: E402
from helioslm_v5.agent.tools import build_default_registry  # noqa: E402
from helioslm_v5.agent.trajectory import (  # noqa: E402
    ReplayMismatch, verify_replay)
from helioslm_v5.eval.artifact_tiers import (  # noqa: E402
    load_registry, tier_of)
from helioslm_v5.eval.handover_resume import (  # noqa: E402
    _scenarios, build_report)

_ARTIFACT = _REPO / "benchmarks" / "handover_resume_2026-10-14.json"


def _prefix_packet(spec, root):
    registry, impls = build_default_registry(root)
    loop = AgentLoop(scripted_file_policy, registry, impls,
                     FixedGate(Route.DIRECT), max_steps=spec["prefix"])
    traj = loop.run(spec["task"], seed=17)
    assert len(traj.steps) == spec["prefix"]
    assert traj.final_answer is None
    packet = build_handoff(traj, spec["plan"], registry, impls, root)
    return packet, registry, impls, traj


def test_packet_build_validate_roundtrip():
    spec = _scenarios()[0]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, traj = _prefix_packet(spec, root)
        assert packet.goal_sha256 == packet.to_dict()["goal_sha256"]
        assert packet.state_sha256 == state_sha256(root)
        assert packet.trajectory_sha256 == canonical_sha256(traj.to_dict())
        assert len(packet.completed_calls) == spec["prefix"]
        assert len(packet.remaining_calls) == (
            len(spec["plan"]) - spec["prefix"])

        restored = HandoffPacket.from_dict(packet.to_dict())
        assert restored == packet
        assert restored.sha256() == packet.sha256()
        validate_handoff(packet, root=root)

        bad_plan = list(spec["plan"])
        bad_plan[0] = ToolCall("calc", {"expr": "1+1"})
        try:
            build_handoff(traj, bad_plan, *build_default_registry(root), root)
        except (HandoverError, ReplayMismatch):
            pass
        else:
            raise AssertionError("wrong oracle plan must not hand over")
    print("PASS test_packet_build_validate_roundtrip")


def test_fresh_resume_executes_suffix_without_rework():
    spec = _scenarios()[1]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _prefix_packet(spec, root)
        registry, impls = build_default_registry(root)  # fresh session objects
        resumed = run_handoff_resume(
            packet, scripted_file_policy, registry, impls,
            FixedGate(Route.DIRECT), root, max_steps=4)
        verify_replay(resumed, registry, impls)
        metrics = resume_metrics(packet, resumed, spec["expected"])
        assert metrics["faithful"]
        assert metrics["correct_final"]
        assert metrics["prefix_repeats"] == 0
        assert metrics["goal_drift_events"] == 0
        assert len(resumed.steps) == len(spec["plan"]) - spec["prefix"]
        assert resumed.final_answer == spec["expected"]
    print("PASS test_fresh_resume_executes_suffix_without_rework")


def test_stale_state_goal_and_truncation_abort():
    spec = _scenarios()[0]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _prefix_packet(spec, root)
        (root / "f.txt").write_text("stale", encoding="utf-8")
        registry, impls = build_default_registry(root)
        try:
            run_handoff_resume(packet, scripted_file_policy, registry, impls,
                               FixedGate(Route.DIRECT), root)
        except HandoverError as e:
            assert "state differs" in str(e)
        else:
            raise AssertionError("stale sandbox state resumed silently")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _prefix_packet(spec, root)
        try:
            validate_handoff(replace(packet, task=packet.task + " drift"),
                             root=root)
        except HandoverError as e:
            assert "goal hash" in str(e)
        else:
            raise AssertionError("goal-tampered packet validated")
        try:
            validate_handoff(replace(packet, remaining_calls=()), root=root)
        except HandoverError as e:
            assert "no remaining plan" in str(e)
        else:
            raise AssertionError("truncated handoff validated")
    print("PASS test_stale_state_goal_and_truncation_abort")


def test_restart_policy_records_goal_drift():
    spec = _scenarios()[0]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _prefix_packet(spec, root)
        first = json.loads(packet.completed_calls[0])

        def restarting_policy(prompt, seed, step):
            return render_tool_call([ToolCall(first["name"], first["args"])])

        registry, impls = build_default_registry(root)
        resumed = run_handoff_resume(packet, restarting_policy, registry,
                                     impls, FixedGate(Route.DIRECT), root,
                                     max_steps=1)
        metrics = resume_metrics(packet, resumed, spec["expected"])
        assert not metrics["faithful"]
        assert metrics["prefix_repeats"] == 1
        assert metrics["plan_mismatches"] == 2  # wrong call + missing finish
        assert metrics["wrong_final"] == 1
        assert metrics["goal_drift_events"] == 4
    print("PASS test_restart_policy_records_goal_drift")


def test_committed_artifact_regenerates_and_registry():
    artifact = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema"] == SCHEMA_ID
    assert artifact["verification_tier"] == "oracle-tested"
    assert artifact["results"]["summary"] == {
        "handover_tasks": 3,
        "faithful_resumes": 3,
        "correct_finals": 3,
        "prefix_repeated_steps": 0,
        "goal_drift_events": 0,
        "tamper_controls": 4,
        "tamper_controls_caught": 4,
    }
    signed = {k: v for k, v in artifact.items() if k != "results_sha256"}
    assert canonical_sha256(signed) == artifact["results_sha256"]
    assert build_report() == artifact, "constructive artifact must regenerate"

    registry = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_ARTIFACT.name, registry) == "oracle-tested"
    print("PASS test_committed_artifact_regenerates_and_registry")


def main():
    test_packet_build_validate_roundtrip()
    test_fresh_resume_executes_suffix_without_rework()
    test_stale_state_goal_and_truncation_abort()
    test_restart_policy_records_goal_drift()
    test_committed_artifact_regenerates_and_registry()
    print("T87 PASS: 5/5")


if __name__ == "__main__":
    main()
