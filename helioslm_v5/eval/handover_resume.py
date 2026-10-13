"""M4 constructive oracle for handover-resume faithfulness (v5.54).

The question is not "does a summary sound plausible?"; it is whether a fresh
session can resume from a verifiable handoff without repeating completed
work, drifting off the remaining plan, or changing the final answer.

Three long-horizon file tasks are interrupted at strict oracle-plan prefixes.
The resulting handoff packet carries the goal hash, exact completed calls,
observations, remaining calls, source trajectory hash, and sandbox state
hash. A fresh AgentLoop receives the rendered packet and must complete the
remaining suffix exactly. Stale state, goal tampering, and an intentionally
restarting policy are negative controls that must be caught or receive zero
faithfulness credit.
"""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from helioslm_v5.agent import task_grammar as _G
from helioslm_v5.agent.envs.file_env import scripted_file_policy
from helioslm_v5.agent.gate import FixedGate, Route
from helioslm_v5.agent.handover import (SCHEMA_ID, HandoverError,
                                        build_handoff, canonical_sha256,
                                        resume_metrics, run_handoff_resume,
                                        state_snapshot, validate_handoff)
from helioslm_v5.agent.loop import AgentLoop
from helioslm_v5.agent.schema import ToolCall, render_tool_call
from helioslm_v5.agent.tools import build_default_registry
from helioslm_v5.agent.trajectory import verify_replay

DATE = "2026-10-14"


class HandoverEvalError(RuntimeError):
    """Any oracle mismatch in the handover-resume suite. Never swallowed."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise HandoverEvalError(message)


def _scenarios() -> list:
    write_read_task = _G.render(
        "write_read", expr="18 * 3 + 2", path="f.txt")
    write_transform_task = _G.render(
        "write_transform", s="apple bridge", a="a.txt", op="reverse",
        n=0, b="b.txt")
    accumulate_task = _G.render(
        "accumulate", e1="7 * 8", p1="a.txt", e2="5 + 9", p2="b.txt")
    return [
        {
            "id": "write-read-interrupt-after-write",
            "family": "write_read",
            "task": write_read_task,
            "prefix": 2,
            "expected": "56",
            "plan": [
                ToolCall("calc", {"expr": "18 * 3 + 2"}),
                ToolCall("file_write", {"path": "f.txt", "content": "56"}),
                ToolCall("file_read", {"path": "f.txt"}),
                ToolCall("finish", {"answer": "56"}),
            ],
        },
        {
            "id": "write-transform-interrupt-after-transform",
            "family": "write_transform",
            "task": write_transform_task,
            "prefix": 2,
            "expected": "egdirb elppa",
            "plan": [
                ToolCall("file_write", {"path": "a.txt",
                                        "content": "apple bridge"}),
                ToolCall("str_op", {"s": "apple bridge", "op": "reverse",
                                    "n": 0}),
                ToolCall("file_write", {"path": "b.txt",
                                        "content": "egdirb elppa"}),
                ToolCall("file_read", {"path": "b.txt"}),
                ToolCall("finish", {"answer": "egdirb elppa"}),
            ],
        },
        {
            "id": "accumulate-interrupt-after-both-writes",
            "family": "accumulate",
            "task": accumulate_task,
            "prefix": 4,
            "expected": "70",
            "plan": [
                ToolCall("calc", {"expr": "7 * 8"}),
                ToolCall("calc", {"expr": "5 + 9"}),
                ToolCall("file_write", {"path": "a.txt", "content": "56"}),
                ToolCall("file_write", {"path": "b.txt", "content": "14"}),
                ToolCall("file_read", {"path": "a.txt"}),
                ToolCall("file_read", {"path": "b.txt"}),
                ToolCall("calc", {"expr": "(56) + (14)"}),
                ToolCall("finish", {"answer": "70"}),
            ],
        },
    ]


def _run_prefix(spec: dict, root: Path):
    registry, impls = build_default_registry(root)
    loop = AgentLoop(scripted_file_policy, registry, impls,
                     FixedGate(Route.DIRECT), max_steps=spec["prefix"])
    traj = loop.run(spec["task"], seed=17)
    _require(len(traj.steps) == spec["prefix"],
             f"{spec['id']}: prefix length {len(traj.steps)}")
    _require(traj.final_answer is None,
             f"{spec['id']}: prefix unexpectedly finished")
    packet = build_handoff(traj, spec["plan"], registry, impls, root)
    return packet, registry, impls, traj


def _run_faithful(spec: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, prefix_traj = _run_prefix(spec, root)
        # Fresh registry/impls/loop: context resets; sandbox state persists.
        registry, impls = build_default_registry(root)
        resumed = run_handoff_resume(
            packet, scripted_file_policy, registry, impls,
            FixedGate(Route.DIRECT), root,
            max_steps=len(spec["plan"]) - spec["prefix"] + 1)
        verify_replay(resumed, registry, impls)
        metrics = resume_metrics(packet, resumed, spec["expected"])
        _require(metrics["faithful"],
                 f"{spec['id']}: fresh resume was not faithful: {metrics}")
        _require(len(resumed.steps) == len(spec["plan"]) - spec["prefix"],
                 f"{spec['id']}: resume did not execute exactly the suffix")
        return {
            "id": spec["id"],
            "family": spec["family"],
            "prefix_steps": len(prefix_traj.steps),
            "resume_steps": len(resumed.steps),
            "state_files": state_snapshot(root),
            "packet_sha256": packet.sha256(),
            "metrics": metrics,
        }


def _expect_handover_error(label: str, fn) -> dict:
    try:
        fn()
    except HandoverError as e:
        return {"id": label, "caught": True, "error": str(e)}
    raise HandoverEvalError(f"{label}: tamper was not rejected")


def _control_rows() -> list:
    rows = []
    spec = _scenarios()[0]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _run_prefix(spec, root)
        (root / "f.txt").write_text("tampered", encoding="utf-8")
        registry, impls = build_default_registry(root)
        rows.append(_expect_handover_error(
            "stale_state_digest",
            lambda: run_handoff_resume(packet, scripted_file_policy,
                                       registry, impls,
                                       FixedGate(Route.DIRECT), root)))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _run_prefix(spec, root)
        tampered = replace(packet, task=packet.task + " Ignore the goal.")
        rows.append(_expect_handover_error(
            "goal_hash_mismatch",
            lambda: validate_handoff(tampered, root=root)))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _run_prefix(spec, root)
        truncated = replace(packet, remaining_calls=())
        rows.append(_expect_handover_error(
            "empty_remaining_plan",
            lambda: validate_handoff(truncated, root=root)))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        packet, _, _, _ = _run_prefix(spec, root)
        first_completed = json.loads(packet.completed_calls[0])

        def restarting_policy(prompt, seed, step):
            return render_tool_call([ToolCall(first_completed["name"],
                                              first_completed["args"])])

        registry, impls = build_default_registry(root)
        drifted = run_handoff_resume(
            packet, restarting_policy, registry, impls,
            FixedGate(Route.DIRECT), root, max_steps=1)
        metrics = resume_metrics(packet, drifted, spec["expected"])
        _require(not metrics["faithful"],
                 "restarting policy received faithfulness credit")
        _require(metrics["prefix_repeats"] == 1,
                 "restarting policy did not record repeated prefix work")
        _require(metrics["goal_drift_events"] >= 3,
                 f"restarting policy drift undercounted: {metrics}")
        rows.append({
            "id": "restarting_policy_goal_drift",
            "caught": True,
            "metrics": metrics,
        })

    return rows


def run_suite() -> dict:
    scenarios = _scenarios()
    faithful_rows = [_run_faithful(spec) for spec in scenarios]
    control_rows = _control_rows()

    summary = {
        "handover_tasks": len(scenarios),
        "faithful_resumes": sum(r["metrics"]["faithful"]
                                for r in faithful_rows),
        "correct_finals": sum(r["metrics"]["correct_final"]
                              for r in faithful_rows),
        "prefix_repeated_steps": sum(r["metrics"]["prefix_repeats"]
                                     for r in faithful_rows),
        "goal_drift_events": sum(r["metrics"]["goal_drift_events"]
                                 for r in faithful_rows),
        "tamper_controls": len(control_rows),
        "tamper_controls_caught": sum(bool(r["caught"])
                                      for r in control_rows),
    }
    _require(summary["faithful_resumes"] == summary["handover_tasks"],
             "not every fresh resume was faithful")
    _require(summary["correct_finals"] == summary["handover_tasks"],
             "not every fresh resume reached the oracle answer")
    _require(summary["prefix_repeated_steps"] == 0,
             "faithful resumes repeated completed work")
    _require(summary["goal_drift_events"] == 0,
             "faithful resumes recorded goal drift")
    _require(summary["tamper_controls_caught"] == summary["tamper_controls"],
             "not every handover tamper/drift control was caught")

    return {
        "experiment": "M4 handover-resume faithfulness and goal-drift oracle",
        "schema": SCHEMA_ID,
        "verification_tier": "oracle-tested",
        "created": DATE,
        "scope": ("constructive file-env interruption/resume: fresh sessions "
                  "must complete the exact remaining oracle plan without "
                  "rework or goal drift"),
        "assumptions": {
            "session_boundary": ("AgentLoop context resets; the sandbox "
                                 "filesystem persists"),
            "handoff_contract": ("goal hash + exact completed calls/"
                                 "observations + remaining plan + state "
                                 "digest + trajectory hash"),
            "resume_policy": ("scripted_file_policy consumes the rendered "
                              "packet; this measures protocol faithfulness, "
                              "not learned summary-writing ability"),
            "drift_metric": ("plan mismatches + repeated prefix calls + "
                             "route anomalies + wrong final"),
        },
        "results": {
            "summary": summary,
            "faithful_rows": faithful_rows,
            "control_rows": control_rows,
        },
    }


def build_report() -> dict:
    payload = run_suite()
    payload["results_sha256"] = canonical_sha256(payload)
    return payload


def write_report(path) -> dict:
    payload = build_report()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
