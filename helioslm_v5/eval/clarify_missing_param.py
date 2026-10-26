"""M3 constructive oracle for TrustGate CLARIFY (v5.53).

The suite separates three behaviors that a two-state gate collapses:

- a syntactically valid call missing a required parameter must CLARIFY and
  must not execute;
- the same call with the parameter supplied must follow the wrapped trust
  gate (DIRECT in the positive control);
- a complete but distrusted call must still ESCALATE.

This is an oracle-tested artifact by construction: the expected missing
parameter is planted in ``MissingParamEnv.SCENARIOS``, and any execution or
wrong route fails the suite loudly instead of being averaged away.
"""
import json
import tempfile
from pathlib import Path

from helioslm_v5.agent.envs.toy_envs import MissingParamEnv
from helioslm_v5.agent.gate import FixedGate, OracleGate, Route
from helioslm_v5.agent.loop import AgentLoop
from helioslm_v5.agent.schema import ToolCall, render_tool_call
from helioslm_v5.agent.tools import build_default_registry
from helioslm_v5.agent.trust_gate_clarify import (SCHEMA_ID,
                                                  TrustGateClarify,
                                                  inspect_call)
from helioslm_v5.eval.cost_per_task import canonical_sha256

DATE = "2026-10-13"


class ClarifyEvalError(RuntimeError):
    """Any oracle mismatch in the CLARIFY suite. Never swallowed."""


_FILLERS = {
    "n": 2,
    "content": "alpha beta",
    "path": "note.txt",
    "expr": "1+1",
}


def _completed_args(spec: dict) -> dict:
    args = dict(spec["args"])
    for name in spec["missing"]:
        if name not in _FILLERS:
            raise ClarifyEvalError(f"no deterministic filler for {name!r}")
        args[name] = _FILLERS[name]
    return args


def _scripted_call(call: ToolCall):
    text = render_tool_call([call])

    def model_fn(prompt, seed, step):
        return text

    return model_fn


def _run_call(call: ToolCall, task_text: str, gate, root: Path,
              defer_validation: bool = True):
    registry, impls = build_default_registry(root)
    # The file-read control needs one deterministic readable file; create it
    # only for file_read so the file_write no-execution oracle is meaningful.
    if call.name == "file_read":
        (root / "note.txt").write_text("alpha beta", encoding="utf-8")
    loop = AgentLoop(_scripted_call(call), registry, impls, gate,
                     max_steps=2, defer_validation=defer_validation)
    return loop.run(task_text, seed=0), registry


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ClarifyEvalError(message)


def run_suite() -> dict:
    env = MissingParamEnv()
    missing_rows = []
    direct_rows = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(len(env.SCENARIOS)):
            spec = env.scenario(i)
            partial = ToolCall(spec["tool"], dict(spec["args"]))
            gate_registry, _ = build_default_registry(root)
            gate = TrustGateClarify(FixedGate(Route.DIRECT), gate_registry)

            pre = inspect_call(partial, gate_registry)
            _require(pre["status"] == "missing",
                     f"scenario {i} did not plant a missing parameter")
            _require(tuple(pre["missing"]) == spec["missing"],
                     f"scenario {i} missing mismatch: {pre['missing']}")

            traj, registry = _run_call(partial, spec["text"], gate, root)
            _require(len(traj.steps) == 1,
                     f"scenario {i}: CLARIFY must terminate in one step")
            step = traj.steps[0]
            _require(step.route == "CLARIFY",
                     f"scenario {i}: route {step.route} != CLARIFY")
            _require(step.observation.startswith("CLARIFY:"),
                     f"scenario {i}: observation is not a question")
            for name in spec["missing"]:
                _require(name in step.observation,
                         f"scenario {i}: question omits {name!r}")
            _require(traj.final_answer is None,
                     f"scenario {i}: clarify leaked a final answer")
            if spec["tool"] == "file_write":
                target = root / spec["args"]["path"]
                _require(not target.exists(),
                         f"scenario {i}: file_write executed despite CLARIFY")
            missing_rows.append({
                "id": f"missing-{i:02d}",
                "family": spec["family"],
                "tool": spec["tool"],
                "missing": list(spec["missing"]),
                "route": step.route,
                "observation": step.observation,
                "final_answer": traj.final_answer,
            })

            complete = ToolCall(spec["tool"], _completed_args(spec))
            direct_gate = TrustGateClarify(FixedGate(Route.DIRECT),
                                           registry)
            direct_traj, _ = _run_call(complete, spec["text"], direct_gate,
                                       root)
            _require(len(direct_traj.steps) >= 1,
                     f"control {i}: complete call produced no step")
            direct_step = direct_traj.steps[0]
            _require(direct_step.route == "DIRECT",
                     f"control {i}: complete call did not DIRECT")
            _require(not direct_step.observation.startswith("CLARIFY:"),
                     f"control {i}: complete call incorrectly clarified")
            direct_rows.append({
                "id": f"control-{i:02d}",
                "family": spec["family"],
                "tool": spec["tool"],
                "route": direct_step.route,
                "observation": direct_step.observation,
            })

        # Complete but distrusted calls remain ESCALATE; ambiguity does not
        # weaken the wrapped gate's trust decision.
        spec = env.scenario(0)
        complete = ToolCall(spec["tool"], _completed_args(spec))
        oracle = OracleGate(lambda call, ctx: "ORACLE_REVIEW")
        escalate_gate = TrustGateClarify(oracle, build_default_registry(root)[0])
        rec = escalate_gate.decide_explain(complete, {"task": spec["text"]})
        _require(rec["route"] == "ESCALATE",
                 "complete low-trust control did not ESCALATE")
        _require(escalate_gate.escalate(complete, {}) == "ORACLE_REVIEW",
                 "escalate did not delegate to oracle")

        # Extra/wrong-typed args are schema-invalid, not a clarification.
        invalid = ToolCall("str_op", {"s": "ab", "op": "repeat", "n": 2,
                                      "bogus": "x"})
        invalid_rec = escalate_gate.decide_explain(invalid, {})
        _require(invalid_rec["route"] == "ESCALATE",
                 "invalid schema control did not ESCALATE")

    summary = {
        "missing_tasks": len(missing_rows),
        "clarified": sum(r["route"] == "CLARIFY" for r in missing_rows),
        "clarify_no_final_answer": sum(r["final_answer"] is None
                                       for r in missing_rows),
        "complete_controls": len(direct_rows),
        "direct_after_completion": sum(r["route"] == "DIRECT"
                                       for r in direct_rows),
        "low_trust_controls": 1,
        "escalated_low_trust": 1,
        "invalid_schema_controls": 1,
        "escalated_invalid_schema": 1,
    }
    _require(summary["clarified"] == summary["missing_tasks"],
             "not every missing-param task clarified")
    _require(summary["direct_after_completion"] ==
             summary["complete_controls"],
             "not every completed control routed DIRECT")

    return {
        "experiment": "M3 TrustGate CLARIFY three-state missing-parameter oracle",
        "schema": SCHEMA_ID,
        "verification_tier": "oracle-tested",
        "created": DATE,
        "scope": ("constructive toy-env oracle: missing required parameters "
                  "must ask a concrete clarification and never execute"),
        "assumptions": {
            "three_states": ["DIRECT", "CLARIFY", "ESCALATE"],
            "missing_param_policy": "CLARIFY before trust scoring or execution",
            "invalid_schema_policy": "ESCALATE; extras/wrong types are not clarifications",
            "loop_mode": "AgentLoop(defer_validation=True) parses syntax and defers arg validation to the gate",
        },
        "results": {
            "summary": summary,
            "missing_rows": missing_rows,
            "complete_control_rows": direct_rows,
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
