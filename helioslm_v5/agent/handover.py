"""HeliosLM v5.54 — handover-resume packets and goal-drift metrics (M4).

A handover is not a prose summary here. It is a small verifiable object:

- the original goal and its SHA-256;
- the exact completed tool calls and observations;
- the exact remaining oracle plan;
- the sandbox state digest at the interruption point;
- the source trajectory hash.

The fresh session receives the rendered packet in its prompt, while the
harness validates the packet against the live sandbox before resuming.
That separates two failure modes that free-text handoffs usually blur:

- unfaithful handoff: stale state, wrong goal, malformed/missing plan;
- goal drift after resume: repeated completed work, off-plan calls, route
  anomalies, or a wrong final answer.

The module is protocol machinery. It does not claim that a learned model is
good at writing handoffs; it makes handoff/resume behavior measurable under
an oracle so that claim can be tested later without moving the goalposts.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from .gate import Route
    from .loop import AgentLoop
    from .schema import (ToolCall, ToolCallError, ToolRegistry,
                         parse_tool_call, render_tool_call)
    from .trajectory import Trajectory, verify_replay
except ImportError:
    from gate import Route
    from loop import AgentLoop
    from schema import (ToolCall, ToolCallError, ToolRegistry,
                        parse_tool_call, render_tool_call)
    from trajectory import Trajectory, verify_replay

SCHEMA_ID = "handover-resume/1"


class HandoverError(ValueError):
    """Malformed, stale, or oracle-mismatched handover. Never swallowed."""


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def canonical_sha256(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def goal_sha256(task_text: str) -> str:
    if not isinstance(task_text, str) or not task_text.strip():
        raise HandoverError("handover goal must be a non-empty string")
    return _sha256_bytes(task_text.encode("utf-8"))


def call_key(call: ToolCall) -> str:
    """Canonical JSON identity for equality, hashing, and drift checks."""
    if not isinstance(call, ToolCall):
        raise HandoverError(f"plan entry is not a ToolCall: {call!r}")
    return canonical_json({"name": call.name, "args": call.args})


def call_from_key(key: str, registry: ToolRegistry | None = None) -> ToolCall:
    """Decode a canonical call, optionally validating it against a registry."""
    if not isinstance(key, str):
        raise HandoverError(f"canonical call must be a string, got {key!r}")
    try:
        payload = json.loads(key)
    except json.JSONDecodeError as e:
        raise HandoverError(f"canonical call is not JSON: {e}") from e
    if not isinstance(payload, dict) or set(payload) != {"name", "args"}:
        raise HandoverError("canonical call must have exactly {name, args}")
    if not isinstance(payload["name"], str) or not payload["name"]:
        raise HandoverError("canonical call name must be a non-empty string")
    if not isinstance(payload["args"], dict):
        raise HandoverError("canonical call args must be an object")
    call = ToolCall(payload["name"], payload["args"])
    if registry is not None:
        try:
            parsed = parse_tool_call(render_tool_call([call]), registry)[0]
        except ToolCallError as e:
            raise HandoverError(f"canonical call fails registry: {e}") from e
        if parsed != call:
            raise HandoverError("registry round-trip changed a canonical call")
    return call


def state_snapshot(root) -> list:
    """Sorted file digest list for the sandbox at the interruption point."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise HandoverError(f"sandbox root does not exist: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        })
    return rows


def state_sha256(root) -> str:
    return canonical_sha256({"files": state_snapshot(root)})


def _require_hex64(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise HandoverError(f"{field} must be a 64-char SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as e:
        raise HandoverError(f"{field} is not hex: {value!r}") from e


@dataclass(frozen=True)
class HandoffPacket:
    task: str
    seed: int
    completed_calls: tuple
    completed_observations: tuple
    remaining_calls: tuple
    state_sha256: str
    trajectory_sha256: str
    goal_sha256: str

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_ID,
            "task": self.task,
            "seed": self.seed,
            "completed_calls": list(self.completed_calls),
            "completed_observations": list(self.completed_observations),
            "remaining_calls": list(self.remaining_calls),
            "state_sha256": self.state_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            "goal_sha256": self.goal_sha256,
        }

    @staticmethod
    def from_dict(payload: dict) -> "HandoffPacket":
        keys = {"schema", "task", "seed", "completed_calls",
                "completed_observations", "remaining_calls", "state_sha256",
                "trajectory_sha256", "goal_sha256"}
        if not isinstance(payload, dict) or set(payload) != keys:
            raise HandoverError(
                "handoff packet must have exactly the schema keys")
        if payload["schema"] != SCHEMA_ID:
            raise HandoverError(
                f"unknown handoff schema {payload['schema']!r}")
        packet = HandoffPacket(
            task=payload["task"],
            seed=payload["seed"],
            completed_calls=tuple(payload["completed_calls"]),
            completed_observations=tuple(payload["completed_observations"]),
            remaining_calls=tuple(payload["remaining_calls"]),
            state_sha256=payload["state_sha256"],
            trajectory_sha256=payload["trajectory_sha256"],
            goal_sha256=payload["goal_sha256"],
        )
        validate_handoff(packet)
        return packet

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_handoff(traj: Trajectory, plan: list, registry: ToolRegistry,
                  impls: dict, root) -> HandoffPacket:
    """Build a packet only if the interrupted trajectory is exactly a plan
    prefix, replays bitwise, and leaves the sandbox in its recorded state."""
    if not isinstance(traj, Trajectory):
        raise HandoverError("build_handoff requires a Trajectory")
    if not plan:
        raise HandoverError("oracle plan must be non-empty")
    prefix = len(traj.steps)
    if prefix == 0 or prefix >= len(plan):
        raise HandoverError(
            "handover requires a strict non-empty plan prefix")
    if traj.final_answer is not None:
        raise HandoverError("cannot hand over an already-finished trajectory")
    if plan[-1].name != "finish":
        raise HandoverError("oracle plan must end with finish")

    # Do not trust the caller's list of what happened: re-derive it.
    verify_replay(traj, registry, impls)
    expected = [call_key(c) for c in plan]
    actual = []
    observations = []
    for i, step in enumerate(traj.steps):
        if step.parse_error is not None or step.parsed is None:
            raise HandoverError(f"step {i} is not a parsed call")
        if step.route != Route.DIRECT.value:
            raise HandoverError(f"step {i} route {step.route!r} != DIRECT")
        key = call_key(step.parsed)
        if key != expected[i]:
            raise HandoverError(
                f"step {i} diverges from oracle plan: {key} != {expected[i]}")
        actual.append(key)
        observations.append(step.observation)

    return HandoffPacket(
        task=traj.task,
        seed=traj.seed,
        completed_calls=tuple(actual),
        completed_observations=tuple(observations),
        remaining_calls=tuple(expected[prefix:]),
        state_sha256=state_sha256(root),
        trajectory_sha256=canonical_sha256(traj.to_dict()),
        goal_sha256=goal_sha256(traj.task),
    )


def validate_handoff(packet: HandoffPacket, root=None,
                     registry: ToolRegistry | None = None) -> None:
    if not isinstance(packet, HandoffPacket):
        raise HandoverError("validate_handoff requires a HandoffPacket")
    if packet.goal_sha256 != goal_sha256(packet.task):
        raise HandoverError("goal hash does not match packet task")
    if type(packet.seed) is not int or packet.seed < 0:
        raise HandoverError("handoff seed must be a non-negative int")
    if not packet.completed_calls:
        raise HandoverError("handoff has no completed prefix")
    if len(packet.completed_calls) != len(packet.completed_observations):
        raise HandoverError("completed calls/observations length mismatch")
    if not packet.remaining_calls:
        raise HandoverError("handoff has no remaining plan")
    _require_hex64(packet.state_sha256, "state_sha256")
    _require_hex64(packet.trajectory_sha256, "trajectory_sha256")
    for key in packet.completed_calls + packet.remaining_calls:
        call_from_key(key, registry)
    if root is not None and state_sha256(root) != packet.state_sha256:
        raise HandoverError("sandbox state differs from handoff digest")


def render_handoff(packet: HandoffPacket) -> str:
    """Render the packet as prompt-visible, machine-scannable text.

    The ``step i:`` lines intentionally match loop.py's observation format:
    a scripted resume policy can count them exactly like in-session history.
    """
    validate_handoff(packet)
    lines = [
        f"HANDOVER {SCHEMA_ID}",
        f"packet_sha256: {packet.sha256()}",
        f"goal_sha256: {packet.goal_sha256}",
        f"state_sha256: {packet.state_sha256}",
        f"trajectory_sha256: {packet.trajectory_sha256}",
        "completed_observations:",
    ]
    lines += [f"step {i}: {obs}"
              for i, obs in enumerate(packet.completed_observations)]
    lines.append("remaining_plan:")
    lines += [f"remaining {i}: {key}"
              for i, key in enumerate(packet.remaining_calls)]
    return "\n".join(lines)


def resume_prompt(packet: HandoffPacket) -> str:
    validate_handoff(packet)
    return packet.task + "\n" + render_handoff(packet)


def run_handoff_resume(packet: HandoffPacket, model_fn,
                       registry: ToolRegistry, impls: dict, gate,
                       root, max_steps: int = 8) -> Trajectory:
    """Start a fresh AgentLoop whose only prior context is the handoff."""
    validate_handoff(packet, root=root, registry=registry)
    loop = AgentLoop(model_fn, registry, impls, gate, max_steps=max_steps)
    return loop.run(resume_prompt(packet), seed=packet.seed)


def resume_metrics(packet: HandoffPacket, traj: Trajectory,
                   expected_answer: str) -> dict:
    """Measure resume faithfulness and goal drift against the oracle suffix."""
    validate_handoff(packet)
    expected = list(packet.remaining_calls)
    actual = []
    route_anomalies = 0
    for step in traj.steps:
        if step.parse_error is not None or step.parsed is None \
                or step.route != Route.DIRECT.value:
            route_anomalies += 1
        actual.append(None if step.parsed is None else call_key(step.parsed))

    first_divergence = None
    plan_mismatches = 0
    for i in range(max(len(actual), len(expected))):
        got = actual[i] if i < len(actual) else None
        want = expected[i] if i < len(expected) else None
        if got != want:
            plan_mismatches += 1
            if first_divergence is None:
                first_divergence = {"index": i, "expected": want,
                                    "actual": got}

    completed = set(packet.completed_calls)
    prefix_repeats = sum(key in completed for key in actual
                         if key is not None)
    wrong_final = int(traj.final_answer != expected_answer)
    events = plan_mismatches + prefix_repeats + route_anomalies + wrong_final
    return {
        "expected_calls": expected,
        "actual_calls": actual,
        "steps_used": len(traj.steps),
        "final_answer": traj.final_answer,
        "correct_final": not wrong_final,
        "plan_mismatches": plan_mismatches,
        "prefix_repeats": prefix_repeats,
        "route_anomalies": route_anomalies,
        "wrong_final": wrong_final,
        "goal_drift_events": events,
        "first_divergence": first_divergence,
        "faithful": events == 0,
    }
