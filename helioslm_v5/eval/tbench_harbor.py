"""Isolated Harbor command/observation bridge core (Terminal-Bench Phase 2B.1).

This module supplies the agent-side bridge only.  Harbor orchestration still
owns task construction, environment lifecycle, episode timeouts, the verifier,
and raw rewards.  Nothing here imports Harbor at module import time, starts a
container, opens a socket, reads host credentials, or executes host commands.
The actual container-only oracle/no-op smoke remains a separate runtime gate.

The public ``build_agent_class`` lazy-imports Harbor only when an authorized,
isolated run constructs its agent.  Unit tests inject a synthetic base class and
fake environment; they are schema/protocol oracles, not Terminal-Bench results.

Harbor interface audit pin: commit
``90f2ee3c18d394eb0da0070415c6297eb13b87f8`` (2026-11-25).
"""

import copy
import inspect
import re
from dataclasses import asdict, dataclass

from helioslm_v5.eval.term_bench import parse_reply


SCHEMA = "tbench-harbor-episode/1"
INVENTORY_SCHEMA = "tbench-inventory/1"
HARBOR_API_COMMIT = "90f2ee3c18d394eb0da0070415c6297eb13b87f8"
DEFAULT_MAX_STEPS = 10
DEFAULT_OUTPUT_CHARS = 4000
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class HarborBridgeError(ValueError):
    """Unsafe runtime claims, invalid identity, or a malformed bridge reply."""


@dataclass(frozen=True)
class IsolationAttestation:
    """Explicit runtime boundary supplied by the authorized outer orchestrator.

    These flags describe the *task container*, not merely the Python process.
    The bridge cannot prove a provider's isolation from inside Python; requiring
    complete, explicit attestations prevents an accidental implicit host run.
    """

    outer_runtime: str
    harbor_backend: str
    harbor_version: str
    runtime_id: str
    disposable: bool
    host_credentials_exposed: bool
    host_paths_exposed: bool
    host_docker_socket_exposed: bool

    def validate(self):
        for name in ("outer_runtime", "harbor_backend", "harbor_version", "runtime_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise HarborBridgeError(f"isolation {name} must be a non-empty string")
        for name in ("disposable", "host_credentials_exposed", "host_paths_exposed",
                     "host_docker_socket_exposed"):
            if type(getattr(self, name)) is not bool:
                raise HarborBridgeError(f"isolation {name} must be a boolean")
        if not self.disposable:
            raise HarborBridgeError("Harbor execution requires a disposable container/pod")
        exposed = [name for name in ("host_credentials_exposed", "host_paths_exposed",
                                     "host_docker_socket_exposed")
                   if getattr(self, name)]
        if exposed:
            raise HarborBridgeError(
                "unsafe Harbor isolation attestation: " + ", ".join(exposed))
        return self


def _rows_by_id(report):
    if not isinstance(report, dict) or report.get("schema") != INVENTORY_SCHEMA:
        raise HarborBridgeError(f"inventory report must use {INVENTORY_SCHEMA}")
    if report.get("mode") != "native":
        raise HarborBridgeError("Harbor bridge requires a native inventory report")
    for key, pattern in (("source_commit", _HEX40), ("tasks_tree", _HEX40),
                         ("task_set_sha256", _HEX64)):
        value = report.get(key)
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise HarborBridgeError(f"inventory {key} has an invalid identity")
    if not isinstance(report.get("task_set"), str) or not report["task_set"].strip():
        raise HarborBridgeError("inventory task_set must be non-empty")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HarborBridgeError("inventory rows must be a non-empty list")
    mapped = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise HarborBridgeError("every inventory row needs a task ID")
        if row["id"] in mapped:
            raise HarborBridgeError(f"duplicate inventory task ID: {row['id']}")
        mapped[row["id"]] = row
    return mapped


def _eligible_row(report, task_id):
    row = _rows_by_id(report).get(task_id)
    if row is None:
        raise HarborBridgeError(f"task is absent from the verified inventory: {task_id!r}")
    if row.get("status") != "eligible_static" or row.get("skipped_reasons") != []:
        raise HarborBridgeError(f"task is not eligible for native execution: {task_id!r}")
    return row


def _clip(text, limit):
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n[... {omitted} chars truncated]"


def render_prompt(instruction, history, step, max_steps, output_chars):
    """Render one command/observation turn without changing Harbor paths/env."""
    lines = [
        "You are operating a shell inside the task's Harbor environment. "
        "Complete the task by running shell commands, one per reply.",
        "Reply with exactly one command in a ```bash code block. When the "
        "task is complete, reply with 'DONE confidence=<p>' where <p> is "
        "your probability (0 to 1) that the task is solved correctly.",
        "Do not attempt to access host files, credentials, or container-control "
        "sockets. The task verifier runs separately after this agent finishes.",
        "",
        f"[step {step + 1} of {max_steps}]",
        f"Task: {instruction}",
    ]
    if history:
        lines += ["", "Commands and observations so far:"]
        for event in history:
            if event["kind"] == "invalid":
                lines += ["<no command>", event["error"], "(not executed)"]
                continue
            stdout = _clip(event["stdout"], output_chars)
            stderr = _clip(event["stderr"], output_chars)
            lines += [
                f"$ {event['command']}",
                "[stdout]", stdout.rstrip() if stdout.strip() else "(no output)",
                "[stderr]", stderr.rstrip() if stderr.strip() else "(no output)",
                f"(exit {event['return_code']})",
            ]
    return "\n".join(lines) + "\n"


async def _generate(generate_fn, prompt):
    reply = generate_fn(prompt)
    if inspect.isawaitable(reply):
        reply = await reply
    if isinstance(reply, tuple):
        reply = reply[0]
    if not isinstance(reply, str):
        raise HarborBridgeError("generate_fn must return text or awaitable text")
    return reply


async def run_agent_episode(instruction, environment, generate_fn, inventory_report,
                            task_id, isolation, max_steps=DEFAULT_MAX_STEPS,
                            output_chars=DEFAULT_OUTPUT_CHARS):
    """Run only the agent loop through an attested Harbor-like environment.

    ``environment.exec`` is called with the command alone.  In particular, the
    bridge does not replace Harbor's cwd, environment, user, or timeout policy.
    Full synthetic/runtime observations remain in the episode; only their prompt
    rendering is clipped.  This function never invokes a verifier and emits no
    score, reward, correctness bit, or model-performance claim.
    """
    isolation.validate()
    if not isinstance(instruction, str) or not instruction.strip():
        raise HarborBridgeError("instruction must be non-empty text")
    if not callable(generate_fn):
        raise HarborBridgeError("generate_fn must be callable")
    if type(max_steps) is not int or max_steps <= 0:
        raise HarborBridgeError("max_steps must be a positive integer")
    if type(output_chars) is not int or output_chars <= 0:
        raise HarborBridgeError("output_chars must be a positive integer")
    if not callable(getattr(environment, "exec", None)):
        raise HarborBridgeError("environment must provide async Harbor-style exec")
    row = _eligible_row(inventory_report, task_id)

    history = []
    confidence = None
    ended = "max_steps"
    for step in range(max_steps):
        prompt = render_prompt(instruction, history, step, max_steps, output_chars)
        reply = await _generate(generate_fn, prompt)
        kind, value = parse_reply(reply)
        if kind == "done":
            confidence = value
            ended = "done"
            break
        if kind == "invalid":
            history.append({"kind": "invalid", "error": value})
            continue
        result = await environment.exec(value)
        code = getattr(result, "return_code", None)
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        if type(code) is not int:
            raise HarborBridgeError("Harbor exec result needs an integer return_code")
        if stdout is None:
            stdout = ""
        if stderr is None:
            stderr = ""
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise HarborBridgeError("Harbor exec stdout/stderr must be text or null")
        history.append({"kind": "command", "command": value, "return_code": code,
                        "stdout": stdout, "stderr": stderr})

    return {
        "schema": SCHEMA,
        "evidence_type": "agent_protocol_only",
        "task": task_id,
        "task_set": inventory_report["task_set"],
        "source_commit": inventory_report["source_commit"],
        "tasks_tree": inventory_report["tasks_tree"],
        "task_set_sha256": inventory_report["task_set_sha256"],
        "static_row": copy.deepcopy(row),
        "isolation": asdict(isolation),
        "harbor_api_commit": HARBOR_API_COMMIT,
        "ended": ended,
        "steps": len(history),
        "commands_executed": sum(event["kind"] == "command" for event in history),
        "invalid_replies": sum(event["kind"] == "invalid" for event in history),
        "confidence": confidence,
        "confidence_type": "verbalized" if confidence is not None else None,
        "history": history,
    }


def _agent_class(base_agent, generate_fn, inventory_report, isolation,
                 max_steps=DEFAULT_MAX_STEPS, output_chars=DEFAULT_OUTPUT_CHARS):
    """Build against an injected base for synthetic tests; never runs a task."""
    isolation.validate()
    rows = _rows_by_id(inventory_report)

    class HeliosLMCommandAgent(base_agent):
        @staticmethod
        def name():
            return "helioslm-command"

        def version(self):
            return "phase2b.1"

        async def setup(self, environment):
            isolation.validate()
            if not callable(getattr(environment, "exec", None)):
                raise HarborBridgeError("environment must provide Harbor-style exec")
            # No agent installation or setup command is needed for this bridge.

        async def run(self, instruction, environment, context):
            task_id = getattr(environment, "environment_name", None)
            if task_id not in rows:
                raise HarborBridgeError(
                    f"environment is absent from the verified inventory: {task_id!r}")
            episode = await run_agent_episode(
                instruction, environment, generate_fn, inventory_report, task_id,
                isolation, max_steps=max_steps, output_chars=output_chars)
            self.helioslm_episode = episode
            metadata = dict(getattr(context, "metadata", None) or {})
            metadata["helioslm_episode"] = episode
            context.metadata = metadata
            # Deliberately no verifier call: Harbor Trial owns that lifecycle.

    return HeliosLMCommandAgent


def build_agent_class(generate_fn, inventory_report, isolation,
                      max_steps=DEFAULT_MAX_STEPS,
                      output_chars=DEFAULT_OUTPUT_CHARS):
    """Lazy-import Harbor and return its HeliosLM command-agent class.

    Call this only from the already-authorized isolated runtime.  The returned
    class still does not run a verifier or manufacture a benchmark result.
    """
    try:
        from harbor.agents.base import BaseAgent
    except ImportError as exc:
        raise HarborBridgeError(
            "Harbor is required only when constructing the native agent class") from exc
    return _agent_class(BaseAgent, generate_fn, inventory_report, isolation,
                        max_steps=max_steps, output_chars=output_chars)
