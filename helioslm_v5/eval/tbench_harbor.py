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
import math
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from helioslm_v5.eval.term_bench import parse_reply


SCHEMA = "tbench-harbor-episode/1"
SMOKE_SCHEMA = "tbench-harbor-smoke/1"
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


def _empty(value):
    return value is None or value == {} or value == []


def validate_smoke_config(config, control, task_id):
    """Validate a serialized Harbor TrialConfig for an oracle/no-op control.

    This is a pure data check.  It deliberately excludes model credentials,
    custom agents/verifiers, extra mounts, Compose overlays and instruction
    overrides.  Harbor still owns construction and execution in the isolated
    runtime; calling this function neither imports Harbor nor starts a task.
    """
    if control not in ("oracle", "nop"):
        raise HarborBridgeError("smoke control must be oracle or nop")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise HarborBridgeError("smoke config must be serialized TrialConfig schema 1")
    if config.get("install_only") is not False:
        raise HarborBridgeError("smoke must execute both agent and verifier")
    if config.get("source_trial") is not None or config.get("user_agent") is not None:
        raise HarborBridgeError("smoke rejects regrade and simulated-user configs")
    for key in ("timeout_multiplier", "agent_timeout_multiplier",
                "verifier_timeout_multiplier", "agent_setup_timeout_multiplier",
                "environment_build_timeout_multiplier"):
        value = config.get(key, 1.0)
        if type(value) not in (int, float) or value != 1.0:
            raise HarborBridgeError(f"smoke preserves Harbor timeout policy: {key}")
    if not _empty(config.get("extra_instruction_paths")) or not _empty(
            config.get("extra_instructions")):
        raise HarborBridgeError("smoke rejects instruction overrides")

    task = config.get("task")
    path = task.get("path") if isinstance(task, dict) else None
    if not isinstance(path, str) or not path.startswith("/"):
        raise HarborBridgeError("smoke task path must be absolute")
    parsed = PurePosixPath(path)
    if ".." in parsed.parts or parsed.name != task_id:
        raise HarborBridgeError("smoke task path must identify the inventory task")
    if any(task.get(key) is not None for key in
           ("git_url", "git_commit_id", "name", "ref", "source", "download_dir")):
        raise HarborBridgeError("smoke requires the pinned local task source")
    if task.get("overwrite") is not False:
        raise HarborBridgeError("smoke must not overwrite the pinned task source")

    agent = config.get("agent")
    if not isinstance(agent, dict) or agent.get("name") != control:
        raise HarborBridgeError(f"smoke requires Harbor's built-in {control} agent")
    if agent.get("import_path") is not None:
        raise HarborBridgeError("smoke rejects custom agent imports")
    for key in ("model_name", "base_url", "model_api_key_name", "model_api_format"):
        if agent.get(key) is not None:
            raise HarborBridgeError(f"smoke control must not configure {key}")
    if agent.get("skip_model_parsing") is not False:
        raise HarborBridgeError("smoke control must not configure model parsing")
    for key in ("override_timeout_sec", "override_setup_timeout_sec",
                "max_timeout_sec", "n_concurrent", "concurrency_group",
                "resume_trajectory", "load_trajectory", "bridge"):
        if not _empty(agent.get(key)):
            raise HarborBridgeError(f"smoke preserves Harbor agent policy: {key}")
    for key in ("skills", "extra_allowed_hosts", "mcp_servers", "env", "kwargs"):
        if not _empty(agent.get(key)):
            raise HarborBridgeError(f"smoke agent {key} must be empty")

    environment = config.get("environment")
    if not isinstance(environment, dict) or environment.get("delete") is not True:
        raise HarborBridgeError("smoke environment must be disposable")
    if not isinstance(environment.get("type"), str) or not environment["type"]:
        raise HarborBridgeError("smoke environment needs an explicit Harbor backend")
    if environment.get("import_path") is not None:
        raise HarborBridgeError("smoke rejects custom environment imports")
    if environment.get("force_build") is not False:
        raise HarborBridgeError("smoke preserves the pinned prebuilt image")
    for key in ("override_cpus", "override_memory_mb", "override_storage_mb",
                "override_gpus", "override_tpu", "cpu_enforcement_policy",
                "memory_enforcement_policy"):
        if not _empty(environment.get(key)):
            raise HarborBridgeError(f"smoke preserves task resources/policy: {key}")
    if environment.get("stream") is not False:
        raise HarborBridgeError("smoke preserves Harbor streaming policy")
    for key in ("mounts", "extra_docker_compose", "extra_allowed_hosts", "env", "kwargs"):
        if not _empty(environment.get(key)):
            raise HarborBridgeError(f"smoke environment {key} must be empty")

    verifier = config.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("disable") is not False:
        raise HarborBridgeError("smoke requires the upstream verifier")
    if verifier.get("import_path") is not None:
        raise HarborBridgeError("smoke rejects custom verifier imports")
    for key in ("override_timeout_sec", "max_timeout_sec"):
        if not _empty(verifier.get(key)):
            raise HarborBridgeError(f"smoke preserves verifier timeout policy: {key}")
    for key in ("extra_allowed_hosts", "env", "kwargs"):
        if not _empty(verifier.get(key)):
            raise HarborBridgeError(f"smoke verifier {key} must be empty")
    return config


def _smoke_artifacts(artifacts):
    if not isinstance(artifacts, list) or not artifacts:
        raise HarborBridgeError("smoke evidence needs result and verifier artifacts")
    normalized = []
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise HarborBridgeError("smoke artifact entries must be objects")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise HarborBridgeError("smoke artifact paths must be relative")
        parsed = PurePosixPath(path)
        if ".." in parsed.parts or str(parsed) != path:
            raise HarborBridgeError("smoke artifact paths must be normalized")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise HarborBridgeError("smoke artifact sha256 is invalid")
        if path in seen:
            raise HarborBridgeError(f"duplicate smoke artifact: {path}")
        seen.add(path)
        normalized.append({"path": path, "sha256": digest})
    if not any(PurePosixPath(item["path"]).name == "result.json"
               for item in normalized):
        raise HarborBridgeError("smoke evidence needs Harbor result.json")
    if not any("verifier" in PurePosixPath(item["path"]).parts
               for item in normalized):
        raise HarborBridgeError("smoke evidence needs a verifier log")
    return normalized


def build_smoke_report(result, control, inventory_report, task_id, isolation,
                       resolved_image_digest, artifacts, evidence_type):
    """Validate one serialized Harbor TrialResult and preserve raw rewards.

    ``evidence_type`` is deliberately mandatory: synthetic fixtures remain
    ``synthetic_contract``; only an attested real runtime may supply
    ``container_only_harness_smoke``.  Oracle requires reward 1 and no-op
    requires reward 0.  Any exception, missing verifier timing or other reward
    is a failed smoke gate, never a model score or benchmark aggregate.
    """
    if evidence_type not in ("synthetic_contract", "container_only_harness_smoke"):
        raise HarborBridgeError("smoke evidence type must be explicit")
    if control not in ("oracle", "nop"):
        raise HarborBridgeError("smoke control must be oracle or nop")
    isolation.validate()
    row = _eligible_row(inventory_report, task_id)
    if evidence_type == "container_only_harness_smoke" and (
            isolation.harbor_backend == "synthetic-environment"):
        raise HarborBridgeError("synthetic environment cannot claim container smoke")
    if not isinstance(resolved_image_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", resolved_image_digest):
        raise HarborBridgeError("smoke needs a resolved sha256 image digest")
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise HarborBridgeError("smoke result must be serialized TrialResult schema 1")
    if result.get("task_name") != task_id:
        raise HarborBridgeError("smoke result task does not match inventory")
    if not isinstance(result.get("id"), str) or not result["id"]:
        raise HarborBridgeError("smoke result needs a Harbor trial ID")
    if not isinstance(result.get("trial_name"), str) or not result["trial_name"]:
        raise HarborBridgeError("smoke result needs a Harbor trial name")
    if not isinstance(result.get("task_checksum"), str) or not result["task_checksum"]:
        raise HarborBridgeError("smoke result needs Harbor's task checksum")
    validate_smoke_config(result.get("config"), control, task_id)
    agent_info = result.get("agent_info")
    if not isinstance(agent_info, dict) or agent_info.get("name") != control:
        raise HarborBridgeError("smoke result used the wrong control agent")
    if result.get("exception_info") is not None or result.get("step_results"):
        raise HarborBridgeError("smoke accepts only exception-free single-step trials")
    if result.get("reward_valid") is not True:
        raise HarborBridgeError("Harbor did not mark the smoke reward valid")
    timing = result.get("verifier")
    if not isinstance(timing, dict) or not timing.get("started_at") or not timing.get(
            "finished_at"):
        raise HarborBridgeError("smoke needs completed Harbor verifier timing")
    verifier_result = result.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        raise HarborBridgeError("smoke needs raw Harbor verifier rewards")
    for name, value in rewards.items():
        if not isinstance(name, str) or not name or type(value) not in (int, float):
            raise HarborBridgeError("smoke reward keys/values must be text/numbers")
        if not math.isfinite(value):
            raise HarborBridgeError("smoke rewards must be finite")
    expected = 1 if control == "oracle" else 0
    if rewards.get("reward") != expected:
        raise HarborBridgeError(
            f"{control} smoke expected reward={expected}, got {rewards.get('reward')!r}")
    evidence = _smoke_artifacts(artifacts)
    return {
        "schema": SMOKE_SCHEMA,
        "evidence_type": evidence_type,
        "is_benchmark_result": False,
        "task": task_id,
        "control": control,
        "task_set": inventory_report["task_set"],
        "source_commit": inventory_report["source_commit"],
        "tasks_tree": inventory_report["tasks_tree"],
        "task_set_sha256": inventory_report["task_set_sha256"],
        "static_row": copy.deepcopy(row),
        "isolation": asdict(isolation),
        "harbor_api_commit": HARBOR_API_COMMIT,
        "harbor_trial_id": result["id"],
        "resolved_image_digest": resolved_image_digest,
        "raw_rewards": copy.deepcopy(rewards),
        "expected_reward": expected,
        "passed": True,
        "artifacts": evidence,
    }


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
