"""Terminal-Bench Phase 2C: decontamination ledger and result integration.

This module is pure data validation. It does not download task text, scan a
training corpus, start a container, run a model or call a verifier. The
caller supplies the training-mixture manifest produced by an actual
decontamination scan, an inventory report produced by Phase 2A and a native
run report produced inside the attested Harbor runtime. This module checks
their identities and accounting, keeps every task in the ledger exactly once
and recomputes the terminal component score from the scored rows.
"""
import json
import math

from helioslm_v5.eval.tbench_harbor import (
    HARBOR_API_COMMIT, INVENTORY_SCHEMA, HarborBridgeError, _rows_by_id)

DECONTAM_SCHEMA = "tbench-decontam/1"
RUN_SCHEMA = "tbench-native-run/1"
MIXTURE_SCHEMA = "training-mixture/1"
COMPONENT_SCHEMA = "agentic-component/1"

_FAILURE_CATEGORIES = (
    "wrong_answer", "agent_timeout", "agent_exception",
    "verifier_exception", "environment_failure",
)
_SKIP_REASONS = (
    "contaminated", "native_runtime_unavailable", "image_pull_or_build_failed",
    "absolute_path_isolation_required", "verifier_dependencies_unavailable",
    "unsupported_os", "multi_container", "multi_step", "separate_verifier",
    "artifact_handoff", "accelerator_requested", "mcp_requested",
    "network_policy_unsupported", "invalid_task", "unsupported_schema",
    "missing_required_file",
)


def _require_text(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HarborBridgeError(f"{key} must be a non-empty string")
    return value


def _require_hex64(obj, key):
    value = _require_text(obj, key)
    if (len(value) != 64 or value != value.lower()
            or any(c not in "0123456789abcdef" for c in value)):
        raise HarborBridgeError(f"{key} must be 64 lowercase hexadecimal characters")
    return value


def _identity(report):
    if not isinstance(report, dict):
        raise HarborBridgeError("report must be an object")
    return {
        "task_set": _require_text(report, "task_set"),
        "source_commit": _require_text(report, "source_commit"),
        "tasks_tree": _require_text(report, "tasks_tree"),
        "task_set_sha256": _require_hex64(report, "task_set_sha256"),
    }


def _check_identity(report, inventory, what):
    identity = _identity(report)
    for key, value in identity.items():
        if inventory[key] != value:
            raise HarborBridgeError(
                f"{what} {key} does not match the inventory identity")


def validate_mixture(mixture):
    """Validate a caller-supplied training-mixture decontamination manifest."""
    if not isinstance(mixture, dict):
        raise HarborBridgeError("mixture must be an object")
    if mixture.get("schema") != MIXTURE_SCHEMA:
        raise HarborBridgeError(f"mixture schema must be {MIXTURE_SCHEMA}")
    _require_text(mixture, "name")
    _require_hex64(mixture, "sha256")
    _require_text(mixture, "method")
    matches = mixture.get("matches")
    if not isinstance(matches, list):
        raise HarborBridgeError("mixture matches must be a list")
    matched = set()
    for match in matches:
        if not isinstance(match, dict):
            raise HarborBridgeError("mixture match must be an object")
        task = _require_text(match, "task")
        _require_hex64(match, "evidence_sha256")
        if task in matched:
            raise HarborBridgeError(f"duplicate mixture match for {task!r}")
        matched.add(task)
    return matched


def build_decontamination_report(inventory_report, mixture):
    """Fold the mixture manifest into a decontamination ledger.

    Every inventory task appears in the ledger exactly once. A task matched
    by the mixture scan is recorded as excluded with its match evidence; it
    never silently becomes a zero and never disappears from the accounting.
    """
    if inventory_report.get("schema") != INVENTORY_SCHEMA:
        raise HarborBridgeError(f"inventory schema must be {INVENTORY_SCHEMA}")
    identity = _identity(inventory_report)
    matched = validate_mixture(mixture)
    rows = _rows_by_id(inventory_report)
    unknown = matched - set(rows)
    if unknown:
        raise HarborBridgeError(
            f"mixture matches reference tasks absent from the inventory: "
            f"{sorted(unknown)!r}")
    evidence = {match["task"]: match["evidence_sha256"]
                for match in mixture["matches"]}
    ledger = []
    for row in inventory_report["tasks"]:
        task = row["task"]
        entry = {"task": task, "status": row["status"],
                 "skipped_reasons": list(row["skipped_reasons"])}
        if task in matched:
            entry["excluded"] = True
            entry["exclusion_reason"] = "contaminated"
            entry["evidence_sha256"] = evidence[task]
        ledger.append(entry)
    return {
        "schema": DECONTAM_SCHEMA,
        **identity,
        "mixture": {
            "name": mixture["name"],
            "sha256": mixture["sha256"],
            "method": mixture["method"],
        },
        "n_tasks": len(ledger),
        "n_contaminated": len(matched),
        "excluded_ids": sorted(matched),
        "rows": ledger,
    }


def _validate_scored_row(row):
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise HarborBridgeError("reward must be a finite number")
    if not math.isfinite(reward):
        raise HarborBridgeError("reward must be a finite number")
    if not 0.0 <= float(reward) <= 1.0:
        raise HarborBridgeError("reward must be within [0, 1]")
    if row.get("verifier_status") != "completed":
        raise HarborBridgeError("scored rows require verifier_status 'completed'")
    category = row.get("failure_category")
    if float(reward) >= 1.0:
        if category is not None:
            raise HarborBridgeError(
                "a passed task must not carry a failure category")
    elif category not in _FAILURE_CATEGORIES:
        raise HarborBridgeError(
            f"a failed task requires a failure category in {_FAILURE_CATEGORIES!r}")
    digest = _require_text(row, "resolved_image_digest")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise HarborBridgeError(
            "resolved_image_digest must be a sha256: image digest")
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HarborBridgeError("scored rows require non-empty hashed artifacts")
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise HarborBridgeError("artifact must be an object")
        path = _require_text(artifact, "path")
        if path in seen:
            raise HarborBridgeError(f"duplicate artifact path {path!r}")
        seen.add(path)
        _require_hex64(artifact, "sha256")


def _validate_skipped_row(row):
    reasons = row.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        raise HarborBridgeError("skipped rows require non-empty reasons")
    for reason in reasons:
        if reason not in _SKIP_REASONS:
            raise HarborBridgeError(f"unknown skip reason {reason!r}")


def build_terminal_component(run_report, inventory_report, decontam_report,
                             *, model, checkpoint_sha256, harbor_identity,
                             evidence_type):
    """Integrate a native run report into a terminal component descriptor.

    Every ``eligible_static`` inventory row must appear in the run report
    exactly once, either as a scored row or as a skipped row with taxonomy
    reasons. A contaminated task must never be scored and must be skipped
    with the stable ``contaminated`` reason. The component score is the mean
    raw reward recomputed from the scored rows over the decontaminated
    denominator; a summary field is never trusted. Identity mismatches
    between the inventory, decontamination and run reports are hard failures.
    """
    if not isinstance(run_report, dict):
        raise HarborBridgeError("run report must be an object")
    if run_report.get("schema") != RUN_SCHEMA:
        raise HarborBridgeError(f"run report schema must be {RUN_SCHEMA}")
    inventory_identity = _identity(inventory_report)
    _check_identity(decontam_report, inventory_identity,
                    "decontamination report")
    _check_identity(run_report, inventory_identity, "run report")
    if decontam_report.get("schema") != DECONTAM_SCHEMA:
        raise HarborBridgeError(
            f"decontamination report schema must be {DECONTAM_SCHEMA}")
    if evidence_type not in ("model_run", "schema_fixture"):
        raise HarborBridgeError("evidence_type must be model_run or schema_fixture")
    _require_text(run_report, "harbor_identity")
    inventory_rows = _rows_by_id(inventory_report)
    eligible = [task for task, row in inventory_rows.items()
                if row["status"] == "eligible_static"
                and not row["skipped_reasons"]]
    contaminated = set(decontam_report.get("excluded_ids", []))

    scored = {}
    skipped = {}
    run_rows = run_report.get("rows")
    if not isinstance(run_rows, list):
        raise HarborBridgeError("run report rows must be a list")
    for row in run_rows:
        if not isinstance(row, dict):
            raise HarborBridgeError("run row must be an object")
        task = _require_text(row, "task")
        if task not in inventory_rows:
            raise HarborBridgeError(
                f"run row {task!r} is absent from the inventory")
        if task in scored or task in skipped:
            raise HarborBridgeError(f"duplicate run row for {task!r}")
        if "reward" in row:
            _validate_scored_row(row)
            scored[task] = row
        else:
            _validate_skipped_row(row)
            skipped[task] = row

    for task in eligible:
        if task not in scored and task not in skipped:
            raise HarborBridgeError(
                f"eligible tasks missing from the run accounting: {task!r}")
    for task in sorted(contaminated & set(scored)):
        raise HarborBridgeError(
            f"contaminated tasks must not be scored: {task!r}")
    for task in sorted(contaminated & set(eligible)):
        if task not in skipped:
            raise HarborBridgeError(
                f"contaminated eligible tasks must be skipped with the "
                f"'contaminated' reason: {task!r}")
        if "contaminated" not in skipped[task]["reasons"]:
            raise HarborBridgeError(
                f"contaminated eligible tasks must be skipped with the "
                f"'contaminated' reason: {task!r}")

    denominator = sorted(set(eligible) - contaminated)
    if not denominator:
        raise HarborBridgeError(
            "no scorable tasks remain after decontamination exclusions")
    if sorted(scored) != denominator:
        raise HarborBridgeError(
            "scored rows must cover every non-contaminated eligible task")
    score = sum(float(scored[task]["reward"]) for task in denominator)
    score /= len(denominator)

    return {
        "schema": COMPONENT_SCHEMA,
        "component": "terminal",
        "model": _require_text({"model": model}, "model"),
        "checkpoint_sha256": _require_hex64(
            {"checkpoint_sha256": checkpoint_sha256}, "checkpoint_sha256"),
        "task_set": inventory_identity["task_set"],
        "task_set_sha256": inventory_identity["task_set_sha256"],
        "score": score,
        "n": len(denominator),
        "evidence_type": evidence_type,
        "verification_tier": (
            "measured" if evidence_type == "model_run" else "schema-fixture"),
        "tbench": {
            "source_commit": inventory_identity["source_commit"],
            "tasks_tree": inventory_identity["tasks_tree"],
            "harbor_api_commit": HARBOR_API_COMMIT,
            "harbor_identity": run_report["harbor_identity"],
            "n_eligible": len(eligible),
            "n_contaminated": len(contaminated & set(eligible)),
            "excluded_ids": sorted(contaminated & set(eligible)),
            "skipped": {task: list(skipped[task]["reasons"])
                        for task in sorted(skipped)},
            "rows": [
                {
                    "task": task,
                    "reward": scored[task]["reward"],
                    "failure_category": scored[task].get("failure_category"),
                    "resolved_image_digest":
                        scored[task]["resolved_image_digest"],
                    "artifacts": [dict(a) for a in scored[task]["artifacts"]],
                }
                for task in denominator
            ],
        },
    }


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


__all__ = [
    "COMPONENT_SCHEMA", "DECONTAM_SCHEMA", "MIXTURE_SCHEMA", "RUN_SCHEMA",
    "build_decontamination_report", "build_terminal_component",
    "validate_mixture",
]
