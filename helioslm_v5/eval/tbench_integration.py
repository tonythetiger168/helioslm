"""Terminal-Bench Phase 2C: decontamination ledger and result integration.

Pure-data validation only.  Nothing here downloads task text, scans a training
corpus, starts a container, runs a model, or invokes a Harbor verifier.  The
n-gram matching itself lives in ``helioslm_v5.eval.decontam`` and is run by an
authorized outer process; this module validates the caller-supplied mixture
manifest and enforces the accounting rules from
``docs/TBENCH_PHASE2_FEASIBILITY.md``:

- a contaminated task is recorded as excluded with its reason and evidence —
  it never silently becomes a zero and never disappears from the accounting;
- a native result report must account for every ``eligible_static`` inventory
  row exactly once: either a scored row or a skipped row with stable reasons;
- the terminal component score is recomputed from the scored rows, never
  copied from a summary field, and keeps the exact task-set identity.

Input identities are pinned: the run report, decontamination report and
inventory must agree on ``task_set``, ``source_commit``, ``tasks_tree`` and
``task_set_sha256``.
"""

import math
import re

from helioslm_v5.eval.tbench_harbor import HARBOR_API_COMMIT, _rows_by_id


DECONTAM_SCHEMA = "tbench-decontam/1"
NATIVE_RUN_SCHEMA = "tbench-native-run/1"
MIXTURE_SCHEMA = "training-mixture/1"
COMPONENT_SCHEMA = "agentic-component/1"

FAILURE_CATEGORIES = (
    "wrong_answer",
    "agent_timeout",
    "agent_exception",
    "verifier_exception",
    "environment_failure",
)
SKIP_REASONS = (
    "contaminated",
    "native_runtime_unavailable",
    "image_pull_or_build_failed",
    "absolute_path_isolation_required",
    "verifier_dependencies_unavailable",
    "unsupported_os",
    "multi_container",
    "multi_step",
    "separate_verifier",
    "artifact_handoff",
    "accelerator_requested",
    "mcp_requested",
    "network_policy_unsupported",
    "invalid_task",
    "unsupported_schema",
    "missing_required_file",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TbenchIntegrationError(ValueError):
    """Identity mismatch, bad accounting, or malformed Phase 2C evidence."""


def _require_text(value, what):
    if not isinstance(value, str) or not value.strip():
        raise TbenchIntegrationError(f"{what} must be a non-empty string")
    return value


def _require_hex64(value, what):
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise TbenchIntegrationError(f"{what} must be 64 lowercase hexadecimal")
    return value


def _identity(report):
    return {
        "task_set": report["task_set"],
        "source_commit": report["source_commit"],
        "tasks_tree": report["tasks_tree"],
        "task_set_sha256": report["task_set_sha256"],
    }


def _check_identity(report, inventory, what):
    for key in ("task_set", "source_commit", "tasks_tree", "task_set_sha256"):
        if report.get(key) != inventory.get(key):
            raise TbenchIntegrationError(
                f"{what} {key} does not match the verified inventory")


def validate_mixture(mixture):
    """Validate a training-mixture decontamination manifest.

    The manifest is produced by an authorized outer scan (for example with
    ``decontam.EvalIndex``).  Every match must name a task ID plus the scan
    method and a hash of the matched evidence; identities let a later audit
    distinguish which corpus and which scan produced the exclusion list.
    """
    if not isinstance(mixture, dict) or mixture.get("schema") != MIXTURE_SCHEMA:
        raise TbenchIntegrationError(f"mixture manifest must use {MIXTURE_SCHEMA}")
    _require_text(mixture.get("name"), "mixture name")
    _require_hex64(mixture.get("sha256"), "mixture sha256")
    _require_text(mixture.get("method"), "mixture scan method")
    matches = mixture.get("matches")
    if not isinstance(matches, list):
        raise TbenchIntegrationError("mixture matches must be a list")
    seen = set()
    for match in matches:
        if not isinstance(match, dict):
            raise TbenchIntegrationError("mixture match entries must be objects")
        task = _require_text(match.get("task"), "mixture match task")
        if task in seen:
            raise TbenchIntegrationError(f"duplicate mixture match: {task}")
        seen.add(task)
        _require_hex64(match.get("evidence_sha256"), "mixture match evidence")
    return mixture


def build_decontamination_report(inventory_report, mixture):
    """Ledger of clean vs contaminated tasks for one verified inventory.

    Every inventory row appears exactly once in the output, so an exclusion
    can never hide a task.  Contaminated rows keep ``excluded: True`` and
    their match evidence; nothing is scored or dropped here.
    """
    rows = _rows_by_id(inventory_report)
    mixture = validate_mixture(mixture)
    matched = {match["task"]: match for match in mixture["matches"]}
    unknown = sorted(task for task in matched if task not in rows)
    if unknown:
        raise TbenchIntegrationError(
            f"mixture matches unknown inventory tasks: {unknown}")
    ledger = []
    for task_id in sorted(rows):
        match = matched.get(task_id)
        ledger_row = {
            "id": task_id,
            "status": rows[task_id].get("status"),
            "contaminated": match is not None,
            "excluded": match is not None,
        }
        if match is not None:
            ledger_row["exclusion_reason"] = "contaminated"
            ledger_row["evidence_sha256"] = match["evidence_sha256"]
        ledger.append(ledger_row)
    excluded = sorted(matched)
    return {
        "schema": DECONTAM_SCHEMA,
        **_identity(inventory_report),
        "mixture": {
            "name": mixture["name"],
            "sha256": mixture["sha256"],
            "method": mixture["method"],
        },
        "rows": ledger,
        "n_tasks": len(rows),
        "n_contaminated": len(excluded),
        "excluded_ids": excluded,
    }


def _validate_scored_row(row, rows):
    if not isinstance(row, dict):
        raise TbenchIntegrationError("run rows must be objects")
    task = _require_text(row.get("id"), "run row id")
    if task not in rows:
        raise TbenchIntegrationError(f"run row is absent from the inventory: {task}")
    reward = row.get("reward")
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise TbenchIntegrationError(f"run row {task} reward must be a finite number")
    if row.get("verifier_status") != "completed":
        raise TbenchIntegrationError(
            f"run row {task} needs a completed Harbor verifier status")
    category = row.get("failure_category")
    if reward >= 1:
        if category is not None:
            raise TbenchIntegrationError(
                f"run row {task} passed but still declares a failure category")
    elif category not in FAILURE_CATEGORIES:
        raise TbenchIntegrationError(
            f"run row {task} needs a stable failure category")
    digest = row.get("resolved_image_digest")
    if not isinstance(digest, str) or not _IMAGE_DIGEST.fullmatch(digest):
        raise TbenchIntegrationError(
            f"run row {task} needs a resolved sha256 image digest")
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TbenchIntegrationError(f"run row {task} needs hashed raw artifacts")
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise TbenchIntegrationError("run row artifacts must be objects")
        path = _require_text(artifact.get("path"), "run row artifact path")
        if path in seen:
            raise TbenchIntegrationError(f"duplicate run row artifact: {path}")
        seen.add(path)
        _require_hex64(artifact.get("sha256"), "run row artifact sha256")
    return task


def _validate_skipped_row(row, rows):
    if not isinstance(row, dict):
        raise TbenchIntegrationError("skipped rows must be objects")
    task = _require_text(row.get("id"), "skipped row id")
    if task not in rows:
        raise TbenchIntegrationError(f"skipped row is absent from the inventory: {task}")
    reasons = row.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        raise TbenchIntegrationError(f"skipped row {task} needs explicit reasons")
    for reason in reasons:
        if reason not in SKIP_REASONS:
            raise TbenchIntegrationError(
                f"skipped row {task} has an unstable reason: {reason!r}")
    return task


def build_terminal_component(run_report, inventory_report, decontam_report, *,
                             model, checkpoint_sha256, harbor_identity,
                             evidence_type):
    """Recompute the terminal component from accounted native run rows.

    The score is the mean raw reward over tasks that are eligible, executed
    and not contaminated.  Contaminated and skipped tasks stay in the
    descriptor with their reasons; the denominator change is explicit in
    ``n`` versus ``n_eligible``.  This is data validation only — it runs no
    model, no container and no verifier.
    """
    if evidence_type not in ("model_run", "schema_fixture"):
        raise TbenchIntegrationError(
            "terminal component evidence must be model_run or schema_fixture")
    _require_text(model, "model")
    _require_hex64(checkpoint_sha256, "checkpoint_sha256")
    _require_text(harbor_identity, "harbor identity")

    rows = _rows_by_id(inventory_report)
    eligible = {task for task, row in rows.items()
                if row.get("status") == "eligible_static"
                and row.get("skipped_reasons") == []}
    if not isinstance(decontam_report, dict) or decontam_report.get(
            "schema") != DECONTAM_SCHEMA:
        raise TbenchIntegrationError(
            f"decontamination report must use {DECONTAM_SCHEMA}")
    _check_identity(decontam_report, inventory_report, "decontamination report")
    contaminated = set(decontam_report.get("excluded_ids") or [])
    if not contaminated <= set(rows):
        raise TbenchIntegrationError(
            "decontamination exclusions must be inventory tasks")

    if not isinstance(run_report, dict) or run_report.get(
            "schema") != NATIVE_RUN_SCHEMA:
        raise TbenchIntegrationError(f"run report must use {NATIVE_RUN_SCHEMA}")
    _check_identity(run_report, inventory_report, "run report")

    scored = {}
    for row in run_report.get("rows") or []:
        task = _validate_scored_row(row, rows)
        if task in scored:
            raise TbenchIntegrationError(f"duplicate run row: {task}")
        scored[task] = row
    skipped = {}
    for row in run_report.get("skipped") or []:
        task = _validate_skipped_row(row, rows)
        if task in skipped:
            raise TbenchIntegrationError(f"duplicate skipped row: {task}")
        skipped[task] = row
    overlap = sorted(set(scored) & set(skipped))
    if overlap:
        raise TbenchIntegrationError(
            f"tasks cannot be both scored and skipped: {overlap}")

    missing = sorted(eligible - set(scored) - set(skipped))
    if missing:
        raise TbenchIntegrationError(
            f"eligible tasks missing from the run accounting: {missing}")
    scored_contaminated = sorted(set(scored) & contaminated)
    if scored_contaminated:
        raise TbenchIntegrationError(
            f"contaminated tasks must not be scored: {scored_contaminated}")
    for task in sorted(contaminated & eligible):
        reasons = skipped.get(task, {}).get("reasons", [])
        if "contaminated" not in reasons:
            raise TbenchIntegrationError(
                f"contaminated task {task} must be skipped with the "
                "'contaminated' reason")

    denominator = sorted(eligible - contaminated)
    if not denominator:
        raise TbenchIntegrationError(
            "no scorable tasks remain after decontamination exclusions")
    if sorted(scored) != denominator:
        raise TbenchIntegrationError(
            "scored rows must cover every non-contaminated eligible task")
    rewards = [scored[task]["reward"] for task in denominator]
    score = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "schema": COMPONENT_SCHEMA,
        "component": "terminal",
        "model": model,
        "checkpoint_sha256": checkpoint_sha256,
        "task_set": inventory_report["task_set"],
        "task_set_sha256": inventory_report["task_set_sha256"],
        "score": score,
        "n": len(rewards),
        "evidence_type": evidence_type,
        "tbench": {
            "source_commit": inventory_report["source_commit"],
            "tasks_tree": inventory_report["tasks_tree"],
            "harbor_api_commit": HARBOR_API_COMMIT,
            "harbor_identity": harbor_identity,
            "n_eligible": len(eligible),
            "n_contaminated": len(contaminated & eligible),
            "excluded_ids": sorted(contaminated & eligible),
            "skipped": {task: list(skipped[task]["reasons"])
                        for task in sorted(skipped)},
            "rows": [{
                "id": task,
                "reward": scored[task]["reward"],
                "failure_category": scored[task].get("failure_category"),
                "resolved_image_digest": scored[task]["resolved_image_digest"],
                "artifacts": [dict(a) for a in scored[task]["artifacts"]],
            } for task in denominator],
        },
    }
