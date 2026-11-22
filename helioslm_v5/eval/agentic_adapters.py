"""Adapt native candidate-run artifacts to ``agentic-component/1``.

This module performs no model execution and manufactures no model score. It
recomputes a component metric from the rows in an existing benchmark artifact,
then writes a deterministic descriptor with source-byte provenance. Callers
must explicitly choose ``model_run`` (with ``measured`` evidence) or
``schema_fixture`` (with ``schema-fixture`` evidence).

Accepted native artifacts are the row list returned by ``run_term_bench`` /
``run_swe_bench``, the dictionary returned by ``evaluate_policy``, the
``hdl-bench/1`` dictionary returned by ``hdl_bench.score``, and a strict
``tool-recovery-run/1`` row artifact. Known harness self-check shapes are
rejected. In particular, long-horizon ``run_suite()`` output is not candidate
evidence.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from helioslm_v5.eval.agentic_index import (
    COMPONENT_SCHEMA_ID, AgenticIndexError, normalize_component,
    render_index as render_json,
)

TOOL_SCHEMA_ID = "tool-recovery-run/1"


class AgenticAdapterError(ValueError):
    """Malformed, inconsistent, or ineligible source evidence."""


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgenticAdapterError(f"{field} must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AgenticAdapterError(f"{field} must be a finite number in [0, 1]")
    return float(value)


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticAdapterError(f"{field} must be a positive integer")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise AgenticAdapterError(f"{field} must be a non-empty string")
    return value


def _rows(value, field="rows"):
    if not isinstance(value, list) or not value:
        raise AgenticAdapterError(f"{field} must be a non-empty list")
    if any(not isinstance(row, dict) for row in value):
        raise AgenticAdapterError(f"{field} entries must be objects")
    return value


def _unique_tasks(rows):
    names = [_text(row.get("task"), "row.task") for row in rows]
    if len(set(names)) != len(names):
        raise AgenticAdapterError("duplicate task id in source artifact")


def _same(actual, expected, field):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AgenticAdapterError(
            f"{field} does not match recomputed rows: {actual} != {expected}")


def _score_terminal(payload, component):
    """Mean per-task pass@1; n counts all episodes, not just tasks."""
    rows = _rows(payload)
    _unique_tasks(rows)
    scores, total_n = [], 0
    for row in rows:
        if row.get("family") != component:
            raise AgenticAdapterError(
                f"terminal/SWE row family must be {component!r}")
        if not isinstance(row.get("correct"), bool):
            raise AgenticAdapterError("row.correct must be a boolean")
        n_samples = _positive_int(row.get("n_samples"), "row.n_samples")
        score = _number(row.get("pass_at_1"), "row.pass_at_1")
        successes = score * n_samples
        if not math.isclose(successes, round(successes), rel_tol=0.0,
                            abs_tol=1e-9):
            raise AgenticAdapterError("row.pass_at_1 must be successes/n_samples")
        if n_samples == 1 and score != float(row["correct"]):
            raise AgenticAdapterError("single-sample pass_at_1 must equal correct")
        scores.append(score)
        total_n += n_samples
    return math.fsum(scores) / len(scores), total_n


def _score_long_horizon(payload):
    # Exact evaluate_policy shape: run_suite() has schema/summary/controls and
    # is intentionally rejected here as scripted harness evidence.
    fields = {"policy", "rows", "success_rate", "violation_rate",
              "false_completion_rate", "injection_rate"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise AgenticAdapterError(
            "long_horizon requires the evaluate_policy result, not run_suite")
    _text(payload["policy"], "policy")
    rows = _rows(payload["rows"])
    for row in rows:
        for field in ("success", "false_completion", "injection_followed"):
            if not isinstance(row.get(field), bool):
                raise AgenticAdapterError(f"row.{field} must be a boolean")
        if not isinstance(row.get("violations"), list):
            raise AgenticAdapterError("row.violations must be a list")
    n = len(rows)
    rates = {
        "success_rate": math.fsum(row["success"] for row in rows) / n,
        "violation_rate": math.fsum(bool(row["violations"]) for row in rows) / n,
        "false_completion_rate": math.fsum(
            row["false_completion"] for row in rows) / n,
        "injection_rate": math.fsum(row["injection_followed"] for row in rows) / n,
    }
    for field, expected in rates.items():
        _same(_number(payload[field], field), expected, field)
    return rates["success_rate"], n


def _score_tool_recovery(payload):
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "rows"}
            or payload.get("schema") != TOOL_SCHEMA_ID):
        raise AgenticAdapterError(
            f"tool_recovery requires exactly one {TOOL_SCHEMA_ID} artifact")
    rows = _rows(payload["rows"])
    _unique_tasks(rows)
    for row in rows:
        if not isinstance(row.get("correct"), bool):
            raise AgenticAdapterError("row.correct must be a boolean")
    return math.fsum(row["correct"] for row in rows) / len(rows), len(rows)


def _score_hdl(payload):
    if not isinstance(payload, dict) or payload.get("schema") != "hdl-bench/1":
        raise AgenticAdapterError("hdl requires an hdl-bench/1 candidate score")
    required = {"schema", "bench", "commit", "n_problems", "n",
                "failure_codes", "problems", "pass@1"}
    allowed = required | {"area_ratio_mean", "area_ratio_n"}
    unknown = {key for key in payload if not isinstance(key, str)
               or (key not in allowed
                   and not re.fullmatch(r"pass@[1-9][0-9]*", key))}
    missing = required - set(payload)
    if missing or unknown:
        raise AgenticAdapterError(
            f"hdl fields mismatch: missing={sorted(missing)} "
            f"unknown={sorted(map(str, unknown))}")
    _text(payload["bench"], "bench")
    commit = payload["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AgenticAdapterError("hdl commit must be 40 lowercase hex characters")
    if not isinstance(payload["failure_codes"], dict):
        raise AgenticAdapterError("failure_codes must be an object")
    rows = _rows(payload["problems"], "problems")
    if payload["n_problems"] != len(rows):
        raise AgenticAdapterError("n_problems does not match problems")
    _positive_int(payload["n_problems"], "n_problems")
    n_samples = _positive_int(payload["n"], "n")
    names = [_text(row.get("name"), "problem.name") for row in rows]
    if len(set(names)) != len(names):
        raise AgenticAdapterError("duplicate HDL problem name")
    scores = []
    for row in rows:
        if row.get("n") != n_samples:
            raise AgenticAdapterError("problem.n must match artifact n")
        correct = row.get("correct")
        if (isinstance(correct, bool) or not isinstance(correct, int)
                or not 0 <= correct <= n_samples):
            raise AgenticAdapterError("problem.correct must be in [0, n]")
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) != n_samples:
            raise AgenticAdapterError("problem.samples must contain n entries")
        if any(not isinstance(sample, dict)
               or not isinstance(sample.get("passed"), bool)
               for sample in samples):
            raise AgenticAdapterError("sample.passed must be a boolean")
        if sum(sample["passed"] for sample in samples) != correct:
            raise AgenticAdapterError("problem.correct does not match samples")
        scores.append(correct / n_samples)
    score = math.fsum(scores) / len(scores)
    _same(_number(payload["pass@1"], "pass@1"), score, "pass@1")
    return score, len(rows) * n_samples


def score_artifact(component, payload):
    """Return ``(score, scored_item_count)`` from native candidate rows."""
    if component in ("terminal", "swe"):
        return _score_terminal(payload, component)
    if component == "long_horizon":
        return _score_long_horizon(payload)
    if component == "tool_recovery":
        return _score_tool_recovery(payload)
    if component == "hdl":
        return _score_hdl(payload)
    raise AgenticAdapterError(f"unknown component: {component!r}")


def _load_json(path):
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AgenticAdapterError(f"source artifact not found: {path}") from None
    try:
        return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
            AgenticAdapterError(f"non-finite JSON number: {value}")))
    except AgenticAdapterError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AgenticAdapterError(f"invalid JSON in {path}: {exc}") from exc


def _inside(root, path, field):
    path = path.resolve()
    if path == root or root not in path.parents:
        raise AgenticAdapterError(f"{field} must be inside the results directory")
    return path


def generate_component(results_dir, component, artifact, *, model,
                       checkpoint_sha256, task_set, task_set_sha256,
                       evidence_type, output=None, check=False):
    """Adapt one source artifact and write/check its component descriptor."""
    root = Path(results_dir).resolve()
    if not root.is_dir():
        raise AgenticAdapterError(f"results directory not found: {root}")
    artifact = Path(artifact)
    source = _inside(root, artifact if artifact.is_absolute() else root / artifact,
                     "source artifact")
    payload = _load_json(source)
    score, n = score_artifact(component, payload)
    if evidence_type not in ("model_run", "schema_fixture"):
        raise AgenticAdapterError(
            "evidence_type must be model_run or schema_fixture")
    tier = "measured" if evidence_type == "model_run" else "schema-fixture"
    raw = {
        "schema": COMPONENT_SCHEMA_ID,
        "component": component,
        "model": model,
        "checkpoint_sha256": checkpoint_sha256,
        "task_set": task_set,
        "task_set_sha256": task_set_sha256,
        "score": score,
        "n": n,
        "evidence_type": evidence_type,
        "verification_tier": tier,
        "source": {
            "artifact": source.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    try:
        descriptor = normalize_component(raw, root)
    except AgenticIndexError as exc:
        raise AgenticAdapterError(str(exc)) from exc
    output = Path(output) if output is not None else Path(f"{component}.json")
    destination = _inside(
        root, output if output.is_absolute() else root / output,
        "component output")
    if destination == source:
        raise AgenticAdapterError("component output must not overwrite its source")
    rendered = render_json(descriptor)
    if check:
        try:
            actual = destination.read_bytes()
        except FileNotFoundError:
            raise AgenticAdapterError(
                f"output not found for --check: {destination}") from None
        if actual != rendered:
            raise AgenticAdapterError(
                f"stale component: regenerate {destination} without --check")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered)
    return descriptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--component", required=True,
                        choices=("terminal", "swe", "long_horizon",
                                 "tool_recovery", "hdl"))
    parser.add_argument("--artifact", required=True, type=Path,
                        help="native source artifact inside --results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task-set", required=True)
    parser.add_argument("--task-set-sha256", required=True)
    parser.add_argument("--evidence-type", required=True,
                        choices=("model_run", "schema_fixture"))
    parser.add_argument("--output", type=Path,
                        help="descriptor inside --results (default: COMPONENT.json)")
    parser.add_argument("--check", action="store_true",
                        help="verify byte-for-byte reproduction without writing")
    args = parser.parse_args(argv)
    try:
        descriptor = generate_component(
            args.results, args.component, args.artifact,
            model=args.model, checkpoint_sha256=args.checkpoint_sha256,
            task_set=args.task_set, task_set_sha256=args.task_set_sha256,
            evidence_type=args.evidence_type, output=args.output,
            check=args.check)
    except AgenticAdapterError as exc:
        parser.exit(2, f"error: {exc}\n")
    if not args.check:
        print(json.dumps(descriptor, sort_keys=True, ensure_ascii=False,
                         allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
