"""Reproducible, fixed-weight Agentic Index (roadmap Phase 3).

The index combines five *actual candidate runs* through one checkpoint
identity and one explicit provenance contract. It deliberately does not
read the scripted std-bench reference, long-horizon oracle self-check,
tool-recovery unit-test report, or HDL reference simulation: those are
harness validation artifacts, not model scores.

Each ``--results`` directory contains ``terminal.json``, ``swe.json``,
``long_horizon.json``, ``tool_recovery.json`` and ``hdl.json``. Every
component uses ``agentic-component/1`` and points to the source run
artifact whose bytes are verified before aggregation. ``schema_fixture``
is available only for schema/math tests; its output is explicitly not a
model score. A publishable model index requires ``model_run`` +
``measured`` evidence for all five components.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA_ID = "agentic-index/1"
COMPONENT_SCHEMA_ID = "agentic-component/1"
WEIGHTS = {
    "terminal": 0.30,
    "swe": 0.30,
    "long_horizon": 0.20,
    "tool_recovery": 0.10,
    "hdl": 0.10,
}
EVIDENCE_TYPES = {"model_run", "schema_fixture"}
_FIELDS = {
    "schema", "component", "model", "checkpoint_sha256", "task_set",
    "task_set_sha256", "score", "n", "evidence_type",
    "verification_tier", "source",
}
_HEX = set("0123456789abcdef")


class AgenticIndexError(ValueError):
    """A malformed or inconsistent input. Never converted to a zero."""


def canonical_sha256(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def render_index(index) -> bytes:
    return (json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise AgenticIndexError(f"{field} must be a non-empty string")
    return value


def _hash(value, field):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in _HEX for c in value)):
        raise AgenticIndexError(
            f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _load_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AgenticIndexError(f"input not found: {path}") from None
    try:
        payload = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(
            AgenticIndexError(f"non-finite JSON number: {x}")))
    except AgenticIndexError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AgenticIndexError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgenticIndexError(f"{path}: top level must be an object")
    return payload


def _source_path(results_dir, source):
    if not isinstance(source, dict) or set(source) != {"artifact", "sha256"}:
        raise AgenticIndexError(
            "source must contain exactly artifact and sha256")
    name = _text(source["artifact"], "source.artifact")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "." in rel.parts or "\\" in name:
        raise AgenticIndexError(
            "source.artifact must be a normalized relative POSIX path")
    _hash(source["sha256"], "source.sha256")
    root = Path(results_dir).resolve()
    path = (root / Path(*rel.parts)).resolve()
    if path != root and root not in path.parents:
        raise AgenticIndexError("source.artifact escapes the results directory")
    return path


def normalize_component(raw, results_dir):
    """Validate one component and verify its source artifact bytes."""
    if not isinstance(raw, dict):
        raise AgenticIndexError("component must be an object")
    missing = sorted(_FIELDS - set(raw))
    extra = sorted(set(raw) - _FIELDS)
    if missing or extra:
        raise AgenticIndexError(
            f"component fields mismatch: missing={missing} unknown={extra}")
    if raw["schema"] != COMPONENT_SCHEMA_ID:
        raise AgenticIndexError(
            f"component schema must be {COMPONENT_SCHEMA_ID}")
    component = raw["component"]
    if component not in WEIGHTS:
        raise AgenticIndexError(f"unknown component: {component!r}")
    _text(raw["model"], "model")
    _hash(raw["checkpoint_sha256"], "checkpoint_sha256")
    _text(raw["task_set"], "task_set")
    _hash(raw["task_set_sha256"], "task_set_sha256")
    score = raw["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise AgenticIndexError("score must be a finite number in [0, 1]")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AgenticIndexError("score must be a finite number in [0, 1]")
    if isinstance(raw["n"], bool) or not isinstance(raw["n"], int) or raw["n"] <= 0:
        raise AgenticIndexError("n must be a positive integer")
    if raw["evidence_type"] not in EVIDENCE_TYPES:
        raise AgenticIndexError(
            "evidence_type must be model_run or schema_fixture")
    expected_tier = ("measured" if raw["evidence_type"] == "model_run"
                     else "schema-fixture")
    if raw["verification_tier"] != expected_tier:
        raise AgenticIndexError(
            f"{raw['evidence_type']} requires verification_tier={expected_tier}")
    source_path = _source_path(results_dir, raw["source"])
    try:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        raise AgenticIndexError(f"source artifact not found: {source_path}") from None
    if digest != raw["source"]["sha256"]:
        raise AgenticIndexError(
            f"source artifact hash mismatch: {raw['source']['artifact']}")
    return dict(raw)


def build_index(components, results_dir):
    """Aggregate exactly five compatible components in fixed weight order."""
    if not isinstance(components, (list, tuple)):
        raise AgenticIndexError("components must be a list")
    rows = [normalize_component(raw, results_dir) for raw in components]
    names = [row["component"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AgenticIndexError(f"duplicate component: {duplicates}")
    missing = sorted(set(WEIGHTS) - set(names))
    if missing:
        raise AgenticIndexError(f"missing component: {missing}")
    identities = {(row["model"], row["checkpoint_sha256"],
                   row["evidence_type"], row["verification_tier"])
                  for row in rows}
    if len(identities) != 1:
        raise AgenticIndexError(
            "all components must share model, checkpoint, evidence type and tier")
    ordered = []
    for name in WEIGHTS:
        row = next(row for row in rows if row["component"] == name)
        row["weight"] = WEIGHTS[name]
        row["weighted_score"] = round(row["score"] * WEIGHTS[name], 6)
        ordered.append(row)
    model, checkpoint, evidence_type, tier = next(iter(identities))
    index = {
        "schema": SCHEMA_ID,
        "model": model,
        "checkpoint_sha256": checkpoint,
        "evidence_type": evidence_type,
        "verification_tier": tier,
        "is_model_score": evidence_type == "model_run",
        "weights": dict(WEIGHTS),
        "components": ordered,
        "agentic_index": round(math.fsum(
            row["score"] * WEIGHTS[row["component"]] for row in ordered), 6),
    }
    index["components_sha256"] = canonical_sha256(ordered)
    return index


def load_components(results_dir):
    return [_load_json(Path(results_dir) / f"{name}.json") for name in WEIGHTS]


def generate(results_dir, output, check=False):
    results_dir, output = Path(results_dir), Path(output)
    index = build_index(load_components(results_dir), results_dir)
    rendered = render_index(index)
    if check:
        try:
            actual = output.read_bytes()
        except FileNotFoundError:
            raise AgenticIndexError(f"output not found for --check: {output}") from None
        if actual != rendered:
            raise AgenticIndexError(
                f"stale Agentic Index: regenerate {output} without --check")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rendered)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path,
                        help="directory containing the five component JSON files")
    parser.add_argument("--output", required=True, type=Path,
                        help="Agentic Index JSON to write or verify")
    parser.add_argument("--check", action="store_true",
                        help="verify byte-for-byte reproduction without writing")
    args = parser.parse_args(argv)
    try:
        index = generate(args.results, args.output, args.check)
    except AgenticIndexError as exc:
        parser.exit(2, f"agentic-index: {exc}\n")
    label = ("model score" if index["is_model_score"]
             else "SCHEMA FIXTURE — NOT A MODEL SCORE")
    action = "verified" if args.check else "wrote"
    print(f"{action} {args.output}: Agentic Index={index['agentic_index']:.6f} "
          f"({label})")


if __name__ == "__main__":
    main()
