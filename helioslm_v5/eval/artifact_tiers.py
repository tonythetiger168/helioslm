"""Artifact verification tiers (v5.51, M2).

The openai/math lesson, mechanized: "tests passed + a human read the
diff" and "tests passed" are different claims, so every artifact must
carry its verification tier. Rather than editing historical JSONs in
place (several are hash-chained into results_sha256 records — editing
them would break provenance), the tiers live in ONE registry:

    benchmarks/verification_tiers.json

Schema ("verification-tiers/1"):

    {
      "schema": "verification-tiers/1",
      "tiers": {"<tier>": "<definition>", ...},
      "artifacts": {
        "<filename in benchmarks/>": {"tier": <tier>, "basis": "<why>"},
        ...
      }
    }

Tiers (most -> least verified):
- formalized    : machine-checked proof of the claim (Lean-class).
                  No HeliosLM artifact holds this tier today; the tier
                  exists so the ladder has a top rung to aim at.
- oracle-tested : the numbers were produced under an enforced oracle
                  gate (bitwise gates, fixed-point checks, routing
                  monotonicity) — a failing gate would have blocked
                  the artifact.
- measured      : an honest measurement without an oracle gate —
                  probe runs, AUROC tables, wall-clock benchmarks.
- unverified    : plans, drafts, or numbers not yet reproduced.

Rules enforced by this module (and T83):
1. every *.json directly in benchmarks/ MUST have a registry entry;
2. every registry entry MUST name a file that exists on disk;
3. every tier value MUST be one of the declared tiers;
4. the registry file itself is JSON-parseable with exactly the
   declared top-level keys.

Honesty note: the tier is a CLAIM ABOUT PROCESS (how the artifact was
produced), not a guarantee the numbers are right. A "measured"
artifact is not weaker for it — it is honestly labeled.
"""
import json
from pathlib import Path

SCHEMA_ID = "verification-tiers/1"
TIERS = ("formalized", "oracle-tested", "measured", "unverified")


class TierSchemaError(ValueError):
    """Raised for any registry/coverage violation. Never swallowed."""


def load_registry(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise TierSchemaError(f"registry not found: {path}")
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TierSchemaError(f"registry is not valid JSON: {e}") from e
    if not isinstance(reg, dict) or set(reg) != {"schema", "tiers",
                                                 "artifacts"}:
        raise TierSchemaError(
            "registry must have exactly {'schema', 'tiers', 'artifacts'}")
    if reg["schema"] != SCHEMA_ID:
        raise TierSchemaError(
            f"unknown schema {reg['schema']!r} (want {SCHEMA_ID!r})")
    declared = tuple(reg["tiers"])
    if set(declared) != set(TIERS):
        raise TierSchemaError(
            f"declared tiers {sorted(declared)} != {sorted(TIERS)} — the "
            "ladder is fixed; extend TIERS in code, not in the JSON")
    if not isinstance(reg["artifacts"], dict):
        raise TierSchemaError("artifacts must be an object")
    for name, entry in reg["artifacts"].items():
        validate_entry(name, entry, declared)
    return reg


def validate_entry(name: str, entry: dict, declared=TIERS) -> None:
    if not isinstance(entry, dict) or set(entry) != {"tier", "basis"}:
        raise TierSchemaError(
            f"{name}: entry must have exactly {{'tier', 'basis'}}")
    if entry["tier"] not in declared:
        raise TierSchemaError(
            f"{name}: tier {entry['tier']!r} not in {tuple(declared)}")
    if not isinstance(entry["basis"], str) or not entry["basis"].strip():
        raise TierSchemaError(f"{name}: basis must be a non-empty string")


def coverage(benchmarks_dir, registry: dict) -> dict:
    """Cross-check the registry against the artifacts on disk.

    Returns {"missing": [...], "phantom": [...]} — missing = json files
    with no entry; phantom = entries naming files that do not exist.
    Callers decide whether to tolerate either; T83 tolerates neither.
    """
    bdir = Path(benchmarks_dir)
    if not bdir.is_dir():
        raise TierSchemaError(f"benchmarks dir not found: {bdir}")
    on_disk = {p.name for p in bdir.glob("*.json")
               if p.name != "verification_tiers.json"}
    registered = set(registry["artifacts"])
    return {"missing": sorted(on_disk - registered),
            "phantom": sorted(registered - on_disk)}


def assert_full_coverage(benchmarks_dir, registry: dict) -> None:
    cov = coverage(benchmarks_dir, registry)
    problems = []
    if cov["missing"]:
        problems.append("unregistered artifacts: "
                        + ", ".join(cov["missing"]))
    if cov["phantom"]:
        problems.append("registry entries without a file: "
                        + ", ".join(cov["phantom"]))
    if problems:
        raise TierSchemaError("benchmarks coverage incomplete — "
                              + "; ".join(problems))


def tier_of(artifact_name: str, registry: dict) -> str:
    try:
        return registry["artifacts"][artifact_name]["tier"]
    except KeyError:
        raise TierSchemaError(
            f"{artifact_name}: not in the verification-tier registry") \
            from None
