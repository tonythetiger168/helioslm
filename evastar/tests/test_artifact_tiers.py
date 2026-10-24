"""T83 — M2: artifact verification-tier registry oracles.

benchmarks/verification_tiers.json gives every benchmark artifact an
honest verification tier (formalized / oracle-tested / measured /
unverified) without editing the historical files themselves (several
are hash-chained into results_sha256 records). These tests pin:

1. validator behavior on a synthetic registry: well-formed accepted;
   unknown tier / missing basis / wrong top-level keys / phantom
   entries / unregistered files all fail LOUDLY with named offenders,
2. REAL coverage: every *.json in the repo's benchmarks/ directory is
   registered, and every registry entry names a file that exists,
3. vacuity guard: deleting one entry from a copy of the real registry
   makes coverage fail — the check is not vacuous,
4. tier_of lookup works and is loud on unknown artifacts.

Run from repo root: python3 evastar/tests/test_artifact_tiers.py
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval.artifact_tiers import (  # noqa: E402
    SCHEMA_ID, TIERS, TierSchemaError, assert_full_coverage, coverage,
    load_registry, tier_of)

_BENCH = _REPO / "benchmarks"
_REGISTRY = _BENCH / "verification_tiers.json"


def _write_registry(dirpath: Path, artifacts: dict) -> Path:
    reg = {"schema": SCHEMA_ID,
           "tiers": {t: f"def {t}" for t in TIERS},
           "artifacts": artifacts}
    p = dirpath / "verification_tiers.json"
    p.write_text(json.dumps(reg), encoding="utf-8")
    return p


def _expect_loud(fn, needle: str):
    try:
        fn()
    except TierSchemaError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected TierSchemaError containing {needle!r}")


def test_validator_accepts_and_rejects():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "ok.json").write_text("{}")
        reg = _write_registry(d, {"ok.json": {"tier": "measured",
                                              "basis": "unit test"}})
        loaded = load_registry(reg)
        assert tier_of("ok.json", loaded) == "measured"

        # unknown tier
        bad = _write_registry(d, {"ok.json": {"tier": "gold",
                                              "basis": "x"}})
        _expect_loud(lambda: load_registry(bad), "gold")
        # missing basis
        bad = _write_registry(d, {"ok.json": {"tier": "measured"}})
        _expect_loud(lambda: load_registry(bad), "basis")
        # tampered top-level keys
        p = d / "verification_tiers.json"
        reg_obj = json.loads(p.read_text())
        reg_obj["extra"] = 1
        p.write_text(json.dumps(reg_obj))
        _expect_loud(lambda: load_registry(p), "exactly")
        # tampered tier ladder
        reg_obj = {"schema": SCHEMA_ID, "tiers": {"gold": "x"},
                   "artifacts": {}}
        p.write_text(json.dumps(reg_obj))
        _expect_loud(lambda: load_registry(p), "ladder")
        # unknown artifact lookup is loud
        _expect_loud(lambda: tier_of("nope.json", loaded), "nope.json")
    print("PASS test_validator_accepts_and_rejects")


def test_real_benchmarks_fully_covered():
    """The repo gate: every benchmark artifact carries a tier. Runs in
    CI on the real tree — adding a benchmarks/*.json without a registry
    entry fails this test."""
    reg = load_registry(_REGISTRY)
    assert_full_coverage(_BENCH, reg)
    # spot-check the tier assignments that matter most
    assert tier_of("nvfp4_qat_2026-10-13.json", reg) == "oracle-tested"
    assert tier_of("api_probe_luna_2026-10-04.json", reg) == "measured"
    assert tier_of("level2_plan.json", reg) == "unverified"
    n_files = len([p for p in _BENCH.glob("*.json")
                   if p.name != "verification_tiers.json"])
    assert len(reg["artifacts"]) == n_files
    print("PASS test_real_benchmarks_fully_covered")


def test_coverage_vacuity_guard():
    reg = load_registry(_REGISTRY)
    cov = coverage(_BENCH, reg)
    assert cov == {"missing": [], "phantom": []}
    # remove one entry -> coverage must fail and NAME the file
    victim = sorted(reg["artifacts"])[0]
    del reg["artifacts"][victim]
    try:
        assert_full_coverage(_BENCH, reg)
        raise AssertionError("coverage passed with a missing entry")
    except TierSchemaError as e:
        assert victim in str(e)
    # phantom entry -> also loud
    reg = load_registry(_REGISTRY)
    reg["artifacts"]["ghost_2026.json"] = {"tier": "measured",
                                           "basis": "vacuity probe"}
    try:
        assert_full_coverage(_BENCH, reg)
        raise AssertionError("coverage passed with a phantom entry")
    except TierSchemaError as e:
        assert "ghost_2026.json" in str(e)
    print("PASS test_coverage_vacuity_guard")


def main():
    test_validator_accepts_and_rejects()
    test_real_benchmarks_fully_covered()
    test_coverage_vacuity_guard()
    print("T83 PASS: 3/3")


if __name__ == "__main__":
    main()
