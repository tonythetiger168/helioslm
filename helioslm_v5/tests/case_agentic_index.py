"""Agentic Index Phase 3: math, provenance and reproducibility pins.

The isolated fixture is deliberately ``schema_fixture``: its 0.625 is a
hand-checkable aggregation example, never a HeliosLM model baseline.
"""
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from helioslm_v5.eval.agentic_index import (
    SCHEMA_ID, WEIGHTS, AgenticIndexError, build_index, canonical_sha256,
    generate, load_components, render_index)
from helioslm_v5.tests.case_common import _pass

_REPO = Path(__file__).resolve().parent.parent.parent
_SCORES = {"terminal": 0.50, "swe": 0.75, "long_horizon": 0.60,
           "tool_recovery": 0.90, "hdl": 0.40}
_NS = {"terminal": 50, "swe": 40, "long_horizon": 5,
       "tool_recovery": 100, "hdl": 20}


def _fixture(root):
    root = Path(root)
    (root / "artifacts").mkdir(parents=True)
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    checkpoint = digest("agentic-index-schema-fixture-checkpoint-v1")
    for name, score in _SCORES.items():
        source = {"schema": "agentic-index-fixture-source/1", "component": name,
                  "score": score, "n": _NS[name],
                  "note": "Schema/math fixture only; not a model run."}
        source_bytes = (json.dumps(source, sort_keys=True, indent=2) + "\n").encode()
        source_rel = f"artifacts/{name}.json"
        (root / source_rel).write_bytes(source_bytes)
        component = {
            "schema": "agentic-component/1", "component": name,
            "model": "schema-math-fixture", "checkpoint_sha256": checkpoint,
            "task_set": f"{name}-schema-fixture-v1",
            "task_set_sha256": digest(f"{name}-schema-fixture-v1"),
            "score": score, "n": _NS[name], "evidence_type": "schema_fixture",
            "verification_tier": "schema-fixture",
            "source": {"artifact": source_rel,
                       "sha256": hashlib.sha256(source_bytes).hexdigest()},
        }
        (root / f"{name}.json").write_text(
            json.dumps(component, sort_keys=True, indent=2) + "\n")
    return root


def _expect_loud(fn, needle):
    try:
        fn()
    except AgenticIndexError as exc:
        assert needle in str(exc), f"{needle!r} not in {exc}"
        return
    raise AssertionError(f"expected AgenticIndexError containing {needle!r}")


def test_metric_order_and_cli_reproduction():
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td) / "results")
        # Reversed input must not affect component order, hash or total.
        index = build_index(list(reversed(load_components(fixture))), fixture)
        assert index["schema"] == SCHEMA_ID
        assert index["agentic_index"] == 0.625
        # .30*.50 + .30*.75 + .20*.60 + .10*.90 + .10*.40 = .625
        assert [r["component"] for r in index["components"]] == list(WEIGHTS)
        assert [r["weighted_score"] for r in index["components"]] == [
            0.15, 0.225, 0.12, 0.09, 0.04]
        assert index["components_sha256"] == canonical_sha256(index["components"])
        assert index["evidence_type"] == "schema_fixture"
        assert index["is_model_score"] is False
        output = fixture / "agentic_index.json"
        generate(fixture, output)
        assert render_index(index) == output.read_bytes()
        checked = subprocess.run([
            sys.executable, "-m", "helioslm_v5.eval.agentic_index",
            "--results", str(fixture), "--output", str(output), "--check"],
            cwd=_REPO, capture_output=True, text=True, timeout=30,
            # 2026-11-22 (v5.274): explicit UTF-8 — text=True alone decodes
            # with the locale codec, and on a cp950 Windows machine the
            # child's banner (0xE2 lead byte, a UTF-8 em-dash) kills the
            # reader thread with UnicodeDecodeError, leaving
            # checked.stdout=None at returncode 0. CI (UTF-8 Linux) never
            # saw this; the explicit codec makes the pin portable.
            encoding="utf-8", errors="replace")
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "NOT A MODEL SCORE" in checked.stdout
    print("PASS test_metric_order_and_cli_reproduction")


def test_loud_schema_and_numeric_validation():
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td) / "results")
        raws = load_components(fixture)
        _expect_loud(lambda: build_index(raws[:-1], fixture), "missing component")
        _expect_loud(
            lambda: build_index(raws + [copy.deepcopy(raws[0])], fixture),
            "duplicate component")
        unknown = copy.deepcopy(raws)
        unknown[0]["component"] = "mystery"
        _expect_loud(lambda: build_index(unknown, fixture), "unknown component")
        for bad in (math.nan, math.inf, -math.inf, -0.01, 1.01, True, "0.5", None):
            malformed = copy.deepcopy(raws)
            malformed[0]["score"] = bad
            _expect_loud(lambda m=malformed: build_index(m, fixture),
                         "score must be a finite number")
        for bad in (0, -1, 1.5, True, "10"):
            malformed = copy.deepcopy(raws)
            malformed[0]["n"] = bad
            _expect_loud(lambda m=malformed: build_index(m, fixture),
                         "n must be a positive integer")
        for field in ("checkpoint_sha256", "task_set_sha256"):
            for bad in ("abc", "A" * 64, 7):
                malformed = copy.deepcopy(raws)
                malformed[0][field] = bad
                _expect_loud(lambda m=malformed: build_index(m, fixture),
                             "64 lowercase hexadecimal")
        malformed = copy.deepcopy(raws)
        malformed[0]["schema"] = "agentic-component/0"
        _expect_loud(lambda: build_index(malformed, fixture), "component schema")
        malformed = copy.deepcopy(raws)
        del malformed[0]["model"]
        _expect_loud(lambda: build_index(malformed, fixture), "fields mismatch")
        malformed = copy.deepcopy(raws)
        malformed[0]["surprise"] = 1
        _expect_loud(lambda: build_index(malformed, fixture), "fields mismatch")
        # Known harness self-check/oracle evidence cannot become a model score.
        malformed = copy.deepcopy(raws)
        for raw in malformed:
            raw["evidence_type"] = "model_run"
            raw["verification_tier"] = "oracle-tested"
        _expect_loud(lambda: build_index(malformed, fixture),
                     "requires verification_tier=measured")
        _expect_loud(lambda: generate(Path(td) / "missing", Path(td) / "out.json"),
                     "input not found")
    print("PASS test_loud_schema_and_numeric_validation")


def test_identity_and_source_provenance():
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td) / "results")
        for field in ("model", "checkpoint_sha256", "evidence_type"):
            raws = load_components(fixture)
            if field == "model":
                raws[1][field] = "another-model"
            elif field == "checkpoint_sha256":
                raws[1][field] = "0" * 64
            else:
                raws[1][field] = "model_run"
                raws[1]["verification_tier"] = "measured"
            _expect_loud(lambda r=raws: build_index(r, fixture),
                         "all components must share")
        source = fixture / "artifacts" / "terminal.json"
        original = source.read_bytes()
        source.write_bytes(original + b"\n")
        _expect_loud(lambda: build_index(load_components(fixture), fixture),
                     "source artifact hash mismatch")
        source.write_bytes(original)
        raws = load_components(fixture)
        raws[0]["source"]["artifact"] = "../outside.json"
        _expect_loud(lambda: build_index(raws, fixture), "normalized relative")
        raws = load_components(fixture)
        raws[0]["source"]["artifact"] = "artifacts/missing.json"
        _expect_loud(lambda: build_index(raws, fixture), "source artifact not found")
    print("PASS test_identity_and_source_provenance")


def test_cli_stale_output_is_loud_and_not_rewritten():
    with tempfile.TemporaryDirectory() as td:
        fixture = _fixture(Path(td) / "results")
        output = fixture / "agentic_index.json"
        generate(fixture, output)
        generate(fixture, output, check=True)
        output.write_text(output.read_text().replace("0.625", "0.626"))
        _expect_loud(lambda: generate(fixture, output, check=True),
                     "stale Agentic Index")
        assert "0.626" in output.read_text(), "--check must never rewrite output"
        cli = subprocess.run([
            sys.executable, "-m", "helioslm_v5.eval.agentic_index",
            "--results", str(fixture), "--output", str(output), "--check"],
            cwd=_REPO, capture_output=True, text=True, timeout=30,
            # Same cp950 guard as test_metric_order_and_cli_reproduction.
            encoding="utf-8", errors="replace")
        assert cli.returncode == 2
        assert "stale Agentic Index" in cli.stderr
        (fixture / "hdl.json").unlink()
        _expect_loud(lambda: generate(fixture, fixture / "incomplete.json"),
                     "input not found")
        assert not (fixture / "incomplete.json").exists()
    print("PASS test_cli_stale_output_is_loud_and_not_rewritten")


def test_agentic_index_suite():
    """Run all Agentic Index oracles through the mandated daily gate."""
    test_cli_stale_output_is_loud_and_not_rewritten()
    test_identity_and_source_provenance()
    test_loud_schema_and_numeric_validation()
    test_metric_order_and_cli_reproduction()
    _pass("test_agentic_index_suite",
          "Phase 3 Agentic Index: 4 oracles (0.625 math, strict provenance, "
          "schema-fixture boundary and non-writing CLI --check)")


if __name__ == "__main__":
    test_agentic_index_suite()
