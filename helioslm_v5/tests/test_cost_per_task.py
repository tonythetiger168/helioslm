"""T85 — M6: $/completed-task baseline oracles.

The M6 artifact combines measured task completions, the oracle-tested
disagg cost model, and score_stream quality audit. These tests pin:

1. completion-evidence validation is loud (missing fields, non-bool
   correctness, and stale correct flags never enter the denominator),
2. the cost-front math is exact against disagg.evaluate and rejects a
   zero-completion denominator,
3. score_stream audit is complete and non-vacuous,
4. the committed artifact's results_sha256 is internally consistent and
   its inputs hash to the recorded values,
5. CLI smoke regenerates the deterministic cost side end-to-end.

Run from repo root: python3 helioslm_v5/tests/test_cost_per_task.py
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.agent.disagg import DisaggConfig, evaluate  # noqa: E402
from helioslm_v5.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from helioslm_v5.eval.cost_per_task import (  # noqa: E402
    SCHEMA_ID, CostSchemaError, attach_score_audit, canonical_sha256,
    cost_front, load_completion_records, requests_from_records,
    sha256_file)

_ARTIFACT = _REPO / "benchmarks" / "cost_per_completed_task_2026-10-13.json"
_SCRIPT = _REPO / "examples" / "cost_per_completed_task.py"


def _records():
    return [
        {"id": "task-000", "family": "a", "prompt": "abcd",
         "expected": "12", "final": "12", "completed": True},
        {"id": "task-001", "family": "b", "prompt": "abcde",
         "expected": "123", "final": "123", "completed": True},
        {"id": "task-002", "family": "c", "prompt": "abc",
         "expected": "1", "final": "2", "completed": False},
    ]


def _expect_loud(fn, needle):
    try:
        fn()
    except CostSchemaError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected CostSchemaError containing {needle!r}")


def test_completion_records_loud_validation():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "eval.json"
        path.write_text(json.dumps({"results": [
            {"family": "x", "task": "2+2", "expected": "4",
             "final": "4", "correct": True},
            {"family": "y", "task": "3+3", "expected": "6",
             "final": "7", "correct": True},  # stale flag: not completed
        ]}), encoding="utf-8")
        records = load_completion_records(path)
        assert [r["completed"] for r in records] == [True, False]

        bad_rows = [
            ({"task": "x", "expected": "1", "final": "1"}, "correct"),
            ({"task": "x", "expected": "1", "final": "1",
              "correct": "yes"}, "bool"),
            ({"task": "x", "expected": "1", "correct": True}, "final"),
        ]
        for row, needle in bad_rows:
            path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
            _expect_loud(lambda p=path: load_completion_records(p), needle)
    print("PASS test_completion_records_loud_validation")


def test_cost_front_math_and_zero_completion_gate():
    records = _records()
    requests = requests_from_records(records)
    assert [(r.prefill_tokens, r.decode_tokens) for r in requests] == [
        (4, 2), (5, 3), (3, 1)]

    configs = [DisaggConfig(1, 1), DisaggConfig(2, 1),
               DisaggConfig(2, 2)]
    got = cost_front(records, configs=configs, worker_hour_usd=1.0)
    assert got["n_records"] == 3
    assert got["n_completed"] == 2
    assert got["completion_rate"] == round(2 / 3, 6)
    assert len(got["front"]) >= 2, "front must retain a trade-off"

    first = got["front"][0]
    expected_ms, expected_ws = evaluate(requests, DisaggConfig(1, 1))
    assert first["cfg"] == "1P+1D"
    assert first["makespan"] == round(expected_ms, 6)
    assert first["worker_seconds_total"] == round(expected_ws, 6)
    want_usd = (expected_ws / 2) / 3600.0
    assert abs(first["usd_per_completed_task"] - round(want_usd, 12)) < 1e-15

    _expect_loud(
        lambda: cost_front([{**r, "completed": False} for r in records],
                           configs=configs),
        "zero completed")
    print("PASS test_cost_front_math_and_zero_completion_gate")


def test_score_audit_complete_and_loud():
    def fake_score(prompt, output):
        assert prompt and output
        return {"n_tokens": len(output), "nll_per_token": 1.25,
                "ppl": 3.49}

    audit = attach_score_audit(_records(), fake_score)
    assert audit["n_scored"] == 2  # only completed rows enter the audit
    assert audit["mean_nll_per_token"] == 1.25
    assert [r["id"] for r in audit["rows"]] == ["task-000", "task-001"]

    def bad_score(prompt, output):
        return {"n_tokens": 1, "nll_per_token": math.inf, "ppl": math.inf}

    _expect_loud(lambda: attach_score_audit(_records(), bad_score),
                 "non-finite")
    print("PASS test_score_audit_complete_and_loud")


def test_committed_artifact_provenance_and_registry():
    artifact = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema"] == SCHEMA_ID
    assert artifact["verification_tier"] == "measured"
    assert artifact["results"]["completion"] == {
        "n_records": 12, "n_completed": 12, "completion_rate": 1.0}
    assert artifact["assumptions"]["usd_is_unit_price_normalization"] is True

    inputs = artifact["inputs"]
    assert sha256_file(_REPO / inputs["completion_evidence"]["path"]) == \
        inputs["completion_evidence"]["sha256"]
    assert sha256_file(_REPO / inputs["disagg_pareto_reference"]["path"]) == \
        inputs["disagg_pareto_reference"]["sha256"]
    assert sha256_file(_REPO / inputs["score_checkpoint"]["path"]) == \
        inputs["score_checkpoint"]["sha256"]

    signed = {k: v for k, v in artifact.items() if k != "results_sha256"}
    assert canonical_sha256(signed) == artifact["results_sha256"]

    registry = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_ARTIFACT.name, registry) == "measured"
    print("PASS test_committed_artifact_provenance_and_registry")


def test_cli_smoke_regenerates_cost_side():
    committed = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cost.json"
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--out", str(out)],
            cwd=str(_REPO), capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr[-1000:]}"
        regenerated = json.loads(out.read_text(encoding="utf-8"))

    assert regenerated["schema"] == SCHEMA_ID
    assert regenerated["inputs"] == committed["inputs"]
    assert regenerated["assumptions"] == committed["assumptions"]
    assert regenerated["results"]["completion"] == \
        committed["results"]["completion"]
    assert regenerated["results"]["cost_front"] == \
        committed["results"]["cost_front"]

    audit = regenerated["results"]["quality_audit"]
    assert audit["n_scored"] == 12
    assert len(audit["rows"]) == 12
    assert 0.0 < audit["mean_nll_per_token"] < 20.0
    assert all(math.isfinite(r["nll_per_token"]) for r in audit["rows"])
    assert len(regenerated["results_sha256"]) == 64
    print("PASS test_cli_smoke_regenerates_cost_side")


def main():
    test_completion_records_loud_validation()
    test_cost_front_math_and_zero_completion_gate()
    test_score_audit_complete_and_loud()
    test_committed_artifact_provenance_and_registry()
    test_cli_smoke_regenerates_cost_side()
    print("T85 PASS: 5/5")


if __name__ == "__main__":
    main()
