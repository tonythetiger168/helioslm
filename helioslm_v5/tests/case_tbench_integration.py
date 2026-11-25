"""Synthetic, non-executing oracles for Phase 2C decontamination/integration."""

import copy

from helioslm_v5.eval import tbench_integration as tbi
from helioslm_v5.tests.case_common import _pass


def _inventory():
    rows = []
    for name, cpus in (("alpha", 1), ("beta", 1), ("gamma", 2)):
        rows.append({
            "id": name,
            "status": "eligible_static",
            "skipped_reasons": [],
            "os": "linux",
            "docker_image": f"fixture/{name}:1",
            "resources": {"cpus": cpus, "memory_mb": 2048},
        })
    rows.append({
        "id": "delta",
        "status": "skipped",
        "skipped_reasons": ["multi_container"],
        "os": "linux",
        "docker_image": "fixture/delta:1",
        "resources": {"cpus": 1, "memory_mb": 2048},
    })
    return {
        "schema": "tbench-inventory/1",
        "mode": "native",
        "task_set": "terminal-bench-synthetic-fixture",
        "source_commit": "a" * 40,
        "tasks_tree": "b" * 40,
        "task_set_sha256": "c" * 64,
        "rows": rows,
    }


def _mixture(matches=()):
    return {
        "schema": "training-mixture/1",
        "name": "synthetic-mixture-v1",
        "sha256": "d" * 64,
        "method": "ngram-13-blake2b8",
        "matches": [
            {"task": task, "evidence_sha256": "e" * 64} for task in matches
        ],
    }


def _decontam(matches=()):
    return tbi.build_decontamination_report(_inventory(), _mixture(matches))


def _run_row(task, reward, **overrides):
    row = {
        "id": task,
        "reward": reward,
        "verifier_status": "completed",
        "failure_category": None if reward >= 1 else "wrong_answer",
        "resolved_image_digest": "sha256:" + "f" * 64,
        "artifacts": [
            {"path": "result.json", "sha256": "1" * 64},
            {"path": "verifier/stdout.log", "sha256": "2" * 64},
        ],
    }
    row.update(overrides)
    return row


def _run(rows=None, skipped=(), identity=None):
    report = {
        "schema": "tbench-native-run/1",
        "task_set": "terminal-bench-synthetic-fixture",
        "source_commit": "a" * 40,
        "tasks_tree": "b" * 40,
        "task_set_sha256": "c" * 64,
        "rows": rows if rows is not None else [
            _run_row("alpha", 1.0),
            _run_row("beta", 0.0),
            _run_row("gamma", 1.0),
        ],
        "skipped": list(skipped),
    }
    if identity:
        report.update(identity)
    return report


def _component(run=None, decontam=None, evidence_type="schema_fixture"):
    return tbi.build_terminal_component(
        run or _run(), _inventory(), _decontam() if decontam is None else decontam,
        model="synthetic-candidate", checkpoint_sha256="0" * 64,
        harbor_identity="harbor synthetic-runtime api-pin-90f2ee3c",
        evidence_type=evidence_type)


def _expect_error(fn, needle):
    try:
        fn()
    except (tbi.TbenchIntegrationError, ValueError) as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected an error containing {needle!r}")


def test_decontamination_ledger_keeps_every_task():
    report = _decontam(["beta"])
    assert report["schema"] == "tbench-decontam/1"
    assert report["n_tasks"] == 4 and report["n_contaminated"] == 1
    assert report["excluded_ids"] == ["beta"]
    assert report["mixture"]["name"] == "synthetic-mixture-v1"
    assert report["task_set_sha256"] == "c" * 64
    by_id = {row["id"]: row for row in report["rows"]}
    assert sorted(by_id) == ["alpha", "beta", "delta", "gamma"]
    assert by_id["beta"]["contaminated"] is True
    assert by_id["beta"]["excluded"] is True
    assert by_id["beta"]["exclusion_reason"] == "contaminated"
    assert by_id["beta"]["evidence_sha256"] == "e" * 64
    assert by_id["alpha"]["excluded"] is False
    assert by_id["delta"]["status"] == "skipped"

    mixture = _mixture(["beta", "beta"])
    _expect_error(lambda: tbi.build_decontamination_report(_inventory(), mixture),
                  "duplicate mixture match")
    _expect_error(lambda: _decontam(["not-a-task"]), "unknown inventory tasks")
    mixture = _mixture()
    mixture["sha256"] = "not-hex"
    _expect_error(lambda: tbi.build_decontamination_report(_inventory(), mixture),
                  "64 lowercase hexadecimal")
    mixture = _mixture()
    mixture["matches"] = [{"task": "beta"}]
    _expect_error(lambda: tbi.build_decontamination_report(_inventory(), mixture),
                  "evidence")
    print("PASS test_decontamination_ledger_keeps_every_task")


def test_component_recomputes_score_and_preserves_identity():
    component = _component()
    assert component["schema"] == "agentic-component/1"
    assert component["component"] == "terminal"
    assert component["evidence_type"] == "schema_fixture"
    # (1 + 0 + 1) / 3, recomputed from rows, not copied from anywhere.
    assert abs(component["score"] - 2 / 3) < 1e-12
    assert component["n"] == 3
    assert component["task_set_sha256"] == "c" * 64
    detail = component["tbench"]
    assert detail["n_eligible"] == 3
    assert detail["n_contaminated"] == 0
    assert detail["source_commit"] == "a" * 40
    assert detail["tasks_tree"] == "b" * 40
    assert [row["id"] for row in detail["rows"]] == ["alpha", "beta", "gamma"]

    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 0.0)],
               skipped=[{"id": "beta", "reasons": ["contaminated"]}])
    component = _component(run, _decontam(["beta"]))
    assert component["n"] == 2 and component["score"] == 0.5
    detail = component["tbench"]
    assert detail["n_eligible"] == 3
    assert detail["n_contaminated"] == 1
    assert detail["excluded_ids"] == ["beta"]
    assert detail["skipped"] == {"beta": ["contaminated"]}
    print("PASS test_component_recomputes_score_and_preserves_identity")


def test_accounting_hard_failures():
    # An eligible task absent from both rows and skipped is a hard failure,
    # never a silent zero.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 1.0)])
    _expect_error(lambda: _component(run), "missing from the run accounting")
    # Contaminated tasks must be skipped with the stable reason, not scored.
    _expect_error(lambda: _component(_run(), _decontam(["beta"])),
                  "must not be scored")
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 0.0)],
               skipped=[{"id": "gamma", "reasons": ["image_pull_or_build_failed"]}])
    _expect_error(lambda: _component(run, _decontam(["gamma"])),
                  "must be skipped with")
    # Scoring a task that is not eligible (or otherwise diverging from the
    # non-contaminated eligible denominator) is a hard failure.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 1.0),
                     _run_row("gamma", 1.0), _run_row("delta", 1.0)])
    _expect_error(lambda: _component(run), "scored rows must cover")
    # No task may be both scored and skipped.
    run = _run(skipped=[{"id": "beta", "reasons": ["contaminated"]}])
    _expect_error(lambda: _component(run), "both scored and skipped")
    # Skipped rows need stable reasons from the taxonomy.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 1.0)],
               skipped=[{"id": "gamma", "reasons": ["ran_out_of_budget"]}])
    _expect_error(lambda: _component(run), "unstable reason")
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 1.0)],
               skipped=[{"id": "gamma", "reasons": []}])
    _expect_error(lambda: _component(run), "explicit reasons")
    # Identity mismatches between reports are hard failures.
    run = _run(identity={"task_set_sha256": "9" * 64})
    _expect_error(lambda: _component(run), "does not match the verified inventory")
    decontam = _decontam()
    decontam["source_commit"] = "9" * 40
    _expect_error(lambda: _component(_run(), decontam),
                  "does not match the verified inventory")
    # All-contaminated full split leaves nothing to score.
    run = _run(rows=[], skipped=[
        {"id": task, "reasons": ["contaminated"]}
        for task in ("alpha", "beta", "gamma")])
    _expect_error(lambda: _component(run, _decontam(["alpha", "beta", "gamma"])),
                  "no scorable tasks")
    print("PASS test_accounting_hard_failures")


def test_run_row_validation_hard_failures():
    def component_with(**changes):
        run = _run()
        run["rows"][1] = _run_row("beta", changes.pop("reward", 0.0), **changes)
        return _component(run)

    _expect_error(lambda: component_with(reward=float("nan")), "finite number")
    _expect_error(lambda: component_with(reward=True), "finite number")
    _expect_error(lambda: component_with(verifier_status="pending"), "completed")
    _expect_error(lambda: component_with(reward=0.0, failure_category="oops"),
                  "stable failure category")
    _expect_error(lambda: component_with(reward=1.0, failure_category="wrong_answer"),
                  "passed but still declares")
    _expect_error(lambda: component_with(resolved_image_digest="fixture/beta:1"),
                  "resolved sha256 image digest")
    _expect_error(lambda: component_with(artifacts=[]), "hashed raw artifacts")
    bad_artifacts = [{"path": "result.json", "sha256": "not-hex"}]
    _expect_error(lambda: component_with(artifacts=bad_artifacts),
                  "artifact sha256")
    dup_artifacts = [{"path": "result.json", "sha256": "1" * 64}] * 2
    _expect_error(lambda: component_with(artifacts=dup_artifacts),
                  "duplicate run row artifact")

    run = _run()
    run["rows"].append(_run_row("alpha", 1.0))
    _expect_error(lambda: _component(run), "duplicate run row")
    run = _run()
    run["rows"][1]["id"] = "not-a-task"
    _expect_error(lambda: _component(run), "absent from the inventory")
    _expect_error(lambda: _component(evidence_type="oracle"),
                  "model_run or schema_fixture")
    print("PASS test_run_row_validation_hard_failures")


def test_tbench_integration_suite():
    test_decontamination_ledger_keeps_every_task()
    test_component_recomputes_score_and_preserves_identity()
    test_accounting_hard_failures()
    test_run_row_validation_hard_failures()
    _pass("test_tbench_integration_suite",
          "Terminal-Bench Phase 2C: decontamination ledger keeps every task "
          "with explicit exclusions, native run accounting covers all eligible "
          "rows exactly once, the terminal score is recomputed from rows, and "
          "identity mismatches are hard failures; synthetic fixtures only")


if __name__ == "__main__":
    test_tbench_integration_suite()
