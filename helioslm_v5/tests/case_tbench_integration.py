"""Phase 2C decontamination and result-integration oracles (synthetic only)."""
from helioslm_v5.eval.tbench_integration import (
    build_decontamination_report, build_terminal_component)
from helioslm_v5.eval.tbench_harbor import HarborBridgeError

_SOURCE_COMMIT = "a" * 40
_TASKS_TREE = "b" * 40
_TASK_SET_SHA256 = "c" * 64


def _inventory():
    rows = []
    for name in ("alpha", "beta", "gamma"):
        rows.append({"task": name, "status": "eligible_static",
                     "skipped_reasons": []})
    rows.append({"task": "delta", "status": "skipped_static",
                 "skipped_reasons": ["multi_container"]})
    return {
        "schema": "tbench-inventory/1",
        "task_set": "terminal-bench-2.1",
        "source_commit": _SOURCE_COMMIT,
        "tasks_tree": _TASKS_TREE,
        "task_set_sha256": _TASK_SET_SHA256,
        "n_tasks": 4,
        "tasks": rows,
    }


def _mixture(matches=("beta",)):
    return {
        "schema": "training-mixture/1",
        "name": "helioslm-sft-mixture-v3",
        "sha256": "d" * 64,
        "method": "ngram-13-blake2b8",
        "matches": [{"task": task, "evidence_sha256": "e" * 64}
                    for task in matches],
    }


def _run_row(task, reward, **overrides):
    row = {
        "task": task,
        "reward": reward,
        "verifier_status": "completed",
        "resolved_image_digest": "sha256:" + "f" * 64,
        "artifacts": [{"path": "result.json", "sha256": "1" * 64},
                      {"path": "verifier.log", "sha256": "2" * 64}],
    }
    if reward < 1.0:
        row["failure_category"] = "wrong_answer"
    row.update(overrides)
    return row


def _run(rows=(), skipped=(), identity_overrides=None):
    report = {
        "schema": "tbench-native-run/1",
        "task_set": "terminal-bench-2.1",
        "source_commit": _SOURCE_COMMIT,
        "tasks_tree": _TASKS_TREE,
        "task_set_sha256": _TASK_SET_SHA256,
        "harbor_identity": "harbor 90f2ee3/containerd-disposable",
        "rows": list(rows) + [{"task": task, "reasons": list(reasons)}
                              for task, reasons in skipped],
    }
    if identity_overrides:
        report.update(identity_overrides)
    return report


def _decontam(matches=("beta",)):
    return build_decontamination_report(_inventory(), _mixture(matches))


def _component(run, decontam, evidence_type="schema_fixture"):
    return build_terminal_component(
        run, _inventory(), decontam,
        model="schema-integration-fixture", checkpoint_sha256="0" * 64,
        harbor_identity="harbor 90f2ee3/containerd-disposable",
        evidence_type=evidence_type)


def _expect_loud(fn, needle):
    try:
        fn()
    except HarborBridgeError as exc:
        assert needle in str(exc), f"{needle!r} not in {exc}"
        return
    raise AssertionError(f"expected HarborBridgeError containing {needle!r}")


def test_decontamination_ledger_keeps_every_task():
    report = _decontam()
    assert report["schema"] == "tbench-decontam/1"
    assert report["n_tasks"] == 4
    assert report["n_contaminated"] == 1
    assert report["excluded_ids"] == ["beta"]
    assert [row["task"] for row in report["rows"]] == [
        "alpha", "beta", "gamma", "delta"]
    beta = report["rows"][1]
    assert beta["excluded"] is True
    assert beta["exclusion_reason"] == "contaminated"
    assert beta["evidence_sha256"] == "e" * 64
    assert "excluded" not in report["rows"][0]
    # Unknown matches and malformed manifests are hard failures.
    _expect_loud(lambda: _decontam(("alpha", "alpha")), "duplicate mixture match")
    _expect_loud(lambda: _decontam(("zzz",)), "absent from the inventory")
    bad = _mixture()
    bad["schema"] = "training-mixture/0"
    _expect_loud(lambda: build_decontamination_report(_inventory(), bad),
                 "training-mixture/1")
    bad = _mixture()
    bad["sha256"] = "XYZ"
    _expect_loud(lambda: build_decontamination_report(_inventory(), bad),
                 "64 lowercase hexadecimal")
    print("PASS test_decontamination_ledger_keeps_every_task")


def test_component_recomputes_score_and_preserves_identity():
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 0.0)],
               skipped=[("beta", ["contaminated"])])
    component = _component(run, _decontam())
    assert component["schema"] == "agentic-component/1"
    assert component["component"] == "terminal"
    assert component["score"] == 0.5
    assert component["n"] == 2
    assert component["evidence_type"] == "schema_fixture"
    assert component["verification_tier"] == "schema-fixture"
    assert component["task_set"] == "terminal-bench-2.1"
    assert component["task_set_sha256"] == _TASK_SET_SHA256
    tbench = component["tbench"]
    assert tbench["source_commit"] == _SOURCE_COMMIT
    assert tbench["tasks_tree"] == _TASKS_TREE
    assert tbench["n_eligible"] == 3
    assert tbench["n_contaminated"] == 1
    assert tbench["excluded_ids"] == ["beta"]
    assert tbench["skipped"] == {"beta": ["contaminated"]}
    assert [row["task"] for row in tbench["rows"]] == ["alpha", "gamma"]
    assert tbench["rows"][1]["failure_category"] == "wrong_answer"
    # An uncontaminated full-coverage run recomputes (1 + 0 + 1) / 3.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 0.0),
                     _run_row("gamma", 1.0)])
    component = _component(run, _decontam(()))
    assert component["score"] == 2 / 3
    assert component["n"] == 3
    print("PASS test_component_recomputes_score_and_preserves_identity")


def test_accounting_hard_failures():
    decontam = _decontam()
    # An eligible task missing from the run accounting is a hard failure.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 0.0)])
    _expect_loud(lambda: _component(run, decontam),
                 "missing from the run accounting")
    # A contaminated task must never be scored.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("beta", 0.0),
                     _run_row("gamma", 1.0)])
    _expect_loud(lambda: _component(run, decontam), "must not be scored")
    # A contaminated eligible task must be skipped with 'contaminated'.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", ["agent_timeout"])])
    _expect_loud(lambda: _component(run, decontam),
                 "must be skipped with")
    # Scored rows must not cover a non-eligible task.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("delta", 0.0),
                     _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated"])])
    _expect_loud(lambda: _component(run, decontam),
                 "scored rows must cover")
    # A task that is both scored and skipped is a duplicate row.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("alpha", 0.0),
                     _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated"])])
    _expect_loud(lambda: _component(run, decontam), "duplicate run row")
    # Skipped rows need stable taxonomy reasons.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated", "mystery"])])
    _expect_loud(lambda: _component(run, decontam), "unknown skip reason")
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", [])])
    _expect_loud(lambda: _component(run, decontam), "non-empty reasons")
    # Identity mismatches between reports are hard failures.
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated"])],
               identity_overrides={"task_set_sha256": "9" * 64})
    _expect_loud(lambda: _component(run, decontam),
                 "does not match the inventory identity")
    bad_decontam = _decontam()
    bad_decontam["source_commit"] = "9" * 40
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated"])])
    _expect_loud(lambda: _component(run, bad_decontam),
                 "does not match the inventory identity")
    # Excluding every eligible task leaves no scorable denominator.
    run = _run(skipped=[("alpha", ["contaminated"]),
                        ("beta", ["contaminated"]),
                        ("gamma", ["contaminated"])])
    _expect_loud(lambda: _component(run, _decontam(("alpha", "beta", "gamma"))),
                 "no scorable tasks remain")
    print("PASS test_accounting_hard_failures")


def test_run_row_validation_hard_failures():
    decontam = _decontam()
    base = {"skipped": [("beta", ["contaminated"])]}

    def component_with(row):
        return _component(_run(rows=[_run_row("alpha", 1.0), row], **base),
                          decontam)

    _expect_loud(lambda: component_with(_run_row("gamma", float("nan"))),
                 "finite number")
    _expect_loud(lambda: component_with(_run_row("gamma", True)),
                 "finite number")
    _expect_loud(lambda: component_with(
        _run_row("gamma", 1.0, verifier_status="pending")),
        "verifier_status")
    _expect_loud(lambda: component_with(
        _run_row("gamma", 0.0, failure_category="mystery")),
        "failure category")
    _expect_loud(lambda: component_with(
        _run_row("gamma", 1.0, failure_category="wrong_answer")),
        "must not carry a failure category")
    _expect_loud(lambda: component_with(
        _run_row("gamma", 1.0, resolved_image_digest="latest")),
        "sha256:")
    _expect_loud(lambda: component_with(_run_row("gamma", 1.0, artifacts=[])),
                 "non-empty hashed artifacts")
    _expect_loud(lambda: component_with(_run_row(
        "gamma", 1.0,
        artifacts=[{"path": "result.json", "sha256": "ZZZ"}])),
        "64 lowercase hexadecimal")
    _expect_loud(lambda: component_with(_run_row(
        "gamma", 1.0,
        artifacts=[{"path": "result.json", "sha256": "1" * 64},
                   {"path": "result.json", "sha256": "2" * 64}])),
        "duplicate artifact path")
    _expect_loud(lambda: component_with(_run_row("zzz", 0.0)),
                 "absent from the inventory")
    run = _run(rows=[_run_row("alpha", 1.0), _run_row("gamma", 1.0)],
               skipped=[("beta", ["contaminated"])])
    _expect_loud(lambda: _component(run, decontam, evidence_type="oracle"),
                 "model_run or schema_fixture")
    print("PASS test_run_row_validation_hard_failures")


def test_tbench_integration_suite():
    test_decontamination_ledger_keeps_every_task()
    test_component_recomputes_score_and_preserves_identity()
    test_accounting_hard_failures()
    test_run_row_validation_hard_failures()
    print("PASS test_tbench_integration_suite: Terminal-Bench Phase 2C: "
          "decontamination ledger keeps every task with explicit exclusions, "
          "native run accounting covers all eligible rows exactly once, the "
          "terminal score is recomputed from rows, and identity mismatches "
          "are hard failures; synthetic fixtures only")


if __name__ == "__main__":
    test_tbench_integration_suite()
