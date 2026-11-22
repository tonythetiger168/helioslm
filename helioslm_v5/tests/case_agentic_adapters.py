"""Agentic Index native-artifact adapters, with schema-fixture evidence only."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from helioslm_v5.eval.agentic_adapters import (
    AgenticAdapterError, generate_component, score_artifact)
from helioslm_v5.eval.agentic_index import (
    AgenticIndexError, build_index, load_components)
from helioslm_v5.tests.case_common import _pass

_REPO = Path(__file__).resolve().parent.parent.parent
_COMPONENTS = ("terminal", "swe", "long_horizon", "tool_recovery", "hdl")


def _row(component, task, correct, pass_at_1, n_samples=1):
    return {"family": component, "task": task, "correct": correct,
            "pass_at_1": pass_at_1, "n_samples": n_samples}


def _payloads():
    long_rows = [
        {"success": True, "violations": [], "false_completion": False,
         "injection_followed": False},
        {"success": False, "violations": ["budget"], "false_completion": True,
         "injection_followed": True},
    ]
    hdl_rows = []
    for name in ("alu", "counter"):
        samples = [{"passed": True}, {"passed": False}]
        hdl_rows.append({"name": name, "n": 2, "correct": 1,
                         "samples": samples})
    return {
        # Mean pass@1 = (1 + .5) / 2 = .75 across three episodes.
        "terminal": [_row("terminal", "logs", True, 1.0),
                     _row("terminal", "json", False, 0.5, 2)],
        "swe": [_row("swe", "repair-a", True, 1.0),
                _row("swe", "repair-b", False, 0.0)],
        "long_horizon": {
            "policy": "schema-policy-fixture", "rows": long_rows,
            "success_rate": 0.5, "violation_rate": 0.5,
            "false_completion_rate": 0.5, "injection_rate": 0.5,
        },
        "tool_recovery": {
            "schema": "tool-recovery-run/1",
            "rows": [{"task": f"call-{i}", "correct": i < 3}
                     for i in range(4)],
        },
        "hdl": {
            "schema": "hdl-bench/1", "bench": "verilog-eval-v2/spec-to-rtl",
            "commit": "a" * 40, "n_problems": 2, "n": 2,
            "failure_codes": {"sim_fail": 2}, "problems": hdl_rows,
            "pass@1": 0.5,
        },
    }


def _fixture(root):
    root = Path(root)
    (root / "artifacts").mkdir(parents=True)
    for component, payload in _payloads().items():
        (root / "artifacts" / f"{component}.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
    return root


def _generate(root, component):
    digest = hashlib.sha256(f"{component}-task-set-v1".encode()).hexdigest()
    return generate_component(
        root, component, f"artifacts/{component}.json",
        model="schema-adapter-fixture", checkpoint_sha256="b" * 64,
        task_set=f"{component}-task-set-v1", task_set_sha256=digest,
        evidence_type="schema_fixture")


def _generate_all(root):
    return [_generate(root, component) for component in _COMPONENTS]


def _expect_loud(fn, needle, error=AgenticAdapterError):
    try:
        fn()
    except error as exc:
        assert needle in str(exc), f"{needle!r} not in {exc}"
        return
    raise AssertionError(f"expected {error.__name__} containing {needle!r}")


def test_native_artifact_math_and_index():
    payloads = _payloads()
    assert score_artifact("terminal", payloads["terminal"]) == (0.75, 3)
    assert score_artifact("swe", payloads["swe"]) == (0.5, 2)
    assert score_artifact("long_horizon", payloads["long_horizon"]) == (0.5, 2)
    assert score_artifact("tool_recovery", payloads["tool_recovery"]) == (0.75, 4)
    assert score_artifact("hdl", payloads["hdl"]) == (0.5, 4)
    with tempfile.TemporaryDirectory() as td:
        root = _fixture(Path(td) / "results")
        rows = _generate_all(root)
        assert [row["verification_tier"] for row in rows] == ["schema-fixture"] * 5
        index = build_index(load_components(root), root)
        # .30*.75 + .30*.50 + .20*.50 + .10*.75 + .10*.50 = .60
        assert index["agentic_index"] == 0.6
        assert index["is_model_score"] is False
        for component in _COMPONENTS:
            before = (root / f"{component}.json").read_bytes()
            _generate(root, component)
            assert (root / f"{component}.json").read_bytes() == before
    print("PASS test_native_artifact_math_and_index")


def test_oracle_shapes_and_row_integrity():
    payloads = _payloads()
    oracle = {"schema": "long-horizon-bench/1", "summary": {"success_rate": 1.0},
              "rows": [], "controls": []}
    _expect_loud(lambda: score_artifact("long_horizon", oracle),
                 "not run_suite")
    _expect_loud(lambda: score_artifact("tool_recovery", {"tests": 20, "failed": 0}),
                 "tool-recovery-run/1")
    _expect_loud(lambda: score_artifact("hdl", {"passed": 20, "failures": []}),
                 "hdl-bench/1")
    _expect_loud(lambda: score_artifact("terminal", {"rows": payloads["terminal"]}),
                 "rows must be a non-empty list")

    bad = copy.deepcopy(payloads["terminal"])
    bad[1]["task"] = "logs"
    _expect_loud(lambda: score_artifact("terminal", bad), "duplicate task")
    bad = copy.deepcopy(payloads["terminal"])
    bad[0]["pass_at_1"] = True
    _expect_loud(lambda: score_artifact("terminal", bad), "finite number")
    bad = copy.deepcopy(payloads["terminal"])
    bad[0]["n_samples"] = 0
    _expect_loud(lambda: score_artifact("terminal", bad), "positive integer")
    bad = copy.deepcopy(payloads["terminal"])
    bad[0]["pass_at_1"] = 0.0
    _expect_loud(lambda: score_artifact("terminal", bad), "must equal correct")

    bad = copy.deepcopy(payloads["long_horizon"])
    bad["success_rate"] = 1.0
    _expect_loud(lambda: score_artifact("long_horizon", bad), "recomputed rows")
    bad = copy.deepcopy(payloads["long_horizon"])
    bad["rows"][0]["success"] = 1
    _expect_loud(lambda: score_artifact("long_horizon", bad), "must be a boolean")
    bad = copy.deepcopy(payloads["tool_recovery"])
    bad["rows"][0]["correct"] = 1
    _expect_loud(lambda: score_artifact("tool_recovery", bad), "must be a boolean")

    bad = copy.deepcopy(payloads["hdl"])
    bad["problems"][0]["correct"] = 2
    _expect_loud(lambda: score_artifact("hdl", bad), "does not match samples")
    bad = copy.deepcopy(payloads["hdl"])
    bad["pass@1"] = 1.0
    _expect_loud(lambda: score_artifact("hdl", bad), "does not match recomputed")
    bad = copy.deepcopy(payloads["hdl"])
    bad["problems"][1]["name"] = "alu"
    _expect_loud(lambda: score_artifact("hdl", bad), "duplicate HDL")
    print("PASS test_oracle_shapes_and_row_integrity")


def test_cli_roundtrip_and_stale_output():
    with tempfile.TemporaryDirectory() as td:
        root = _fixture(Path(td) / "results")
        command = [
            sys.executable, "-m", "helioslm_v5.eval.agentic_adapters",
            "--results", str(root), "--component", "terminal",
            "--artifact", "artifacts/terminal.json",
            "--model", "schema-adapter-fixture",
            "--checkpoint-sha256", "b" * 64,
            "--task-set", "terminal-task-set-v1",
            "--task-set-sha256", hashlib.sha256(
                b"terminal-task-set-v1").hexdigest(),
            "--evidence-type", "schema_fixture",
        ]
        # Explicit UTF-8 on every CLI subprocess (v5.274 precedent in
        # case_agentic_index.py): text=True alone decodes with the locale
        # codec, and on a cp950 Windows host a UTF-8 lead byte in the
        # child's output kills the reader thread, leaving stdout=None at
        # returncode 0. UTF-8 CI never saw it; the codec makes it portable.
        made = subprocess.run(command, cwd=_REPO, capture_output=True,
                              text=True, timeout=30,
                              encoding="utf-8", errors="replace")
        assert made.returncode == 0, made.stderr
        checked = subprocess.run(command + ["--check"], cwd=_REPO,
                                 capture_output=True, text=True, timeout=30,
                                 encoding="utf-8", errors="replace")
        assert checked.returncode == 0, checked.stderr
        output = root / "terminal.json"
        output.write_text(output.read_text().replace('"score": 0.75',
                                                      '"score": 0.5'))
        stale = subprocess.run(command + ["--check"], cwd=_REPO,
                               capture_output=True, text=True, timeout=30,
                               encoding="utf-8", errors="replace")
        assert stale.returncode == 2
        assert "stale component" in stale.stderr
        assert '"score": 0.5' in output.read_text(), "--check rewrote output"
    print("PASS test_cli_roundtrip_and_stale_output")


def test_source_tamper_paths_and_metadata():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = _fixture(base / "results")
        _generate_all(root)
        source = root / "artifacts" / "terminal.json"
        original = source.read_bytes()
        source.write_bytes(original + b"\n")
        _expect_loud(lambda: build_index(load_components(root), root),
                     "source artifact hash mismatch", AgenticIndexError)
        source.write_bytes(original)
        outside = base / "outside.json"
        outside.write_text("[]")
        _expect_loud(lambda: generate_component(
            root, "terminal", outside, model="fixture",
            checkpoint_sha256="b" * 64, task_set="tasks",
            task_set_sha256="c" * 64, evidence_type="schema_fixture"),
            "inside the results directory")
        _expect_loud(lambda: generate_component(
            root, "terminal", "artifacts/terminal.json", model="fixture",
            checkpoint_sha256="bad", task_set="tasks",
            task_set_sha256="c" * 64, evidence_type="schema_fixture"),
            "64 lowercase hexadecimal")
        _expect_loud(lambda: generate_component(
            root, "terminal", "artifacts/terminal.json", model="fixture",
            checkpoint_sha256="b" * 64, task_set="tasks",
            task_set_sha256="c" * 64, evidence_type="oracle"),
            "model_run or schema_fixture")
        assert not (root / "missing.json").exists()
    print("PASS test_source_tamper_paths_and_metadata")


def test_long_horizon_evaluate_policy_shape():
    """Real evaluate_policy rows (int violation counts) score identically.

    The adapter contract promised "the dictionary returned by
    evaluate_policy", but until the first measured candidate run
    (2026-11-23) only list-form fixture rows had ever been scored, and the
    real evaluator emits ``violations`` as an int count. Pin both accepted
    forms, the recomputed rates, and the rejections."""
    rows = [
        {"scenario": "a", "policy": "p", "success": True, "finished": True,
         "final_answer": "x", "files_ok": True, "false_completion": False,
         "budget_exhausted": False, "violations": 0,
         "injection_followed": False, "tool_errors": 0, "parse_errors": 0,
         "steps": 3, "replay_ok": True, "trajectory_sha256": "d" * 64,
         "wall_s": 0.1},
        {"scenario": "b", "policy": "p", "success": False, "finished": False,
         "final_answer": None, "files_ok": False, "false_completion": True,
         "budget_exhausted": True, "violations": 2,
         "injection_followed": False, "tool_errors": 1, "parse_errors": 4,
         "steps": 8, "replay_ok": True, "trajectory_sha256": "e" * 64,
         "wall_s": 0.2},
    ]
    payload = {"policy": "p", "rows": rows, "success_rate": 0.5,
               "violation_rate": 0.5, "false_completion_rate": 0.5,
               "injection_rate": 0.0}
    assert score_artifact("long_horizon", payload) == (0.5, 2)
    bad = copy.deepcopy(payload)
    bad["rows"][0]["violations"] = True
    _expect_loud(lambda: score_artifact("long_horizon", bad),
                 "list or a non-negative count")
    bad = copy.deepcopy(payload)
    bad["rows"][0]["violations"] = -1
    _expect_loud(lambda: score_artifact("long_horizon", bad),
                 "non-negative")
    print("PASS test_long_horizon_evaluate_policy_shape")


def test_agentic_adapters_suite():
    """Run all native-artifact adapter oracles in the mandated daily gate."""
    test_source_tamper_paths_and_metadata()
    test_cli_roundtrip_and_stale_output()
    test_oracle_shapes_and_row_integrity()
    test_native_artifact_math_and_index()
    test_long_horizon_evaluate_policy_shape()
    _pass("test_agentic_adapters_suite",
          "Phase 3 adapters: native row recomputation, oracle rejection, "
          "deterministic descriptors, non-writing CLI --check, and the "
          "real evaluate_policy row shape")


if __name__ == "__main__":
    test_agentic_adapters_suite()
