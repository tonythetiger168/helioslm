"""Synthetic, non-executing oracles for Harbor task-set discovery."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from helioslm_v5.eval import tbench_adapter as tba
from helioslm_v5.tests.case_common import _pass

_REPO = Path(__file__).resolve().parent.parent.parent
_SOURCE = "a" * 40
_BASE = '''schema_version = "1.1"

[environment]
docker_image = "fixture/image:1"
os = "linux"
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
allow_internet = true
mcp_servers = []

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 900.0
'''
_INSTRUCTION = "Repair the output in /app/output.\n"
_DOCKERFILE = "FROM python:3.12-slim\nWORKDIR /app\n"
_TEST = "pip install pytest\npytest /tests/test_outputs.py\n"


def _task(root, task_id="good", config=_BASE):
    base = Path(root) / task_id
    (base / "environment").mkdir(parents=True)
    (base / "tests").mkdir()
    # 2026-11-27 (v5.277): write bytes, not text — Path.write_text uses
    # newline=None, which translates \n to \r\n on Windows and inflates
    # the on-disk task.toml from 238 to 255 bytes. The adapter hashes
    # file bytes, so a CRLF-inflated fixture mismatches the descriptor
    # digests computed from the same strings in memory. write_bytes
    # keeps the fixture byte-identical on every platform.
    (base / "task.toml").write_bytes(config.encode("utf-8"))
    (base / "instruction.md").write_bytes(_INSTRUCTION.encode("utf-8"))
    (base / "environment" / "Dockerfile").write_bytes(
        _DOCKERFILE.encode("utf-8"))
    (base / "tests" / "test.sh").write_bytes(_TEST.encode("utf-8"))
    return base


def _identity(root):
    return tba.task_set_identity(root)


def _inventory(root, mode="native"):
    _, digest, tree = _identity(root)
    return tba.inventory(root, "terminal-bench-fixture", _SOURCE, digest, tree, mode)


def _expect_error(fn, needle):
    try:
        fn()
    except tba.TaskSetError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected TaskSetError containing {needle!r}")


def test_identity_native_and_no_docker_gate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "tasks"
        _task(root)
        manifest, digest, tree = _identity(root)
        contents = {
            "good/task.toml": _BASE.encode(),
            "good/instruction.md": _INSTRUCTION.encode(),
            "good/environment/Dockerfile": _DOCKERFILE.encode(),
            "good/tests/test.sh": _TEST.encode(),
        }
        expected = [
            {"path": path, "mode": "100644",
             "sha256": hashlib.sha256(data).hexdigest()}
            for path, data in sorted(contents.items())
        ]
        canonical = json.dumps(expected, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        assert manifest == expected
        assert digest == hashlib.sha256(canonical).hexdigest()
        assert len(tree) == 40
        report = _inventory(root)
        assert report["schema"] == "tbench-inventory/1"
        assert report["summary"] == {
            "tasks": 1, "eligible_static": 1, "skipped": 0}
        row = report["rows"][0]
        assert row["status"] == "eligible_static"
        assert row["absolute_app_path"] is True
        assert row["verifier_installs_dependencies"] is True
        assert "score" not in report and "reward" not in report
        no_docker = _inventory(root, "no-docker")
        assert no_docker["summary"] == {
            "tasks": 1, "eligible_static": 0, "skipped": 1}
        assert no_docker["rows"][0]["skipped_reasons"] == [
            "absolute_path_isolation_required", "native_runtime_unavailable",
            "verifier_dependencies_unavailable"]
        (root / "good" / "new-file").write_text("tamper", encoding="utf-8")
        _expect_error(lambda: tba.inventory(
            root, "terminal-bench-fixture", _SOURCE, digest, tree),
            "task-set SHA-256 mismatch")
    print("PASS test_identity_native_and_no_docker_gate")


def test_capability_rejections_and_invalid_tasks():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "tasks"
        configs = {
            "windows": _BASE.replace('os = "linux"', 'os = "windows"'),
            "gpu": _BASE.replace("gpus = 0", "gpus = 1"),
            "separate": _BASE.replace(
                '[verifier]\ntimeout_sec = 900.0',
                '[verifier]\ntimeout_sec = 900.0\nenvironment_mode = "separate"'),
            "allowlist": _BASE.replace(
                "[agent]\ntimeout_sec = 900.0",
                '[agent]\ntimeout_sec = 900.0\nnetwork_mode = "allowlist"\n'
                'allowed_hosts = ["pypi.org"]'),
            "artifacts": 'artifacts = ["/app/output"]\n' + _BASE,
            "multi-step": _BASE + '\n[[steps]]\nname = "second"\n',
            "schema": _BASE.replace('schema_version = "1.1"',
                                    'schema_version = "1.4"'),
        }
        for task_id, config in configs.items():
            _task(root, task_id, config)
        _task(root, "compose")
        (root / "compose" / "environment" / "docker-compose.yaml").write_text(
            "services: {}\n", encoding="utf-8")
        _task(root, "mcp", _BASE.replace(
            "mcp_servers = []", 'mcp_servers = [{name = "fixture", url = "http://mcp"}]'))
        _task(root, "missing-tests")
        (root / "missing-tests" / "tests" / "test.sh").unlink()
        broken = root / "broken"
        broken.mkdir()
        (broken / "task.toml").write_text("not = [valid", encoding="utf-8")
        report = _inventory(root)
        rows = {row["id"]: row for row in report["rows"]}
        expected = {
            "windows": "unsupported_os", "gpu": "accelerator_requested",
            "separate": "separate_verifier", "allowlist": "network_policy_unsupported",
            "artifacts": "artifact_handoff", "multi-step": "multi_step",
            "schema": "unsupported_schema", "compose": "multi_container",
            "mcp": "mcp_requested", "missing-tests": "missing_required_file",
            "broken": "invalid_task",
        }
        assert set(rows) == set(expected)
        for task_id, reason in expected.items():
            assert reason in rows[task_id]["skipped_reasons"], (task_id, rows[task_id])
            assert rows[task_id]["status"] == "skipped"
        assert report["summary"]["eligible_static"] == 0
        assert report["summary"]["tasks"] == len(expected)
    print("PASS test_capability_rejections_and_invalid_tasks")


def test_symlink_is_data_not_a_host_read():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = base / "tasks"
        task = _task(root)
        secret = base / "outside-secret.txt"
        secret.write_text("DO-NOT-READ-THIS-HOST-SECRET", encoding="utf-8")
        instruction = task / "instruction.md"
        instruction.unlink()
        try:
            instruction.symlink_to(secret)
        except (OSError, NotImplementedError):
            print("SKIP symlink fixture unsupported on this host")
            return
        report = _inventory(root)
        assert report["rows"][0]["skipped_reasons"] == ["invalid_task"]
        serialized = json.dumps(report)
        assert "DO-NOT-READ-THIS-HOST-SECRET" not in serialized
        manifest, _, _ = _identity(root)
        linked = next(row for row in manifest if row["path"] == "good/instruction.md")
        assert linked["mode"] == "120000"
        assert linked["sha256"] == hashlib.sha256(str(secret).encode()).hexdigest()
    print("PASS test_symlink_is_data_not_a_host_read")


def test_cli_lists_every_skip_and_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = base / "tasks"
        _task(root, "alpha")
        _task(root, "beta")
        (root / "beta" / "tests" / "test.sh").unlink()
        _, digest, tree = _identity(root)
        command = [
            sys.executable, "-m", "helioslm_v5.eval.tbench_adapter", "list",
            "--root", str(root), "--task-set", "terminal-bench-fixture",
            "--source-commit", _SOURCE, "--expected-task-set-sha256", digest,
            "--expected-tasks-tree", tree, "--mode", "no-docker",
        ]
        output = base / "inventory.json"
        made = subprocess.run(command + ["--out", str(output)], cwd=_REPO,
                              capture_output=True, text=True, timeout=30,
                              encoding="utf-8", errors="replace")
        assert made.returncode == 0, made.stderr
        assert 'task="alpha"' in made.stdout and 'task="beta"' in made.stdout
        assert "native_runtime_unavailable" in made.stdout
        assert "missing_required_file" in made.stdout
        first = output.read_bytes()
        report = json.loads(first)
        assert report["summary"] == {"tasks": 2, "eligible_static": 0, "skipped": 2}
        assert [row["id"] for row in report["rows"]] == ["alpha", "beta"]
        again = subprocess.run(command + ["--out", str(output)], cwd=_REPO,
                               capture_output=True, text=True, timeout=30,
                               encoding="utf-8", errors="replace")
        assert again.returncode == 0, again.stderr
        assert output.read_bytes() == first
        mismatched = subprocess.run(
            [part if part != digest else "0" * 64 for part in command],
            cwd=_REPO, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace")
        assert mismatched.returncode == 2
        assert "task-set SHA-256 mismatch" in mismatched.stderr
        rejected = subprocess.run(command + ["--out", str(root / "inside.json")],
                                  cwd=_REPO, capture_output=True, text=True, timeout=30,
                                  encoding="utf-8", errors="replace")
        assert rejected.returncode == 2
        assert not (root / "inside.json").exists()
        no_run = subprocess.run([sys.executable, "-m",
                                 "helioslm_v5.eval.tbench_adapter", "run"],
                                cwd=_REPO, capture_output=True, text=True, timeout=30,
                                encoding="utf-8", errors="replace")
        assert no_run.returncode == 2
        # Direct main() uses the same read-only path; capture keeps daily-gate
        # output deterministic without invoking any task-provided script.
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            assert tba.main(command[3:]) == 0
        assert 'task="alpha"' in stdout.getvalue()
    print("PASS test_cli_lists_every_skip_and_is_deterministic")


def test_tbench_adapter_suite():
    test_identity_native_and_no_docker_gate()
    test_capability_rejections_and_invalid_tasks()
    test_symlink_is_data_not_a_host_read()
    test_cli_lists_every_skip_and_is_deterministic()
    _pass("test_tbench_adapter_suite",
          "Terminal-Bench Phase 2A: pinned canonical identity, Git tree, "
          "read-only native/no-docker classification, symlink confinement, "
          "complete skip IDs/reasons and deterministic CLI; no task execution")


if __name__ == "__main__":
    test_tbench_adapter_suite()
