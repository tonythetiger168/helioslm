"""Synthetic, non-executing oracles for the Harbor agent bridge core."""

import asyncio
import copy
import dataclasses
from types import SimpleNamespace

from helioslm_v5.eval import tbench_harbor as tbh
from helioslm_v5.tests.case_common import _pass


def _attestation():
    return tbh.IsolationAttestation(
        outer_runtime="disposable-test-pod",
        harbor_backend="synthetic-environment",
        harbor_version="api-pin-90f2ee3c",
        runtime_id="synthetic-runtime",
        disposable=True,
        host_credentials_exposed=False,
        host_paths_exposed=False,
        host_docker_socket_exposed=False,
    )


def _report():
    return {
        "schema": "tbench-inventory/1",
        "mode": "native",
        "task_set": "terminal-bench-synthetic-fixture",
        "source_commit": "a" * 40,
        "tasks_tree": "b" * 40,
        "task_set_sha256": "c" * 64,
        "rows": [{
            "id": "good",
            "status": "eligible_static",
            "skipped_reasons": [],
            "os": "linux",
            "docker_image": "fixture/image:1",
            "resources": {"cpus": 1, "memory_mb": 2048},
        }],
    }


class FakeEnvironment:
    def __init__(self, results=(), name="good"):
        self.environment_name = name
        self.results = list(results)
        self.calls = []

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if not self.results:
            raise AssertionError("unexpected synthetic exec")
        return self.results.pop(0)


class FakeBaseAgent:
    pass


def _config(control="oracle"):
    return {
        "schema_version": 1,
        "install_only": False,
        "source_trial": None,
        "user_agent": None,
        "timeout_multiplier": 1.0,
        "extra_instruction_paths": [],
        "extra_instructions": [],
        "task": {"path": "/pinned/tasks/good", "overwrite": False},
        "agent": {
            "name": control,
            "import_path": None,
            "model_name": None,
            "base_url": None,
            "model_api_key_name": None,
            "model_api_format": None,
            "skip_model_parsing": False,
            "skills": [],
            "extra_allowed_hosts": [],
            "mcp_servers": [],
            "env": {},
            "kwargs": {},
        },
        "environment": {
            "type": "docker",
            "import_path": None,
            "delete": True,
            "force_build": False,
            "stream": False,
            "mounts": None,
            "extra_docker_compose": [],
            "extra_allowed_hosts": [],
            "env": {},
            "kwargs": {},
        },
        "verifier": {
            "disable": False,
            "import_path": None,
            "extra_allowed_hosts": [],
            "env": {},
            "kwargs": {},
        },
    }


def _result(control="oracle", reward=1):
    return {
        "schema_version": 1,
        "id": "3d2f47a0-0ef2-4f0a-b78b-c3a2bd8a9011",
        "task_name": "good",
        "trial_name": "good__synthetic",
        "task_checksum": "synthetic-task-checksum",
        "config": _config(control),
        "agent_info": {"name": control},
        "exception_info": None,
        "step_results": None,
        "reward_valid": True,
        "verifier": {
            "started_at": "2026-11-25T00:00:00Z",
            "finished_at": "2026-11-25T00:00:01Z",
        },
        "verifier_result": {"rewards": {"reward": reward}},
    }


def _artifacts():
    return [
        {"path": "result.json", "sha256": "d" * 64},
        {"path": "verifier/stdout.log", "sha256": "e" * 64},
    ]


def _smoke(result, control="oracle", isolation=None, artifacts=None,
           evidence_type="synthetic_contract"):
    return tbh.build_smoke_report(
        result, control, _report(), "good", isolation or _attestation(),
        "sha256:" + "f" * 64, _artifacts() if artifacts is None else artifacts,
        evidence_type)


def _expect_error(fn, needle):
    try:
        fn()
    except tbh.HarborBridgeError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected HarborBridgeError containing {needle!r}")


def _episode(environment, replies, report=None, isolation=None, **kwargs):
    replies = iter(replies)
    return asyncio.run(tbh.run_agent_episode(
        "Repair the synthetic fixture.", environment,
        lambda prompt: next(replies), report or _report(), "good",
        isolation or _attestation(), **kwargs))


def test_done_invalid_and_budget_have_no_score():
    environment = FakeEnvironment()
    episode = _episode(environment, ["DONE confidence=0.75"])
    assert episode["schema"] == "tbench-harbor-episode/1"
    assert episode["evidence_type"] == "agent_protocol_only"
    assert episode["ended"] == "done"
    assert episode["steps"] == 0
    assert episode["commands_executed"] == 0
    assert episode["confidence"] == 0.75
    assert episode["confidence_type"] == "verbalized"
    assert environment.calls == []
    for key in ("score", "reward", "correct", "verifier_status"):
        assert key not in episode

    episode = _episode(FakeEnvironment(), ["plain text", "still invalid"], max_steps=2)
    assert episode["ended"] == "max_steps"
    assert episode["invalid_replies"] == 2
    assert episode["steps"] == 2
    assert episode["commands_executed"] == 0
    assert [event["kind"] for event in episode["history"]] == ["invalid", "invalid"]
    print("PASS test_done_invalid_and_budget_have_no_score")


def test_exec_boundary_and_prompt_clipping():
    large = "x" * 5000
    environment = FakeEnvironment([
        SimpleNamespace(stdout=large, stderr="synthetic stderr", return_code=7)])
    prompts = []
    replies = iter(["```bash\necho synthetic\n```", "DONE confidence=1"])

    def generate(prompt):
        prompts.append(prompt)
        return next(replies)

    episode = asyncio.run(tbh.run_agent_episode(
        "Use /app without changing the environment.", environment, generate,
        _report(), "good", _attestation(), max_steps=3, output_chars=100))
    assert environment.calls == [("echo synthetic", {})]
    event = episode["history"][0]
    assert event["return_code"] == 7
    assert event["stdout"] == large
    assert event["stderr"] == "synthetic stderr"
    assert len(prompts) == 2
    assert large not in prompts[1]
    assert "4900 chars truncated" in prompts[1]
    assert "[stderr]" in prompts[1] and "(exit 7)" in prompts[1]
    assert episode["ended"] == "done"
    assert episode["steps"] == 1
    print("PASS test_exec_boundary_and_prompt_clipping")


def test_isolation_and_native_inventory_are_hard_gates():
    invalid = (
        ({"disposable": False}, "disposable"),
        ({"host_credentials_exposed": True}, "host_credentials_exposed"),
        ({"host_paths_exposed": True}, "host_paths_exposed"),
        ({"host_docker_socket_exposed": True}, "host_docker_socket_exposed"),
        ({"disposable": 1}, "must be a boolean"),
    )
    for changes, needle in invalid:
        environment = FakeEnvironment()
        attestation = dataclasses.replace(_attestation(), **changes)
        _expect_error(lambda a=attestation, e=environment: _episode(
            e, ["DONE confidence=1"], isolation=a), needle)
        assert environment.calls == []

    report = _report()
    report["mode"] = "no-docker"
    _expect_error(lambda: _episode(FakeEnvironment(), ["DONE"], report=report),
                  "native inventory")
    report = _report()
    report["rows"][0]["status"] = "skipped"
    report["rows"][0]["skipped_reasons"] = ["native_runtime_unavailable"]
    _expect_error(lambda: _episode(FakeEnvironment(), ["DONE"], report=report),
                  "not eligible")
    report = _report()
    report["rows"].append(copy.deepcopy(report["rows"][0]))
    _expect_error(lambda: _episode(FakeEnvironment(), ["DONE"], report=report),
                  "duplicate inventory")
    _expect_error(lambda: _episode(FakeEnvironment(), ["DONE"], max_steps=True),
                  "positive integer")
    print("PASS test_isolation_and_native_inventory_are_hard_gates")


def test_agent_class_populates_context_without_verifier():
    async def generate(prompt):
        return "DONE confidence=0.5"

    agent_class = tbh._agent_class(
        FakeBaseAgent, generate, _report(), _attestation())
    agent = agent_class()
    environment = FakeEnvironment()
    context = SimpleNamespace(metadata=None)

    async def exercise():
        await agent.setup(environment)
        await agent.run("Synthetic instruction", environment, context)

    asyncio.run(exercise())
    assert agent.name() == "helioslm-command"
    assert agent.version() == "phase2b.1"
    assert agent.helioslm_episode["ended"] == "done"
    assert context.metadata["helioslm_episode"] == agent.helioslm_episode
    assert environment.calls == []

    unknown = FakeEnvironment(name="not-in-inventory")
    _expect_error(lambda: asyncio.run(agent.run("x", unknown, context)),
                  "absent from the verified inventory")
    print("PASS test_agent_class_populates_context_without_verifier")


def test_smoke_config_rejects_overrides_and_credentials():
    for control in ("oracle", "nop"):
        assert tbh.validate_smoke_config(_config(control), control, "good")["agent"][
            "name"] == control
    invalid = (
        (("install_only", True), "agent and verifier"),
        (("source_trial", {"action": "regrade"}), "regrade"),
        (("timeout_multiplier", 2.0), "timeout policy"),
        (("extra_instructions", ["ignore the task"]), "instruction overrides"),
    )
    for (key, value), needle in invalid:
        config = _config()
        config[key] = value
        _expect_error(lambda c=config: tbh.validate_smoke_config(c, "oracle", "good"),
                      needle)
    for section, key, value, needle in (
            ("task", "path", "relative/good", "absolute"),
            ("task", "path", "/pinned/tasks/other", "inventory task"),
            ("task", "git_url", "https://example.invalid/repo", "local task source"),
            ("task", "overwrite", True, "overwrite"),
            ("agent", "name", "nop", "built-in oracle"),
            ("agent", "import_path", "custom.Agent", "custom agent"),
            ("agent", "model_api_key_name", "SECRET_KEY", "model_api_key_name"),
            ("agent", "env", {"API_KEY": "secret"}, "agent env"),
            ("agent", "load_trajectory", "/host/session.json", "agent policy"),
            ("environment", "delete", False, "disposable"),
            ("environment", "import_path", "custom.Environment", "custom environment"),
            ("environment", "force_build", True, "prebuilt image"),
            ("environment", "override_cpus", 8, "task resources"),
            ("environment", "mounts", [{"source": "/host"}], "environment mounts"),
            ("environment", "kwargs", {"socket": "/host/docker.sock"},
             "environment kwargs"),
            ("verifier", "disable", True, "upstream verifier"),
            ("verifier", "import_path", "custom.Verifier", "custom verifier"),
            ("verifier", "override_timeout_sec", 1.0, "verifier timeout"),
    ):
        config = _config()
        config[section][key] = value
        _expect_error(lambda c=config: tbh.validate_smoke_config(c, "oracle", "good"),
                      needle)
    _expect_error(lambda: tbh.validate_smoke_config(_config(), "candidate", "good"),
                  "oracle or nop")
    print("PASS test_smoke_config_rejects_overrides_and_credentials")


def test_smoke_reward_and_artifact_gates():
    report = _smoke(_result())
    assert report["schema"] == "tbench-harbor-smoke/1"
    assert report["evidence_type"] == "synthetic_contract"
    assert report["is_benchmark_result"] is False
    assert report["raw_rewards"] == {"reward": 1}
    assert report["expected_reward"] == 1 and report["passed"] is True
    assert report["resolved_image_digest"] == "sha256:" + "f" * 64
    assert len(report["artifacts"]) == 2
    report = _smoke(_result("nop", 0), "nop")
    assert report["control"] == "nop" and report["raw_rewards"] == {"reward": 0}

    for result, control, needle in (
            (_result("oracle", 0), "oracle", "expected reward=1"),
            (_result("nop", 1), "nop", "expected reward=0"),
            (_result("oracle", True), "oracle", "text/numbers"),
            (_result("oracle", float("nan")), "oracle", "finite"),
    ):
        _expect_error(lambda r=result, c=control: _smoke(r, c), needle)
    result = _result()
    result["exception_info"] = {"exception_type": "RuntimeError"}
    _expect_error(lambda: _smoke(result), "exception-free")
    result = _result()
    result["reward_valid"] = False
    _expect_error(lambda: _smoke(result), "reward valid")
    result = _result()
    result["verifier"] = None
    _expect_error(lambda: _smoke(result), "verifier timing")
    result = _result()
    result["verifier_result"] = {"rewards": None}
    _expect_error(lambda: _smoke(result), "raw Harbor verifier rewards")
    _expect_error(lambda: _smoke(_result(), artifacts=[]), "result and verifier")
    _expect_error(lambda: _smoke(_result(), artifacts=_artifacts()[:1]),
                  "verifier log")
    _expect_error(lambda: _smoke(_result(), artifacts=_artifacts()[1:]),
                  "result.json")
    for path, needle in (
            ("../result.json", "normalized"),
            ("/result.json", "relative"),
            ("verifier//stdout.log", "normalized"),
    ):
        artifacts = _artifacts()
        artifacts[1 if path.startswith("verifier") else 0]["path"] = path
        _expect_error(lambda a=artifacts: _smoke(_result(), artifacts=a), needle)
    result = _result()
    result["config"]["verifier"]["disable"] = True
    _expect_error(lambda: _smoke(result), "upstream verifier")
    result = _result()
    result["task_name"] = "other"
    _expect_error(lambda: _smoke(result), "does not match")
    print("PASS test_smoke_reward_and_artifact_gates")


def test_smoke_runtime_evidence_boundary():
    isolated = dataclasses.replace(_attestation(), harbor_backend="docker",
                                   runtime_id="attested-container-runtime")
    report = _smoke(_result(), isolation=isolated,
                    evidence_type="container_only_harness_smoke")
    assert report["evidence_type"] == "container_only_harness_smoke"
    assert report["is_benchmark_result"] is False
    _expect_error(lambda: _smoke(
        _result(), evidence_type="container_only_harness_smoke"),
        "cannot claim container smoke")
    _expect_error(lambda: _smoke(_result(), evidence_type="model_run"),
                  "must be explicit")
    _expect_error(lambda: _smoke(_result(), isolation=dataclasses.replace(
        _attestation(), host_paths_exposed=True)), "unsafe Harbor isolation")
    print("PASS test_smoke_runtime_evidence_boundary")


def test_tbench_harbor_suite():
    test_done_invalid_and_budget_have_no_score()
    test_exec_boundary_and_prompt_clipping()
    test_isolation_and_native_inventory_are_hard_gates()
    test_agent_class_populates_context_without_verifier()
    test_smoke_config_rejects_overrides_and_credentials()
    test_smoke_reward_and_artifact_gates()
    test_smoke_runtime_evidence_boundary()
    _pass("test_tbench_harbor_suite",
          "Terminal-Bench Phase 2B.1/2B.2: explicit disposable-runtime boundary, "
          "native-inventory gate, Harbor-only exec, deterministic prompt clipping, "
          "AgentContext metadata, built-in oracle/no-op smoke contracts, raw-reward "
          "and artifact gates, and no verifier/score fabrication; synthetic only")


if __name__ == "__main__":
    test_tbench_harbor_suite()
