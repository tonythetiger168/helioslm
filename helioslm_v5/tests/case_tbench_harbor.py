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


def test_tbench_harbor_suite():
    test_done_invalid_and_budget_have_no_score()
    test_exec_boundary_and_prompt_clipping()
    test_isolation_and_native_inventory_are_hard_gates()
    test_agent_class_populates_context_without_verifier()
    _pass("test_tbench_harbor_suite",
          "Terminal-Bench Phase 2B.1: explicit disposable-runtime boundary, "
          "native-inventory gate, Harbor-only exec, deterministic prompt clipping, "
          "AgentContext metadata, and no verifier/score fabrication; synthetic only")


if __name__ == "__main__":
    test_tbench_harbor_suite()
