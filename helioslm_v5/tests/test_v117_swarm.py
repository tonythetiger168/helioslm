"""T59 - v1.17: multi-agent swarm (leader-worker + blackboard)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentSwarmPlugin, AgentWorkflowPlugin,
                                     ModelPlugin, PresetPlugin)


def _worker_a(prompt, seed, step):
    if "compute" in prompt.lower() or "calculate" in prompt.lower():
        return "42"
    return "A result"


def _worker_b(prompt, seed, step):
    if "compute" in prompt.lower() or "calculate" in prompt.lower():
        return "42"
    return "B result"


def test_swarm_spawn_and_delegate():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(AgentSwarmPlugin(n_workers=2))
    ctx.use(AgentWorkflowPlugin())
    ctx.use(ModelPlugin("a", _worker_a))
    ctx.use(ModelPlugin("b", _worker_b))
    s = ctx.swarm.spawn("Compute 6*7", {"a": _worker_a, "b": _worker_b})
    assert s["n"] == 2
    r1 = ctx.swarm.delegate("a", "Compute 6*7", ctx.model.a)
    r2 = ctx.swarm.delegate("b", "Compute 6*7", ctx.model.b)
    agg = ctx.swarm.aggregate()
    assert agg == "42" or "42" in str(agg)
    print("PASS test_swarm_spawn_and_delegate")


def test_swarm_blackboard():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(AgentSwarmPlugin(n_workers=2))
    ctx.use(AgentWorkflowPlugin())
    ctx.use(ModelPlugin("a", _worker_a))
    ctx.use(ModelPlugin("b", _worker_b))
    ctx.swarm.spawn("task", {"a": _worker_a, "b": _worker_b})
    ctx.swarm.delegate("a", "sub1", ctx.model.a)
    ctx.swarm.delegate("b", "sub2", ctx.model.b)
    bb = ctx.swarm.blackboard()
    assert "a" in bb and "b" in bb
    print("PASS test_swarm_blackboard")


if __name__ == "__main__":
    test_swarm_spawn_and_delegate()
    test_swarm_blackboard()
    print("\nswarm tests done")
