"""T80 - B1-6: true parallel swarm delegation (ThreadPoolExecutor).

v1.17's delegate() was synchronous — parallelism was 0. B1-6 adds
swarm.delegate_all: workers run concurrently on threads, each in a child
Context (own state dict, own effect buffer), and child effect logs merge
into the parent in worker-NAME order with worker_id-prefixed kinds, so
the replay log is byte-identical regardless of thread scheduling.
aggregate() gains join semantics: all / any / quorum over the recorded
completion order.

Run from repo root: python3 evastar/tests/test_swarm_parallel.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from harness.harness_core import Context  # noqa: E402
from harness.harness_plugins import (AgentSwarmPlugin,  # noqa: E402
                                     AgentWorkflowPlugin, ModelPlugin,
                                     PresetPlugin)


def _mk_ctx(n=3):
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(AgentSwarmPlugin(n_workers=n))
    ctx.use(AgentWorkflowPlugin())
    return ctx


def _sleeper(answer, delay):
    """Worker that burns `delay` seconds on its first step (stands in for
    torch native inference time) and then answers."""
    def fn(prompt, seed, step):
        if step == 0:
            time.sleep(delay)
        return answer
    return fn


def test_parallel_speedup_and_isolation():
    ctx = _mk_ctx(n=2)
    fns = {"a": _sleeper("A", 0.6), "b": _sleeper("B", 0.6)}
    ctx.swarm.spawn("task", fns)
    t0 = time.perf_counter()
    out = ctx.swarm.delegate_all([("a", "sub-a", fns["a"]),
                                  ("b", "sub-b", fns["b"])])
    wall = time.perf_counter() - t0
    # serial would take >= 1.2s; parallel should land near 0.6s
    assert wall < 0.95, f"no parallelism: delegate_all took {wall:.2f}s"
    # per-worker state isolation: each worker got its OWN workflow state
    assert out == {"a": "A", "b": "B"}, out
    workers = ctx.state["swarm"]["workers"]
    assert workers["a"]["status"] == "done"
    assert workers["b"]["status"] == "done"
    print(f"PASS test_parallel_speedup_and_isolation ({wall:.2f}s)")


def test_effect_log_deterministic():
    def run_once():
        ctx = _mk_ctx(n=2)
        fns = {"a": _sleeper("A", 0.05), "b": _sleeper("B", 0.2)}
        ctx.swarm.spawn("task", fns)
        ctx.swarm.delegate_all([("a", "s1", fns["a"]),
                                ("b", "s2", fns["b"])])
        return [e.kind for e in ctx.effects], ctx
    kinds1, ctx1 = run_once()
    kinds2, _ctx2 = run_once()
    # name-ordered merge => identical log regardless of thread scheduling
    assert kinds1 == kinds2, f"nondeterministic log: {kinds1} vs {kinds2}"
    # worker effects carry the worker_id prefix
    assert any(k.startswith("a:step_") for k in kinds1), kinds1
    assert any(k.startswith("b:step_") for k in kinds1), kinds1
    assert "swarm_delegate_all" in kinds1 and "swarm_join" in kinds1
    # the join effect records the observed completion order (audit)
    join = [e for e in ctx1.effects if e.kind == "swarm_join"][0]
    assert join.payload["completion_order"][0] == "a"  # a sleeps less
    assert ctx1.verify_all_effects()["failed"] == 0
    print("PASS test_effect_log_deterministic")


def test_join_semantics():
    ctx = _mk_ctx(n=3)
    fns = {"fast": _sleeper("x", 0.05), "mid": _sleeper("y", 0.25),
           "slow": _sleeper("z", 0.5)}
    ctx.swarm.spawn("task", fns)
    ctx.swarm.delegate_all([(n, f"s-{n}", f) for n, f in fns.items()])
    order = ctx.state["swarm"]["completion_order"]
    assert order == ["fast", "mid", "slow"], order
    assert ctx.swarm.aggregate(mode="any") == "x"
    # quorum over first floor(3/2)+1 = 2 completions, differing -> list
    assert ctx.swarm.aggregate(mode="quorum") == ["x", "y"]
    # explicit quorum_k=1 equals any here
    assert ctx.swarm.aggregate(mode="quorum", quorum_k=1) == "x"
    # all: differing results -> full list (v1.17 concat behaviour)
    assert ctx.swarm.aggregate() == ["x", "y", "z"]
    print("PASS test_join_semantics")


def test_sync_delegate_compat():
    """v1.17 sync delegate path is unchanged and also feeds the
    completion order, so any/quorum work after sync delegates too."""
    ctx = _mk_ctx(n=2)
    fns = {"a": _sleeper("A", 0.0), "b": _sleeper("B", 0.0)}
    ctx.swarm.spawn("task", fns)
    r1 = ctx.swarm.delegate("a", "sub1", fns["a"])
    r2 = ctx.swarm.delegate("b", "sub2", fns["b"])
    assert (r1, r2) == ("A", "B")
    assert ctx.state["swarm"]["completion_order"] == ["a", "b"]
    assert ctx.swarm.aggregate(mode="any") == "A"
    assert ctx.swarm.aggregate() == ["A", "B"]
    print("PASS test_sync_delegate_compat")


def main():
    test_parallel_speedup_and_isolation()
    test_effect_log_deterministic()
    test_join_semantics()
    test_sync_delegate_compat()
    print("T80 PASS: 4/4")


if __name__ == "__main__":
    main()
