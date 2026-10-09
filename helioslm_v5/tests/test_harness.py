"""T42 - helios-harness oracles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (DecisionPlugin, LoopPlugin, ModelPlugin,
                                     SessionPlugin, ToolPlugin)

def test_context_dependency_injection():
    ctx = Context()
    ctx.register("model.test", lambda p, s, st: "hello")
    assert ctx.model.test("x", 0, 0) == "hello"
    print("PASS test_context_dependency_injection")

def test_effect_replay_verification():
    ctx = Context()
    ctx.effect("add", {"a": 1, "b": 2}, replay_fn=lambda p: p["a"]+p["b"] == 3)
    ctx.effect("bad", {"a": 1}, replay_fn=lambda p: False)
    ctx.effect("noreplay", {})
    rep = ctx.verify_all_effects()
    assert rep["total"] == 3 and rep["ok"] == 2 and rep["failed"] == 1
    print("PASS test_effect_replay_verification")

def test_plugins_compose():
    ctx = Context()
    ctx.use(ToolPlugin())
    ctx.use(SessionPlugin())
    ctx.use(DecisionPlugin())
    ctx.use(LoopPlugin())
    ctx.use(ModelPlugin("fake", lambda p, s, st: "42"))
    for name in ("tools.registry", "decision.grounding", "decision.trust",
                 "loop.make", "model.fake"):
        assert name in ctx._services, name
    assert any(e.kind == "session_store" for e in ctx.effects)
    print("PASS test_plugins_compose")

def test_grounded_loop_through_harness():
    from envs import make_envs
    import random
    from grounding import GroundingGate
    from gate import FixedGate, Route
    ctx = Context()
    ctx.use(ToolPlugin())
    ctx.use(DecisionPlugin(gate=FixedGate(Route.DIRECT)))
    ctx.use(LoopPlugin())
    def confab(prompt, seed, step):
        import re
        from schema import ToolCall, render_tool_call
        obs = re.findall(r"step \d+: (.*)", prompt)
        if not obs:
            return render_tool_call([ToolCall("calc", {"expr": "1+1"})])
        return render_tool_call([ToolCall("finish", {"answer": "2"})])
    rng = random.Random(7)
    env = make_envs()[0]
    ok = 0
    for _ in range(3):
        task = env.sample(rng)
        loop = ctx.loop.make(confab, max_steps=task.step_budget)
        traj = loop.run(task.text, seed=1)
        if traj.final_answer and env.verify(task, traj.final_answer):
            ok += 1
    assert ok == 3
    print("PASS test_grounded_loop_through_harness (3/3)")

if __name__ == "__main__":
    test_context_dependency_injection()
    test_effect_replay_verification()
    test_plugins_compose()
    test_grounded_loop_through_harness()
    print("\n4/4 helios-harness tests passed")
