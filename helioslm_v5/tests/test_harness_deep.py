"""T43 - harness deep workflow: state + preset + effect viewer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (DecisionPlugin, LoopPlugin, ModelPlugin,
                                     PresetPlugin, SessionPlugin, StatePlugin,
                                     ToolPlugin)


def test_state_persists_across_plugins():
    ctx = Context()
    ctx.use(StatePlugin())
    ctx.state["step"] = 1
    ctx.state["trace"] = ["a"]
    ctx.use(ModelPlugin("m", lambda p, s, st: "x"))
    assert ctx.state["step"] == 1
    ctx.state["trace"].append("b")
    assert ctx.state["trace"] == ["a", "b"]
    print("PASS test_state_persists_across_plugins")


def test_full_preset_wires_everything():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=lambda p, s, st: "42"))
    for name in ("tools.registry", "tools.impls", "session.new",
                 "decision.grounding", "decision.trust", "loop.make",
                 "model.full", "state", "preset"):
        assert name in ctx._services, name
    assert ctx.preset == "full"
    print("PASS test_full_preset_wires_everything")


def test_effect_log_as_audit_viewer():
    ctx = Context()
    for i in range(3):
        ctx.effect(f"step_{i}", {"i": i}, replay_fn=lambda p: p["i"] >= 0)
    ctx.effect("bad", {}, replay_fn=lambda p: False)
    rep = ctx.verify_all_effects()
    assert rep["total"] == 4 and rep["ok"] == 3
    kinds = [e.kind for e in ctx.effects]
    assert kinds == ["step_0", "step_1", "step_2", "bad"]
    print("PASS test_effect_log_as_audit_viewer")


def test_multi_step_pipeline_with_state():
    from envs import make_envs
    import random
    from grounding import GroundingGate
    from gate import FixedGate, Route
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ModelPlugin("worker", lambda p, s, st: "ignored"))
    ctx.state["results"] = []
    def confab(prompt, seed, step):
        import re
        from schema import ToolCall, render_tool_call
        obs = re.findall(r"step \d+: (.*)", prompt)
        if not obs:
            return render_tool_call([ToolCall("calc", {"expr": "1+1"})])
        return render_tool_call([ToolCall("finish", {"answer": "2"})])
    rng = random.Random(11)
    env = make_envs()[0]
    for i in range(3):
        task = env.sample(rng)
        loop = ctx.loop.make(confab, max_steps=task.step_budget)
        traj = loop.run(task.text, seed=1)
        ctx.state["results"].append(env.verify(task, traj.final_answer))
        ctx.effect(f"task_{i}", {"ok": ctx.state["results"][-1]},
                   replay_fn=lambda p: p["ok"])
    assert all(ctx.state["results"])
    rep = ctx.verify_all_effects()
    assert rep["failed"] == 0
    print("PASS test_multi_step_pipeline_with_state (3 tasks, all effects verified)")


if __name__ == "__main__":
    test_state_persists_across_plugins()
    test_full_preset_wires_everything()
    test_effect_log_as_audit_viewer()
    test_multi_step_pipeline_with_state()
    print("\n4/4 harness deep-workflow tests passed")
