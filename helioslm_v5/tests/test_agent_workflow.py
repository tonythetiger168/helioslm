"""T44 - v1.2: agent workflow plugin (ReAct with per-step audit)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentWorkflowPlugin, DecisionPlugin,
                                     LoopPlugin, ModelPlugin, PresetPlugin,
                                     SessionPlugin, StatePlugin, ToolPlugin)


def _confab(prompt, seed, step):
    import re
    from schema import ToolCall, render_tool_call
    obs = re.findall(r"step \d+: (.*)", prompt)
    if not obs:
        return render_tool_call([ToolCall("calc", {"expr": "1+1"})])
    return render_tool_call([ToolCall("finish", {"answer": "2"})])


def test_agent_workflow_runs_and_audits():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ModelPlugin("worker", _confab))
    ctx.use(AgentWorkflowPlugin())
    wf = ctx.agent.run("Compute the value of: 6 * 7", _confab,
                       max_steps=4)
    assert wf["final"] == "2"
    assert len(wf["steps"]) >= 2
    # per-step effects recorded
    step_effects = [e for e in ctx.effects if e.kind.startswith("step_")]
    assert len(step_effects) == len(wf["steps"])
    rep = ctx.verify_all_effects()
    assert rep["failed"] == 0
    print("PASS test_agent_workflow_runs_and_audits "
          f"({len(wf['steps'])} steps, all effects verified)")


def test_agent_workflow_long_horizon():
    from envs import make_long_envs
    import random
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ModelPlugin("w", _confab))
    ctx.use(AgentWorkflowPlugin())
    # confabulating policy: grounding fixes args, agent loop handles it
    rng = random.Random(3)
    env = make_long_envs()[0]
    task = env.sample(rng)
    wf = ctx.agent.run(task.text, _confab, max_steps=task.step_budget)
    # grounding not wired here, so final may be wrong; but workflow ran
    assert len(wf["steps"]) > 0
    assert ctx.state["workflow"]["task"] == task.text
    print("PASS test_agent_workflow_long_horizon "
          f"({len(wf['steps'])} steps on {task.family})")


if __name__ == "__main__":
    test_agent_workflow_runs_and_audits()
    test_agent_workflow_long_horizon()
    print("\n2/2 agent workflow tests passed")
