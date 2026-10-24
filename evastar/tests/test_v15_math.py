"""T47 - v1.5: math augmentation (symbolic verify + CoT scaffold)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentWorkflowPlugin, MathPlugin,
                                     MathWorkflowPlugin, ModelPlugin,
                                     PresetPlugin, StatePlugin, ToolPlugin)


def test_math_verify_numeric():
    ctx = Context()
    ctx.use(MathPlugin())
    assert ctx.math.verify("42", "42.0")
    assert ctx.math.verify("42", "42.0000001", tol=1e-6)
    assert not ctx.math.verify("42", "41")
    assert not ctx.math.verify("abc", "42")
    print("PASS test_math_verify_numeric")


def test_math_workflow_with_scaffold():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MathPlugin())
    ctx.use(MathWorkflowPlugin(scaffold=True))
    ctx.use(ModelPlugin("m", lambda p, s, st: "42"))
    ctx.use(AgentWorkflowPlugin())
    ctx.use(ToolPlugin())
    r = ctx.math.run("Compute the value of: 6 * 7", ctx.model.m)
    assert r["workflow"]["final"] == "42"
    print("PASS test_math_workflow_with_scaffold")


def test_math_workflow_no_scaffold():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MathPlugin())
    ctx.use(MathWorkflowPlugin(scaffold=False))
    ctx.use(ModelPlugin("m", lambda p, s, st: "42"))
    ctx.use(AgentWorkflowPlugin())
    ctx.use(ToolPlugin())
    r = ctx.math.run("Compute the value of: 6 * 7", ctx.model.m)
    assert r["workflow"]["final"] == "42"
    assert not ctx.state["math"]["task"].startswith("Let me solve")
    print("PASS test_math_workflow_no_scaffold")


if __name__ == "__main__":
    test_math_verify_numeric()
    test_math_workflow_with_scaffold()
    test_math_workflow_no_scaffold()
    print("\n3/3 math augmentation tests passed")
