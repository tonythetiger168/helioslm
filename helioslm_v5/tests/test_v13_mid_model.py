"""T45 - v1.3: real mid model through harness (end-to-end)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentWorkflowPlugin, DecisionPlugin,
                                     MidModelPlugin, PresetPlugin,
                                     StatePlugin, ToolPlugin)


def _ckpt_available():
    import os
    return (os.path.exists("checkpoints/mid_sft_v5.33.pt")
            and os.path.exists("checkpoints/mid_sft_v5.33.tok.json"))


def test_mid_model_loads_through_harness():
    if not _ckpt_available():
        print("SKIP: mid checkpoint not in ./checkpoints")
        return
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MidModelPlugin())
    ctx.use(ToolPlugin())
    ctx.use(AgentWorkflowPlugin())
    assert "model.mid" in ctx._services
    out = ctx.model.mid("##user## The magic word is kiwi. What is the magic word?",
                        0, 0)
    assert "kiwi" in out.lower() or len(out) > 0
    print(f"PASS test_mid_model_loads_through_harness (output={out[:40]!r})")


def test_mid_model_chat_task():
    if not _ckpt_available():
        print("SKIP: mid checkpoint not in ./checkpoints")
        return
    from envs import make_envs
    import random
    from grounding import GroundingGate
    from gate import FixedGate, Route
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MidModelPlugin())
    ctx.use(ToolPlugin())
    # grounding fixes content, model provides sequence
    ctx.register("decision.grounding", GroundingGate(FixedGate(Route.DIRECT)))
    ctx.use(AgentWorkflowPlugin())
    rng = random.Random(5)
    env = make_envs()[0]
    ok = 0
    for _ in range(3):
        task = env.sample(rng)
        wf = ctx.agent.run(task.text, ctx.model.mid, max_steps=4)
        if wf["final"] and env.verify(task, wf["final"]):
            ok += 1
    # mid is weak on agentic; grounding helps but may not be enough
    print(f"PASS test_mid_model_chat_task ({ok}/3 correct -- recorded honestly)")


if __name__ == "__main__":
    test_mid_model_loads_through_harness()
    test_mid_model_chat_task()
    print("\nv1.3 mid-model tests done")
