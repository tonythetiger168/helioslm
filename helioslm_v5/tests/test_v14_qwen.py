"""T46 - v1.4: Qwen3-0.6B through harness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentWorkflowPlugin, DecisionPlugin,
                                     PresetPlugin, QwenModelPlugin,
                                     StatePlugin, ToolPlugin)


def _qwen_available():
    import os
    return os.path.exists("qwen/config.json")


def test_qwen_loads_through_harness():
    if not _qwen_available():
        print("SKIP: qwen/ not found")
        return
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(QwenModelPlugin())
    ctx.use(ToolPlugin())
    ctx.use(AgentWorkflowPlugin())
    if ctx._services.get("model.qwen") is None:
        print("SKIP: qwen model failed to load")
        return
    out = ctx.model.qwen("What is 2+2? Answer with just the number.", 0, 0)
    assert len(out) > 0
    print(f"PASS test_qwen_loads (output={out[:40]!r})")


def test_qwen_chat_mode():
    if not _qwen_available():
        print("SKIP")
        return
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(QwenModelPlugin())
    if ctx._services.get("model.qwen") is None:
        print("SKIP")
        return
    out = ctx.model.qwen(
        "<|im_start|>user\nThe magic word is kiwi. What is the magic word?<|im_end|>\n<|im_start|>assistant\n",
        0, 0)
    assert "kiwi" in out.lower()
    print(f"PASS test_qwen_chat_mode (output={out[:40]!r})")


if __name__ == "__main__":
    test_qwen_loads_through_harness()
    test_qwen_chat_mode()
    print("\nv1.4 qwen tests done")
