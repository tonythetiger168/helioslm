"""T48 - v1.6: code execution sandbox (file_env + grounding)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AgentWorkflowPlugin, CodePlugin,
                                     ModelPlugin, PresetPlugin, StatePlugin,
                                     ToolPlugin)


def _confab(prompt, seed, step):
    import re
    from schema import ToolCall, render_tool_call
    obs = re.findall(r"step \d+: (.*)", prompt)
    if not obs:
        return render_tool_call([ToolCall("calc", {"expr": "1+1"})])
    return render_tool_call([ToolCall("finish", {"answer": "2"})])


def test_code_execution_sandbox():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(CodePlugin())
    ctx.use(ModelPlugin("m", _confab))
    ctx.use(AgentWorkflowPlugin())
    ctx.use(ToolPlugin())
    r = ctx.code.run("Write a file with 2+2 then read it", _confab)
    assert "file_env_ok" in r
    print(f"PASS test_code_execution_sandbox (ok={r['file_env_ok']})")


def test_code_plugin_registers():
    ctx = Context()
    ctx.use(CodePlugin())
    assert "code.run" in ctx._services
    print("PASS test_code_plugin_registers")


if __name__ == "__main__":
    test_code_plugin_registers()
    test_code_execution_sandbox()
    print("\n2/2 code sandbox tests passed")
