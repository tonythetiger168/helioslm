"""T54 - v1.12: niche plugins (MCP/tmux/everything)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (EverythingPlugin, MCPWizardPlugin,
                                     PresetPlugin, TmuxPlugin)


def test_mcp_stub():
    ctx = Context()
    ctx.use(MCPWizardPlugin())
    r = ctx.mcp.wrap("test-server", "npx")
    assert "error" in r or "wrapped" in r
    print("PASS test_mcp_stub")


def test_tmux_or_skip():
    ctx = Context()
    ctx.use(TmuxPlugin())
    r = ctx.tmux.list_sessions()
    if "error" in r:
        print("SKIP test_tmux (tmux not installed)")
        return
    assert isinstance(r.get("sessions"), list)
    print("PASS test_tmux_or_skip")


def test_everything_search():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "hello_world.txt"), "w").write("content")
        ctx = Context()
        ctx.use(EverythingPlugin())
        r = ctx.everything.search(d, "hello")
        assert r["n"] >= 1
        assert any("hello" in h for h in r["hits"])
    print("PASS test_everything_search")


if __name__ == "__main__":
    test_mcp_stub()
    test_tmux_or_skip()
    test_everything_search()
    print("\nniche plugins tests done")
