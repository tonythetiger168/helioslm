"""T50 - v1.8: DeepSeek-harness style agent plugins (memory/terminal/fetch/fs/time)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (FetchPlugin, FilesystemPlugin,
                                     MemoryPlugin, PresetPlugin,
                                     TerminalPlugin, TimePlugin)


def test_memory_recall():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MemoryPlugin())
    ctx.memory.remember("task1", "thought A", "1.0")
    assert ctx.memory.recall("task1") == "thought A"
    print("PASS test_memory_recall")


def test_terminal_gated():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(TerminalPlugin())
    r = ctx.terminal.run("echo hello", gated=False)
    assert r["stdout"].strip() == "hello"
    print("PASS test_terminal_gated")


def test_fetch():
    ctx = Context()
    ctx.use(FetchPlugin())
    # 不網路測試——用 data: URL 或 mock
    r = ctx.fetch.get("data:text/plain,hello")
    assert r.get("body") == "hello" or "error" in r
    print("PASS test_fetch")


def test_filesystem_roundtrip():
    import tempfile, os
    with tempfile.TemporaryDirectory() as root:
        ctx = Context()
        ctx.use(FilesystemPlugin(root=root))
        ctx.fs.write("test.txt", "hello ic")
        r = ctx.fs.read("test.txt")
        assert r["content"] == "hello ic"
        # path escape blocked
        r2 = ctx.fs.read("../escape.txt")
        assert "blocked" in r2.get("error", "")
    print("PASS test_filesystem_roundtrip")


def test_time():
    ctx = Context()
    ctx.use(TimePlugin())
    assert "T" in ctx.time.now()
    assert ctx.time.utc().endswith("Z")
    print("PASS test_time")


if __name__ == "__main__":
    test_memory_recall()
    test_terminal_gated()
    test_fetch()
    test_filesystem_roundtrip()
    test_time()
    print("\n5/5 agent plugin tests passed")
