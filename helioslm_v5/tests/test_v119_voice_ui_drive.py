"""T61 - v1.19: voice dialog / terminal UI / auto driving."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (AutoDrivePlugin, PresetPlugin,
                                     TerminalUIPlugin, VoiceDialogPlugin)


def test_voice_dialog_turn():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(VoiceDialogPlugin())
    r = ctx.voice.turn("hello", lambda p, s, st: "hi there")
    assert r["response"] == "hi there"
    print("PASS test_voice_dialog_turn")


def test_terminal_ui():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(TerminalUIPlugin())
    ctx.ui.progress(5, 10, "halfway")
    m = ctx.ui.menu(["opt1", "opt2"], "Choose:")
    assert len(m) == 2
    print("PASS test_terminal_ui")


def test_autodrive_sim():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(AutoDrivePlugin())
    r = ctx.autodrive.simulate("straight", n_steps=100)
    assert r["obstacles"] >= 4
    assert isinstance(r["path"], list)
    print("PASS test_autodrive_sim")


if __name__ == "__main__":
    test_voice_dialog_turn()
    test_terminal_ui()
    test_autodrive_sim()
    print("\nv1.19 tests done")
