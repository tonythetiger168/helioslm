"""T56 - v1.14: blender/omi plugins (completes DeepSeek coverage)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (BlenderPlugin, OmiPlugin, PresetPlugin)


def test_blender_gen_script():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        ctx = Context()
        ctx.use(BlenderPlugin())
        r = ctx.blender.gen("a red cube", os.path.join(d, "scene"))
        assert os.path.exists(r["script"])
        content = open(r["script"]).read()
        assert "primitive_cube_add" in content
    print("PASS test_blender_gen_script")


def test_blender_run_or_missing():
    ctx = Context()
    ctx.use(BlenderPlugin())
    r = ctx.blender.run("/nonexistent.py")
    assert "error" in r
    print("PASS test_blender_run_or_missing")


def test_omi_stub():
    ctx = Context()
    ctx.use(OmiPlugin())
    r = ctx.omi.connect("Omi Test")
    assert "error" in r or "stub" in r
    print("PASS test_omi_stub")


if __name__ == "__main__":
    test_blender_gen_script()
    test_blender_run_or_missing()
    test_omi_stub()
    print("\nblender/omi plugin tests done -- DeepSeek coverage complete")
