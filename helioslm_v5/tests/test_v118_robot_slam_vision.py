"""T60 - v1.18: multi-robot / SLAM / vision."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (MultiRobotPlugin, PresetPlugin,
                                     SLAMPlugin, VisionPlugin)


def test_multi_robot_allocation():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(MultiRobotPlugin())
    ctx.fleet.register("r1", caps=["arm"])
    ctx.fleet.register("r2", caps=["base"])
    r = ctx.fleet.allocate("pick task", required_caps=["arm"])
    assert r["robot"] == "r1"
    r2 = ctx.fleet.allocate("another pick")
    assert "error" in r2 or r2.get("robot") != "r1"
    ctx.fleet.release("r1")
    print("PASS test_multi_robot_allocation")


def test_slam_scan_and_integrate():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(SLAMPlugin())
    m = ctx.slam.create_map(20, 20)
    ranges = ctx.slam.scan(m, (5.0, 5.0, 0.0))
    assert len(ranges) == 8
    m = ctx.slam.integrate(m, (5.0, 5.0, 0.0), ranges)
    assert m["pose"] == (5.0, 5.0, 0.0)
    fr = ctx.slam.frontier(m)
    assert isinstance(fr, list)
    print("PASS test_slam_scan_and_integrate")


def test_vision_or_error():
    ctx = Context()
    ctx.use(VisionPlugin())
    r = ctx.vision.detect("/nonexistent.jpg")
    assert "error" in r or "boxes" in r
    r2 = ctx.vision.ocr("/nonexistent.png")
    assert "error" in r2
    print("PASS test_vision_or_error")


if __name__ == "__main__":
    test_multi_robot_allocation()
    test_slam_scan_and_integrate()
    test_vision_or_error()
    print("\nv1.18 tests done")
