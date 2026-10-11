"""T58 - v1.16: robot control plugin (mock mode + TrustGate gating)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (PresetPlugin, RobotControlPlugin)


def test_robot_mock_movement():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(RobotControlPlugin(mode="mock"))
    r = ctx.robot.move_base(1.0, 0.5, 0.0, duration_s=2.0)
    assert r["ok"]
    s = ctx.robot.state()
    assert abs(s["x"] - 2.0) < 0.01 and abs(s["y"] - 1.0) < 0.01
    print("PASS test_robot_mock_movement")


def test_robot_arm_and_gripper():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(RobotControlPlugin(mode="mock"))
    r = ctx.robot.move_arm([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert r["ok"] and len(r["joints"]) == 6
    r2 = ctx.robot.gripper(0.7)
    assert abs(r2["gripper"] - 0.7) < 0.01
    r3 = ctx.robot.gripper(1.5)   # clamped
    assert r3["gripper"] == 1.0
    print("PASS test_robot_arm_and_gripper")


def test_robot_trustgate_gating():
    from decision_head import DecisionHead
    from trust_gate import TrustGate
    from gate import FixedGate, Route
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    # trust head that distrusts everything
    head = DecisionHead({"trust": ("noul", None)})
    import torch, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
    from trajectory import text_to_ids
    recs = [{"ids": torch.tensor(text_to_ids("robot move")),
             "answers": {"trust": "no", "trust__target_conf": 0.0}}
            for _ in range(20)]
    head.fit(recs, epochs=300, lr=1e-1)
    head.eval()
    gate = TrustGate(head, lo=0.5, inner=FixedGate(Route.DIRECT))
    ctx.register("decision.trust", gate)
    ctx.use(RobotControlPlugin(mode="mock"))
    r = ctx.robot.move_base(1.0, 0.0, 0.0)
    assert "blocked" in r.get("error", ""), f"gating failed: {r}"
    print("PASS test_robot_trustgate_gating")


if __name__ == "__main__":
    test_robot_mock_movement()
    test_robot_arm_and_gripper()
    test_robot_trustgate_gating()
    print("\nrobot control plugin tests done")
