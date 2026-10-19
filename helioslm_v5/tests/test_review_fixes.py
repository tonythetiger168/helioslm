"""T87 — repo-review fixes (post v5.53).

Pins four behaviors found in review:

1. defer_validation=True with a non-clarify gate no longer hands an
   incomplete / wrong-typed call to the impl: execute() validates, the
   loop records TOOL_ERROR instead of crashing, and a malformed `finish`
   does not end the run;
2. inspect_call: missing + extra/wrong-type is "invalid" (ESCALATE), not
   "missing" (CLARIFY);
3. calc arithmetic faults (overflow, 0**-1) are ToolCallError;
4. harness side-effect plugins fail CLOSED without a TrustGate and only
   execute on Route.DIRECT (CLARIFY and ESCALATE both block); a
   TrustGateClarify wrapper defers to its inner trust gate for them;
5. NanoVideo uses a per-call concat list (removed afterwards) and quotes
   paths containing "'" correctly.

Run from repo root: python3 helioslm_v5/tests/test_review_fixes.py
"""
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from harness.harness_core import Context  # noqa: E402
from harness.harness_plugins import (  # noqa: E402
    GitHubPlugin, NanoVideoPlugin, RobotControlPlugin, TerminalPlugin)
from helioslm_v5.agent.chat import ChatSession  # noqa: E402
from helioslm_v5.agent.gate import FixedGate, Route  # noqa: E402
from helioslm_v5.agent.loop import AgentLoop  # noqa: E402
from helioslm_v5.agent.schema import (  # noqa: E402
    ToolCall, ToolCallError, render_tool_call)
from helioslm_v5.agent.tools import build_default_registry, calc  # noqa: E402
from helioslm_v5.agent.trust_gate_clarify import (  # noqa: E402
    TrustGateClarify, inspect_call)


def _scripted(*calls):
    texts = [render_tool_call([c]) for c in calls]

    def model_fn(prompt, seed, step):
        return texts[min(step, len(texts) - 1)]

    return model_fn


def test_deferred_validation_with_plain_gate_does_not_crash():
    with tempfile.TemporaryDirectory() as root:
        reg, impls = build_default_registry(root)
        partial = ToolCall("file_write", {"path": "a.txt"})
        wrong = ToolCall("str_op", {"s": "ab", "op": "repeat", "n": "3"})
        for call in (partial, wrong):
            traj = AgentLoop(_scripted(call), reg, impls,
                             FixedGate(Route.DIRECT), max_steps=1,
                             defer_validation=True).run("t")
            obs = traj.steps[0].observation
            assert obs.startswith("TOOL_ERROR"), obs
        assert not (Path(root) / "a.txt").exists()

        # a malformed finish must not end the run (and must not KeyError)
        bad_finish = ToolCall("finish", {})
        good_finish = ToolCall("finish", {"answer": "42"})
        traj = AgentLoop(_scripted(bad_finish, good_finish), reg, impls,
                         FixedGate(Route.DIRECT), max_steps=3,
                         defer_validation=True).run("t")
        assert traj.final_answer == "42" and len(traj.steps) == 2

        chat = ChatSession(_scripted(bad_finish, good_finish), reg, impls,
                           FixedGate(Route.DIRECT), max_steps=3,
                           defer_validation=True)
        assert chat.send("hi") == "42"
    print("PASS test_deferred_validation_with_plain_gate_does_not_crash")


def test_missing_plus_invalid_escalates():
    with tempfile.TemporaryDirectory() as root:
        reg, _ = build_default_registry(root)
        gate = TrustGateClarify(FixedGate(Route.DIRECT), reg)
        both = ToolCall("str_op", {"s": "ab", "n": "2"})   # op missing, n str
        info = inspect_call(both, reg)
        assert info["status"] == "invalid" and info["missing"] == ["op"]
        assert gate.decide(both, {}) == Route.ESCALATE
        only_missing = ToolCall("str_op", {"s": "ab", "n": 2})
        assert gate.decide(only_missing, {}) == Route.CLARIFY
    print("PASS test_missing_plus_invalid_escalates")


def test_calc_arithmetic_faults_are_tool_errors():
    for expr in ("1e300**2", "0**-1"):
        try:
            calc(expr)
        except ToolCallError:
            continue
        raise AssertionError(f"calc({expr!r}) did not raise ToolCallError")
    assert calc("2**10") == "1024"
    print("PASS test_calc_arithmetic_faults_are_tool_errors")


def test_harness_gates_fail_closed_and_block_clarify():
    # no TrustGate wired -> refused, never executed
    ctx = Context()
    ctx.use(TerminalPlugin())
    ctx.use(GitHubPlugin(token="fake-token"))
    ctx.use(RobotControlPlugin(mode="mock"))
    r = ctx.terminal.run("echo should-not-run")
    assert "refused" in r.get("error", "") and "stdout" not in r, r
    assert "refused" in ctx.github.create_issue("o", "r", "t").get("error", "")
    assert "refused" in ctx.robot.move_base(1.0, 0.0, 0.0).get("error", "")
    assert ctx.robot.state()["x"] == 0.0
    # explicit opt-out still works
    assert ctx.terminal.run("echo hi", gated=False)["stdout"].strip() == "hi"

    for route in (Route.CLARIFY, Route.ESCALATE):
        ctx = Context()
        ctx.register("decision.trust", FixedGate(route))
        ctx.use(TerminalPlugin())
        ctx.use(RobotControlPlugin(mode="mock"))
        r = ctx.terminal.run("echo should-not-run")
        assert "blocked" in r.get("error", "") and "stdout" not in r, r
        assert "blocked" in ctx.robot.move_base(1.0, 0.0, 0.0)["error"]

    ctx = Context()
    ctx.register("decision.trust", FixedGate(Route.DIRECT))
    ctx.use(TerminalPlugin())
    assert ctx.terminal.run("echo ok")["stdout"].strip() == "ok"
    print("PASS test_harness_gates_fail_closed_and_block_clarify")


def test_clarify_gate_defers_to_inner_for_harness_ops():
    # harness ops are not registry tools: without unwrapping, the clarify
    # schema check calls them "unknown tool" and ESCALATEs every one
    with tempfile.TemporaryDirectory() as root:
        reg, _ = build_default_registry(root)
        for inner, ok in ((FixedGate(Route.DIRECT), True),
                          (FixedGate(Route.ESCALATE), False)):
            ctx = Context()
            ctx.register("decision.trust", TrustGateClarify(inner, reg))
            ctx.use(TerminalPlugin())
            r = ctx.terminal.run("echo through-clarify")
            if ok:
                assert r.get("stdout", "").strip() == "through-clarify", r
            else:
                assert "blocked" in r.get("error", ""), r
    print("PASS test_clarify_gate_defers_to_inner_for_harness_ops")


def test_nanovideo_concat_list_quoting_and_cleanup():
    import glob
    import os
    import shutil
    import subprocess
    ctx = Context()
    ctx.use(NanoVideoPlugin())
    r = ctx.nanovideo.from_images(["a.png\nfile '/etc/passwd'"])
    assert "newline" in r.get("error", ""), r
    if shutil.which("ffmpeg") is None:
        print("SKIP test_nanovideo_concat_list_quoting_and_cleanup (no ffmpeg)")
        return
    tmp = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp, "ffmpeg_list_*")))
    with tempfile.TemporaryDirectory() as d:
        imgs = []
        for i, name in enumerate(("plain.png", "it's quoted.png")):
            path = os.path.join(d, name)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f",
                            "lavfi", "-i", f"color=c=red:s=32x32:d=1",
                            "-frames:v", "1", path], check=True)
            imgs.append(path)
        out = os.path.join(d, "out.mp4")
        r = ctx.nanovideo.from_images(imgs, out_path=out,
                                      duration_per_image=1)
        assert r.get("ok") and os.path.getsize(out) > 0, r
    assert set(glob.glob(os.path.join(tmp, "ffmpeg_list_*"))) == before
    print("PASS test_nanovideo_concat_list_quoting_and_cleanup")


if __name__ == "__main__":
    test_deferred_validation_with_plain_gate_does_not_crash()
    test_missing_plus_invalid_escalates()
    test_calc_arithmetic_faults_are_tool_errors()
    test_harness_gates_fail_closed_and_block_clarify()
    test_clarify_gate_defers_to_inner_for_harness_ops()
    test_nanovideo_concat_list_quoting_and_cleanup()
