import random
import string
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from schema import (ToolCall, ToolCallError, ToolRegistry, ToolSpec,
                    parse_tool_call, render_tool_call)
from tools import build_default_registry, calc, execute, str_op


def _rand_expr(rng) -> str:
    nums = [str(rng.randint(-99, 99)) for _ in range(3)]
    ops = rng.choices(["+", "-", "*", "//", "%"], k=2)
    for i, op in enumerate(ops):
        if op in ("//", "%") and int(nums[i + 1]) == 0:
            nums[i + 1] = str(rng.randint(1, 99))
    return f"({nums[0]} {ops[0]} {nums[1]}) {ops[1]} {nums[2]}"


def test_t1_render_parse_roundtrip():
    rng = random.Random(0)
    reg = ToolRegistry([
        ToolSpec("calc", "", {"expr": str}),
        ToolSpec("mix", "", {"a": int, "b": str, "c": float, "d": bool}),
    ])
    for _ in range(500):
        if rng.random() < 0.5:
            call = ToolCall("calc", {"expr": _rand_expr(rng)})
        else:
            call = ToolCall("mix", {
                "a": rng.randint(-10**6, 10**6),
                "b": "".join(rng.choices(string.printable,
                                         k=rng.randint(0, 30))),
                "c": rng.uniform(-1e6, 1e6),
                "d": rng.random() < 0.5,
            })
        calls = [call] if rng.random() < 0.7 else [call, call]
        assert parse_tool_call(render_tool_call(calls), reg) == calls


def _expect_error(text, reg, label):
    try:
        parse_tool_call(text, reg)
    except ToolCallError:
        return
    raise AssertionError(f"T2 FAIL: {label}: expected ToolCallError")


def test_t2_parse_boundaries():
    reg, _ = build_default_registry()
    cases = {
        "bad escape":       '@@tool@@{"calls":[{"name":"calc","args":{"expr":"\\q"}}]}@@end@@',
        "truncated json":   '@@tool@@{"calls":[{"name":"calc","args":{"expr":"1+',
        "truncated marker": '@@tool@@{"calls":[{"name":"finish","args":{"answer":"x"}]}',
        "type mismatch":    '@@tool@@{"calls":[{"name":"calc","args":{"expr":123}}]}@@end@@',
        "bool is not int":  '@@tool@@{"calls":[{"name":"str_op","args":{"s":"a","op":"repeat","n":true}}]}@@end@@',
        "unknown tool":     '@@tool@@{"calls":[{"name":"hack","args":{}}]}@@end@@',
        "missing arg":      '@@tool@@{"calls":[{"name":"calc","args":{}}]}@@end@@',
        "extra arg":        '@@tool@@{"calls":[{"name":"calc","args":{"expr":"1","x":1}}]}@@end@@',
        "garbage outside":  'thinking... @@tool@@{"calls":[{"name":"finish","args":{"answer":"x"}}]}@@end@@',
        "whitespace inside": '@@tool@@ {"calls":[{"name":"finish","args":{"answer":"x"}}]} @@end@@',
        "empty payload":    '@@tool@@@@end@@',
        "calls not list":   '@@tool@@{"calls":{"name":"finish"}}@@end@@',
        "empty calls":      '@@tool@@{"calls":[]}@@end@@',
        "extra key":        '@@tool@@{"calls":[{"name":"finish","args":{"answer":"x"}}],"x":1}@@end@@',
        "nested too deep":  '@@tool@@{"calls":[{"name":"finish","args":{"answer":'
                            + "[" * 40 + "1" + "]" * 40 + '}}]}@@end@@',
        "double block":     '@@tool@@{"calls":[{"name":"finish","args":{"answer":"x"}}]}@@end@@'
                            '@@tool@@{"calls":[{"name":"finish","args":{"answer":"y"}}]}@@end@@',
    }
    for label, text in cases.items():
        _expect_error(text, reg, label)


def test_t3_tool_equivalence():
    rng = random.Random(42)
    for _ in range(1000):
        expr = _rand_expr(rng)
        assert calc(expr) == repr(eval(expr)), expr          # noqa: S307
    s = "Hello, HeliosLM! 123"
    assert str_op(s, "upper") == s.upper()
    assert str_op(s, "lower") == s.lower()
    assert str_op(s, "reverse") == s[::-1]
    assert str_op(s, "repeat", 3) == s * 3
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        t1 = render_tool_call([ToolCall("file_write",
                                        {"path": "a/b.txt", "content": "hi"})])
        assert execute(parse_tool_call(t1, reg)[0], reg, impls) \
            == "WROTE 2 chars -> a/b.txt"
        t2 = render_tool_call([ToolCall("file_read", {"path": "a/b.txt"})])
        assert execute(parse_tool_call(t2, reg)[0], reg, impls) == "hi"


from envs import make_envs                                        # noqa: E402
from trajectory import (ReplayMismatch, Step, Trajectory,        # noqa: E402
                        text_to_ids, verify_replay)


def _scripted_traj(reg, impls):
    steps = [("Compute 2+2", ToolCall("calc", {"expr": "2+2"}), "DIRECT"),
             ("then fix it", None, "DIRECT"),
             ("wrap up", ToolCall("finish", {"answer": "4"}), "DIRECT")]
    traj = Trajectory(task="demo", seed=0)
    for i, (prompt, call, route) in enumerate(steps):
        gen_text = ("@@tool@@{not json@@end@@" if call is None
                    else render_tool_call([call]))
        try:
            parsed, perr = parse_tool_call(gen_text, reg)[0], None
        except ToolCallError as e:
            parsed, perr = None, str(e)
        obs = (f"PARSE_ERROR: {perr}" if parsed is None
               else execute(parsed, reg, impls))
        traj.steps.append(Step(i, text_to_ids(prompt),
                               text_to_ids(gen_text), parsed, perr,
                               route, obs))
    traj.final_answer = "4"
    return traj


def test_t5_bitwise_replay():
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        traj = _scripted_traj(reg, impls)
        assert Trajectory.loads(traj.dumps()) == traj
        verify_replay(traj, reg, impls)
        bad_obs = replace(traj.steps[0], observation="5")
        try:
            verify_replay(Trajectory(traj.task, traj.seed, [bad_obs]),
                          reg, impls)
        except ReplayMismatch:
            pass
        else:
            raise AssertionError("T5 FAIL: tampered observation not caught")
        bad_call = replace(traj.steps[0],
                           parsed=ToolCall("calc", {"expr": "3+3"}))
        try:
            verify_replay(Trajectory(traj.task, traj.seed, [bad_call]),
                          reg, impls)
        except ReplayMismatch:
            pass
        else:
            raise AssertionError("T5 FAIL: tampered call not caught")


def test_t9_envs_oracle_correct():
    rng = random.Random(7)
    for env in make_envs():
        for _ in range(200):
            t = env.sample(rng)
            assert env.verify(t, t.answer), f"{t.env}: truth rejected"
            assert not env.verify(t, t.answer + "x")
            assert not env.verify(t, "")


from gate import (FixedGate, GateViolation, OracleGate, Route,    # noqa: E402
                  ThresholdGate, eval_tau_grid, routing_gate)
from loop import AgentLoop                                        # noqa: E402


def _mk_loop(reg, impls, gate, script, conf=None, max_steps=8):
    state = {"i": 0}

    def model_fn(prompt, seed, step):
        j = min(state["i"], len(script) - 1)
        state["i"] += 1
        return script[j]

    return AgentLoop(model_fn, reg, impls, gate, max_steps=max_steps,
                     confidence_fn=(lambda p, s: conf[state["i"] - 1]
                                    if conf is not None else None))


def test_t4_termination():
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        loop = _mk_loop(reg, impls, FixedGate(Route.DIRECT),
                        ['@@tool@@{"calls":[{"name":"finish","args":{"answer":"42"}}]}@@end@@'])
        t = loop.run("demo")
        assert len(t.steps) == 1 and t.final_answer == "42"
        calc_call = '@@tool@@{"calls":[{"name":"calc","args":{"expr":"1+1"}}]}@@end@@'
        loop = _mk_loop(reg, impls, FixedGate(Route.DIRECT),
                        [calc_call], max_steps=5)
        t = loop.run("demo")
        assert len(t.steps) == 5 and t.final_answer is None


def test_t6_routing_monotonicity():
    rng = random.Random(1)
    decisions = [(rng.random(), rng.random() < 0.6) for _ in range(200)]
    make_gate = lambda tau: ThresholdGate(tau, oracle=lambda c, ctx: "ORACLE_OK")

    def run_eval(gate):
        correct = 0
        for conf, direct_ok in decisions:
            r = gate.decide(None, {"confidence": conf, "task": "x", "step": 0})
            correct += (direct_ok if r == Route.DIRECT else True)
        return correct / len(decisions)

    grid = eval_tau_grid([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], make_gate, run_eval)
    routing_gate(grid)
    bad = [(t, c * 0.5 if t == 0.4 else c) for t, c in grid]
    try:
        routing_gate(bad)
    except GateViolation:
        pass
    else:
        raise AssertionError("T6 FAIL: tampered grid passed the gate")


def test_t7_gate_value_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        oracle = lambda call, ctx: "FINISH: 4"
        good = render_tool_call([ToolCall("finish", {"answer": "4"})])
        bad = render_tool_call([ToolCall("finish", {"answer": "5"})])
        conf_seq = [1.0, 0.0] * 50

        def run(gate):
            state = {"task": 0}

            def model_fn(prompt, seed, step):
                return [good, bad][state["task"] % 2]

            def conf_fn(prompt, s):
                return conf_seq[state["task"] % len(conf_seq)]

            loop = AgentLoop(model_fn, reg, impls, gate,
                             confidence_fn=conf_fn)
            correct = 0
            for k in range(100):
                state["task"] = k          # 瘥?task 鈭斗�� good/bad
                t = loop.run(f"task-{k}")
                correct += t.final_answer == "4"
            return correct / 100

        direct, routed, upper = (run(FixedGate(Route.DIRECT)),
                                 run(ThresholdGate(0.5, oracle)),
                                 run(OracleGate(oracle)))
        assert direct <= routed <= upper, (direct, routed, upper)
        assert upper == 1.0 and routed > direct


def test_t8_parse_error_recovery():
    from trajectory import verify_replay
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        good = '@@tool@@{"calls":[{"name":"finish","args":{"answer":"ok"}}]}@@end@@'
        loop = _mk_loop(reg, impls, FixedGate(Route.DIRECT),
                        ["@@tool@@{not json@@end@@", good])
        t = loop.run("demo")
        assert t.steps[0].parse_error is not None
        assert t.steps[0].observation.startswith("PARSE_ERROR")
        assert t.final_answer == "ok"
        verify_replay(t, reg, impls)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
