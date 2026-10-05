import re
import tempfile
import zlib

try:
    from .gate import FixedGate, OracleGate, Route, ThresholdGate
    from .loop import AgentLoop
    from .schema import ToolCall, render_tool_call
    from .tools import build_default_registry
except ImportError:
    from gate import FixedGate, OracleGate, Route, ThresholdGate
    from loop import AgentLoop
    from schema import ToolCall, render_tool_call
    from tools import build_default_registry


def scripted_correct_policy(prompt: str, seed: int, step: int) -> str:
    task = next(l[len("Task: "):] for l in prompt.splitlines()
                if l.startswith("Task: "))
    obs = re.findall(r"^step \d+: (.*)$", prompt, flags=re.M)
    m = re.match(r"Compute the value of: (.*)", task)
    if m:
        if not obs:
            return render_tool_call([ToolCall("calc", {"expr": m.group(1)})])
        return render_tool_call([ToolCall("finish", {"answer": obs[-1]})])
    m = re.match(r'Apply (\w+) to the string: "(.*)"', task)
    if m:
        if not obs:
            return render_tool_call([ToolCall(
                "str_op", {"s": m.group(2), "op": m.group(1), "n": 0})])
        return render_tool_call([ToolCall("finish", {"answer": obs[-1]})])
    m = re.match(
        r"First compute: (.*?)\. Then apply (\w+) to the digits of the result\.",
        task)
    if m:
        if not obs:
            return render_tool_call([ToolCall("calc", {"expr": m.group(1)})])
        if len(obs) == 1:
            return render_tool_call([ToolCall(
                "str_op", {"s": obs[-1], "op": m.group(2), "n": 0})])
        return render_tool_call([ToolCall("finish", {"answer": obs[-1]})])
    return render_tool_call([ToolCall(
        "finish", {"answer": obs[-1] if obs else ""})])


def make_env_oracle(tasks):
    by_text = {t.text: t for t in tasks}

    def oracle(call, ctx):
        task = by_text[ctx["task"]]
        if call.name == "finish":
            return f"FINISH: {task.answer}"
        return "ORACLE_OK"
    return oracle


def run_modes(tasks, verify, model_fn, oracle, max_steps=8,
              confidence_fn=None, taus=(0.5,)):
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)

        def run(gate, tag):
            loop = AgentLoop(model_fn, reg, impls, gate,
                             max_steps=max_steps, confidence_fn=confidence_fn)
            n_ok, trajs = 0, []
            for task in tasks:
                traj = loop.run(task.text,
                                seed=zlib.crc32(task.text.encode()) & 0xFFFF)
                trajs.append(traj)
                if traj.final_answer is not None \
                        and verify(task, traj.final_answer):
                    n_ok += 1
            out[tag] = (n_ok / len(tasks), trajs)

        run(FixedGate(Route.DIRECT), "direct")
        run(OracleGate(oracle), "oracle")
        for tau in taus:
            run(ThresholdGate(tau, oracle), f"routed_tau{tau}")
    return out


def export_score_stream(trajectories, path: str) -> None:
    import json
    try:
        from .trajectory import ids_to_text
    except ImportError:
        from trajectory import ids_to_text
    with open(path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            for s in traj.steps:
                f.write(json.dumps({
                    "prompt": ids_to_text(s.prompt_ids),
                    "output": ids_to_text(s.generated_ids),
                    "route": s.route,
                    "parse_error": s.parse_error},
                    ensure_ascii=False) + "\n")


def _demo():
    import random
    try:
        from .envs import make_envs
    except ImportError:
        from envs import make_envs
    rng = random.Random(3)
    tasks = [e.sample(rng) for e in make_envs() for _ in range(20)]
    verify = lambda t, a: a.strip() == t.answer
    oracle = make_env_oracle(tasks)
    for label, conf in [("confidence=1.0 (routed==direct)",
                         lambda p, s: 1.0),
                        ("confidence=0.0 (routed==oracle)",
                         lambda p, s: 0.0)]:
        print(f"--- {label} ---")
        res = run_modes(tasks, verify, scripted_correct_policy, oracle,
                        confidence_fn=conf)
        for tag, (acc, trajs) in res.items():
            print(f"  {tag:16s} correctness={acc:.2f}  n={len(trajs)}")
    export_score_stream(res["direct"][1], "/tmp/agent_stream.jsonl")
    print("wrote /tmp/agent_stream.jsonl")


if __name__ == "__main__":
    _demo()
