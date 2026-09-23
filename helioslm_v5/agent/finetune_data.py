import json
import random

try:
    from .benchmark import scripted_correct_policy
    from .schema import parse_tool_call
    from .tools import build_default_registry, execute
except ImportError:
    from benchmark import scripted_correct_policy
    from schema import parse_tool_call
    from tools import build_default_registry, execute


def render_prompt(task_text: str, history: list) -> str:
    obs = "\n".join(f"step {i}: {o}" for i, o in enumerate(history))
    return f"Task: {task_text}" + (f"\n{obs}" if obs else "")


def gen_episode(env, rng) -> list:
    task = env.sample(rng)
    reg, impls = build_default_registry()
    history, samples = [], []
    for step in range(task.step_budget):
        prompt = render_prompt(task.text, history)
        response = scripted_correct_policy(prompt, 0, step)
        samples.append({"prompt": prompt, "response": response})
        call = parse_tool_call(response, reg)[0]
        history.append(execute(call, reg, impls))
        if call.name == "finish":
            break
    return samples


def build_dataset(path: str, n_per_env: int = 2000, seed: int = 8) -> None:
    try:
        from .envs import make_envs
    except ImportError:
        from envs import make_envs
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for env in make_envs():
            for _ in range(n_per_env):
                for s in gen_episode(env, rng):
                    assert all(ord(c) < 1024
                               for c in s["prompt"] + s["response"])
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
