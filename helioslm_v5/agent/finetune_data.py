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


# --- v5.30 chat SFT data ----------------------------------------------------

MAGIC_WORDS = ["kiwi", "falcon", "ember", "quartz", "meadow", "onyx",
               "harbor", "saffron", "birch", "cobalt"]
ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL = "##user##", "##assistant##", "##tool##"


def gen_chat_episode(env, rng, system: str | None = None,
                     followup_p: float = 0.5) -> list:
    """One multi-turn chat episode -> list of {"prompt", "response"} samples.

    Each prompt is the full transcript (system + ##role## turns) exactly as
    ChatSession.build_prompt renders it at inference; each response is the
    assistant output — a tool block or a plain text reply. The episode mixes
    three intents so the model learns to CHOOSE the output mode:
    1. env tool task (calc/str/compose) -> tool blocks
    2. follow-up question referencing the previous answer -> tool blocks
       driven by transcript memory
    3. direct question -> plain text reply (no tool needed)
    """
    try:
        from .chat import (CHAT_SYSTEM, FOLLOWUP_TEXT, scripted_chat_policy)
        from .loop import render_tool_docs
        from .schema import parse_chat_turn
    except ImportError:
        from chat import (CHAT_SYSTEM, FOLLOWUP_TEXT, scripted_chat_policy)
        from loop import render_tool_docs
        from schema import parse_chat_turn
    reg, impls = build_default_registry()
    system = system or CHAT_SYSTEM.replace("%%TOOLS%%",
                                           render_tool_docs(reg))
    turns: list = []

    def prompt_text() -> str:
        return "\n".join([system] + [f"{r} {c}" for r, c in turns])

    def drive(user_text: str, budget: int) -> None:
        turns.append((ROLE_USER, user_text))
        for step in range(budget):
            prompt = prompt_text()
            resp = scripted_chat_policy(prompt, 0, step)
            kind, payload = parse_chat_turn(resp, reg)
            assert kind == "tool", f"expected tool block, got {kind}: {resp}"
            call = payload[0]
            samples.append({"prompt": prompt, "response": resp})
            obs = execute(call, reg, impls)
            turns.append((ROLE_ASSISTANT, resp))
            turns.append((ROLE_TOOL, obs))
            if call.name == "finish":
                break

    samples: list = []
    task = env.sample(rng)
    drive(task.text, task.step_budget)
    # follow-up "multiply by 2" is only well-posed when the answer is a
    # legal calc operand: use the executor's own validator as the gate
    # (float("017") passes but ast.parse rejects leading zeros — str/compose
    # envs can produce such strings via reverse)
    try:
        from .tools import calc as _calc
    except ImportError:
        from tools import calc as _calc
    try:
        _calc(f"{task.answer} * 2")
        numeric = True
    except Exception:
        numeric = False
    if numeric and rng.random() < followup_p:
        drive(FOLLOWUP_TEXT, 2)
    w = rng.choice(MAGIC_WORDS)
    turns.append((ROLE_USER, f"The magic word is {w}. What is the magic word?"))
    prompt = prompt_text()
    resp = scripted_chat_policy(prompt, 0, 0)
    kind, _ = parse_chat_turn(resp, reg)
    assert kind == "text", "direct question must get a text reply"
    samples.append({"prompt": prompt, "response": resp})
    return samples


def build_chat_dataset(path: str, n_per_env: int = 2000, seed: int = 8,
                       followup_p: float = 0.5) -> None:
    """Chat SFT dataset; followup_p overrides the in-episode follow-up rate."""
    try:
        from .envs import make_envs
    except ImportError:
        from envs import make_envs
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for env in make_envs():
            for _ in range(n_per_env):
                episode = gen_chat_episode(
                    env, rng,
                    system=None,
                    followup_p=followup_p)
                for s in episode:
                    assert all(ord(c) < 1024
                               for c in s["prompt"] + s["response"])
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
