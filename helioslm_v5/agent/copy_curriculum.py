"""HeliosLM v5.35c — copy curriculum (the training-side therapy).

Grounding (v5.34) is the inference-time therapy; this module is the
training-side counterpart -- the retirement path for grounding itself.
The mid autopsy showed the disease is copy-shaped: the model learned
output-shape statistics but not input->output grounding. Curriculum
design (recorded):

1. EchoEnv: 'Repeat back exactly: "{s}"' -> plain-text reply s. Pure
   copy through the v5.30 text channel; strings are random high-entropy
   (letters/digits/spaces/punct) so pattern completion cannot substitute
   for copying. family="echo", step_budget 1, verify = exact match.
2. build_copy_dataset: echo episodes mixed into the existing chat SFT
   stream (ratio configurable). The tool-side copy shape (finish =
   last obs) is already dense in gen_chat_episode data; what was
   missing was copy through the DIRECT channel and high-entropy
   payloads.
3. scripted_echo_policy: deterministic solver shared by tests and SFT
   data generation (v5.35: dispatches through the shared grammar).

Not added to make_envs() on purpose: echo is curriculum, not benchmark
(same discipline as file_env in v5.31).
"""
import json
import random
import string
from dataclasses import dataclass

try:
    from . import task_grammar as _G
    from .chat import CHAT_SYSTEM, scripted_chat_policy
    from .loop import render_tool_docs
    from .schema import parse_chat_turn
    from .tools import build_default_registry, execute
except ImportError:
    import task_grammar as _G
    from chat import CHAT_SYSTEM, scripted_chat_policy
    from loop import render_tool_docs
    from schema import parse_chat_turn
    from tools import build_default_registry, execute

_ECHO_ALPHABET = string.ascii_letters + string.digits + " .,!?"


@dataclass(frozen=True)
class EchoTask:
    text: str
    answer: str
    step_budget: int = 1
    family: str = "echo"


class EchoEnv:
    def sample(self, rng: random.Random) -> EchoTask:
        n = rng.randint(4, 12)
        s = "".join(rng.choice(_ECHO_ALPHABET) for _ in range(n)).strip()
        return EchoTask(_G.render("echo", s=s), s)

    def verify(self, task: EchoTask, final_answer: str) -> bool:
        return str(final_answer).strip() == task.answer


def make_copy_envs():
    return [EchoEnv()]


def scripted_echo_policy(prompt: str, seed: int, step: int) -> str:
    """Plain-text reply = the exact echoed string (v5.30 text channel).
    Falls back to the chat policy for non-echo turns."""
    last_user = None
    for line in prompt.splitlines():
        if line.startswith("##user## "):
            last_user = line[len("##user## "):]
    if last_user is not None:
        spec = _G.parse(last_user)
        if spec is not None and spec["kind"] == "echo":
            return spec["s"]
    return scripted_chat_policy(prompt, seed, step)


def gen_echo_episode(env, rng, system: str | None = None) -> list:
    """One echo episode -> [{"prompt", "response"}]; the prompt is the
    exact inference-time render (chat discipline, v5.30.2)."""
    reg, impls = build_default_registry()
    system = system or CHAT_SYSTEM.replace("%%TOOLS%%",
                                           render_tool_docs(reg))
    task = env.sample(rng)
    prompt = "\n".join([system, f"##user## {task.text}"])
    resp = scripted_echo_policy(prompt, 0, 0)
    kind, _ = parse_chat_turn(resp, reg)
    assert kind == "text", "echo reply must be a plain text copy"
    return [{"prompt": prompt, "response": resp}]


def build_copy_dataset(path: str, n_echo: int = 800, seed: int = 7,
                       chat_path: str | None = None) -> None:
    """Echo episodes + (optionally) an existing chat SFT jsonl appended
    verbatim -- one training file, copy signal guaranteed present."""
    rng = random.Random(seed)
    env = EchoEnv()
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_echo):
            for s in gen_echo_episode(env, rng):
                assert all(ord(c) < 1024 for c in s["prompt"] + s["response"])
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        if chat_path:
            with open(chat_path, encoding="utf-8") as cf:
                for line in cf:
                    f.write(line if line.endswith("\n") else line + "\n")
