"""HeliosLM v5.31 — long-horizon file-system env (GLM-5 direction).

Toy envs run 2-4 steps; long-horizon agentic work needs 10-30. This env
composes the EXISTING tools (calc / str_op / file_read / file_write /
finish) into multi-phase tasks whose intermediate state lives in the
sandbox filesystem — the cheapest credible long-horizon substrate we have.

Three families (budgets 8/10/14 vs 2-4 for toy envs):
1. write_read:    compute -> write file -> read back -> finish
2. write_transform: write raw -> transform -> write result -> read -> finish
3. accumulate:    two computations -> two files -> read both -> combine -> finish

verify() recomputes the expected final answer from the task spec — no
state peeking, same discipline as toy_envs. The env is deliberately NOT
added to make_envs(): build_dataset's scripted_correct_policy does not
cover these tasks (recorded; a chat-native policy + SFT data is future
work tied to milestone 13).
"""
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FileTask:
    text: str
    answer: str
    step_budget: int
    family: str


class FileLongHorizonEnv:
    def sample(self, rng: random.Random) -> FileTask:
        family = rng.choice(["write_read", "write_transform", "accumulate"])
        if family == "write_read":
            a, b = rng.randint(-30, 30), rng.randint(1, 20)
            expr = f"{a} * {b} + {rng.randint(-50, 50)}"
            path = "f.txt"
            text = (f"Compute the value of: {expr}. Write the result to "
                    f"{path}, then read {path} and finish with its exact "
                    f"content.")
            return FileTask(text, repr(eval(expr)), 8, family)
        if family == "write_transform":
            words = ["apple", "bridge", "cloud", "delta", "ember", "frost",
                     "grape", "harbor", "igloo", "joker"]
            s = " ".join(rng.sample(words, 2))
            op = rng.choice(["upper", "lower", "reverse", "repeat"])
            n = rng.randint(2, 3) if op == "repeat" else 0
            op_text = f"{op} {n} times" if op == "repeat" else op
            a_path, b_path = "a.txt", "b.txt"
            text = (f"Write the string \"{s}\" to {a_path}. Then apply "
                    f"{op_text} to it and write the result to {b_path}. "
                    f"Read {b_path} and finish with its exact content.")
            return FileTask(text, _str_op(s, op, n), 10, family)
        e1 = f"{rng.randint(-20, 20)} * {rng.randint(2, 9)}"
        e2 = f"{rng.randint(-20, 20)} + {rng.randint(1, 40)}"
        text = (f"Compute the value of: {e1} and write the result to a.txt. "
                f"Compute the value of: {e2} and write the result to b.txt. "
                f"Read both files, compute the sum of the two values, and "
                f"finish with the sum.")
        return FileTask(text, repr(eval(e1) + eval(e2)), 14, "accumulate")

    def verify(self, task: FileTask, final_answer: str) -> bool:
        return str(final_answer).strip() == task.answer.strip()


def _str_op(s: str, op: str, n: int) -> str:
    if op == "upper":
        return s.upper()
    if op == "lower":
        return s.lower()
    if op == "reverse":
        return s[::-1]
    if op == "repeat":
        return s * n
    raise ValueError(f"unknown op {op}")


def make_long_envs():
    """Long-horizon envs, kept separate from make_envs on purpose (see
    module docstring: SFT coverage is milestone-13 work)."""
    return [FileLongHorizonEnv()]


# --- scripted correct policy (tests + future SFT data) ----------------------

def scripted_file_policy(prompt: str, seed: int, step: int) -> str:
    """Drives all three families from loop.py-rendered prompts.

    Parses the task from the "Task: ..." head, then phases by counting
    observations ("step i: <obs>").
    """
    from schema import ToolCall, render_tool_call

    obs = re.findall(r"step \d+: (.*)", prompt)
    task_m = re.search(r"Task: (.*?)(?:\n|$)", prompt)
    task = task_m.group(1) if task_m else prompt

    m = re.search(r"Compute the value of: (.*?)\. Write the result to "
                  r"(\S+?), then read", task)
    if m:  # write_read
        expr, path = m.group(1), m.group(2)
        return _phase(obs, [
            lambda: render_tool_call([ToolCall("calc", {"expr": expr})]),
            lambda: render_tool_call([ToolCall(
                "file_write", {"path": path, "content": obs[0]})]),
            lambda: render_tool_call([ToolCall("file_read", {"path": path})]),
            lambda: render_tool_call([ToolCall(
                "finish", {"answer": obs[2]})]),
        ])
    m = re.search(r"Write the string \"(.*?)\" to (\S+)\. Then apply (\w+) "
                  r"to it and write the result to (\S+)\.", task)
    if m:  # write_transform
        s, a_path, op, b_path = m.groups()
        n = 0
        m_n = re.search(r"apply repeat (\d+) times", task)
        if m_n:
            n = int(m_n.group(1))
        return _phase(obs, [
            lambda: render_tool_call([ToolCall(
                "file_write", {"path": a_path, "content": s})]),
            lambda: render_tool_call([ToolCall(
                "str_op", {"s": s, "op": op, "n": n})]),
            lambda: render_tool_call([ToolCall(
                "file_write", {"path": b_path, "content": obs[1]})]),
            lambda: render_tool_call([ToolCall("file_read", {"path": b_path})]),
            lambda: render_tool_call([ToolCall(
                "finish", {"answer": obs[3]})]),
        ])
    m = re.search(r"Compute the value of: (.*?) and write the result to "
                  r"(\S+)\. Compute the value of: (.*?) and write", task)
    if m:  # accumulate
        e1, p1, e2 = m.group(1), m.group(2), m.group(3)
        p2 = "b.txt" if p1 == "a.txt" else "a.txt"
        return _phase(obs, [
            lambda: render_tool_call([ToolCall("calc", {"expr": e1})]),
            lambda: render_tool_call([ToolCall("calc", {"expr": e2})]),
            lambda: render_tool_call([ToolCall(
                "file_write", {"path": p1, "content": obs[0]})]),
            lambda: render_tool_call([ToolCall(
                "file_write", {"path": p2, "content": obs[1]})]),
            lambda: render_tool_call([ToolCall("file_read", {"path": p1})]),
            lambda: render_tool_call([ToolCall("file_read", {"path": p2})]),
            lambda: render_tool_call([ToolCall(
                "calc", {"expr": f"({obs[4]}) + ({obs[5]})"})]),
            lambda: render_tool_call([ToolCall(
                "finish", {"answer": obs[6]})]),
        ])
    return render_tool_call([ToolCall(
        "finish", {"answer": obs[-1] if obs else ""})])


def _phase(obs, phases):
    """Lazy phase dispatch: phases are thunks so obs[i] is only evaluated
    for the selected phase (an eager list evaluates every obs[i] at build
    time and IndexErrors at step 0 — found by T24)."""
    i = len(obs)
    return (phases[i] if i < len(phases) else phases[-1])()
