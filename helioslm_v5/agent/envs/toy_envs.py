import random
from dataclasses import dataclass

try:
    from ..tools import calc, str_op
except ImportError:
    from tools import calc, str_op


@dataclass(frozen=True)
class Task:
    env: str
    text: str
    answer: str
    step_budget: int


class CalcEnv:
    def __init__(self):
        self.ops = ["+", "-", "*", "//", "%"]

    def sample(self, rng: random.Random) -> Task:
        n = rng.randint(2, 4)
        nums = [rng.randint(-99, 99) for _ in range(n)]
        expr = str(nums[0])
        for i in range(1, n):
            op = rng.choice(self.ops)
            num = nums[i]
            if op in ("//", "%") and num == 0:
                num = rng.randint(1, 99)
            expr += f" {op} {num}"
        return Task("calc", f"Compute the value of: {expr}",
                    calc(expr), 2)

    def verify(self, task: Task, answer: str) -> bool:
        return answer.strip() == task.answer


class StrEnv:
    def __init__(self):
        self.ops_n0 = ["upper", "lower", "reverse"]
        self.alpha = ("abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,!?")

    def sample(self, rng: random.Random) -> Task:
        s = "".join(rng.choice(self.alpha)
                    for _ in range(rng.randint(3, 12)))
        op = rng.choice(self.ops_n0)
        return Task("str", f"Apply {op} to the string: \"{s}\"",
                    str_op(s, op), 2)

    def verify(self, task: Task, answer: str) -> bool:
        return answer.strip() == task.answer.strip()


class ComposeEnv:
    def __init__(self):
        self._calc, self._str = CalcEnv(), StrEnv()

    def sample(self, rng: random.Random) -> Task:
        c = self._calc.sample(rng)
        expr = c.text.split(": ", 1)[1]
        op = rng.choice(self._str.ops_n0)
        return Task("compose",
                    f"First compute: {expr}. "
                    f"Then apply {op} to the digits of the result.",
                    str_op(c.answer, op), 4)

    def verify(self, task: Task, answer: str) -> bool:
        return answer.strip() == task.answer


def make_envs() -> list:
    return [CalcEnv(), StrEnv(), ComposeEnv()]
