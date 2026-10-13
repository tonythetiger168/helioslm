import random
from dataclasses import dataclass

try:
    from .. import task_grammar as _G
    from ..tools import calc, str_op
except ImportError:
    import task_grammar as _G
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


class MissingParamEnv:
    """M3 toy family: under-specified tasks whose oracle is CLARIFY.

    Each scenario names the required tool argument omitted from the user's
    request. The expected answer is a machine-checkable marker,
    ``CLARIFY:<param>`` — never a guessed tool result. The default
    ``make_envs()`` deliberately does NOT include this family so existing
    completion benchmarks remain completion-only.
    """

    SCENARIOS = (
        {"family": "repeat_omits_n", "tool": "str_op",
         "text": "Apply repeat to the string: \"ab\"",
         "args": {"s": "ab", "op": "repeat"}, "missing": ("n",)},
        {"family": "repeat_omits_n", "tool": "str_op",
         "text": "Repeat the token \"xy\" as requested.",
         "args": {"s": "xy", "op": "repeat"}, "missing": ("n",)},
        {"family": "write_omits_content", "tool": "file_write",
         "text": "Write the result to \"note.txt\".",
         "args": {"path": "note.txt"}, "missing": ("content",)},
        {"family": "write_omits_content", "tool": "file_write",
         "text": "Save the answer under \"out/result.txt\".",
         "args": {"path": "out/result.txt"}, "missing": ("content",)},
        {"family": "read_omits_path", "tool": "file_read",
         "text": "Read the file I mentioned and summarize it.",
         "args": {}, "missing": ("path",)},
        {"family": "calc_omits_expr", "tool": "calc",
         "text": "Compute the expression from earlier.",
         "args": {}, "missing": ("expr",)},
    )

    def scenario(self, index: int) -> dict:
        return dict(self.SCENARIOS[index % len(self.SCENARIOS)])

    def sample(self, rng: random.Random) -> Task:
        spec = self.scenario(rng.randrange(len(self.SCENARIOS)))
        return Task("missing_param", spec["text"],
                    "CLARIFY:" + ",".join(spec["missing"]), 1)

    def verify(self, task: Task, answer: str) -> bool:
        return answer.strip() == task.answer


def make_envs() -> list:
    return [CalcEnv(), StrEnv(), ComposeEnv()]
