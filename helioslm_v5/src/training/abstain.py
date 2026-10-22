"""Abstention-aware rewards for GRPO (OS3, after Hy3's "zhi zhi" / know
when to stop): pay for a correct answer, pay nothing for "I don't know",
and charge for a wrong answer, so guessing stops paying.

Reward (per response):
  - answerable task:   correct -> +1, abstain -> 0, wrong -> -penalty
  - unanswerable task: abstain -> +1, any answer -> -penalty

With these payoffs a policy maximizes reward by answering only when its
probability of being right exceeds ``penalty / (1 + penalty)``
(:func:`answer_threshold`); ``penalty=0`` recovers the plain accuracy
reward, under which guessing is never worse than abstaining.

Unanswerable tasks are CONSTRUCTED by withholding information the
question needs (a missing operand), so the ground truth "this cannot be
answered" is known without a reference model. They teach the abstain
action; they do not measure factual hallucination on open questions,
which needs a real benchmark (SimpleQA-style), recorded here rather than
claimed.

Abstention is detected from an explicit ``ANSWER: UNKNOWN`` line or a
short list of English / Traditional Chinese phrases; a response that both
abstains and states a number counts as an answer (hedging is not
abstaining).
"""
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .envs import MathEnv, _last_number

UNANSWERABLE = None   # target of a task that cannot be answered

ABSTAIN_PATTERNS = (
    r"\bANSWER:\s*UNKNOWN\b",
    r"\bI (?:do not|don't) know\b",
    r"\b(?:cannot|can't|can not) (?:be )?(?:determine|answer)",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"不知道", r"無法(?:確定|判斷|回答|得知)", r"資訊不足", r"信息不足", r"条件不足", r"條件不足",
)
_ABSTAIN_RE = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstention(response: str) -> bool:
    """True if the response declines to answer and states no number."""
    final = response.split("</think>")[-1]
    return bool(_ABSTAIN_RE.search(final)) and _last_number(final) is None


def answer_threshold(penalty: float) -> float:
    """Confidence above which answering beats abstaining in expectation."""
    if penalty < 0:
        raise ValueError("penalty must be >= 0")
    return penalty / (1.0 + penalty)


def abstain_reward(response: str, target: Optional[str],
                   correct: Callable[[str, str], bool],
                   penalty: float = 1.0) -> float:
    if penalty < 0:
        raise ValueError("penalty must be >= 0")
    if is_abstention(response):
        return 1.0 if target is UNANSWERABLE else 0.0
    if target is UNANSWERABLE:
        return -penalty
    return 1.0 if correct(response, target) else -penalty


@dataclass
class AbstainMathEnv:
    """MathEnv plus constructed unanswerable variants (one operand
    withheld). ``p_unanswerable`` of the tasks cannot be answered."""
    name: str = "math-abstain"
    p_unanswerable: float = 0.3
    penalty: float = 1.0
    rng: random.Random = None

    def __post_init__(self):
        if not 0.0 <= self.p_unanswerable <= 1.0:
            raise ValueError("p_unanswerable must be in [0, 1]")
        if self.rng is None:
            self.rng = random.Random(0)
        self._math = MathEnv()

    def task(self, seed: Optional[int] = None) -> dict:
        rng = random.Random(seed) if seed is not None else self.rng
        t = self._math.task(seed=rng.randint(0, 10 ** 9))
        suffix = (" If the question cannot be answered, reply "
                  "ANSWER: UNKNOWN.")
        if rng.random() < self.p_unanswerable:
            a, op, _b = re.match(r"What is (-?\d+) (\S) (-?\d+)\?",
                                 t["prompt"]).groups()
            hidden = rng.choice(["a number I chose", "my age",
                                 "the number on my card"])
            prompt = (f"What is {a} {op} {hidden}? Answer with the number."
                      + suffix)
            return {"prompt": prompt, "target": UNANSWERABLE}
        return {"prompt": t["prompt"] + suffix, "target": t["target"]}

    def reward(self, response: str, target: Optional[str]) -> float:
        return abstain_reward(
            response, target,
            lambda r, tg: self._math.reward(r, tg) == 1.0, self.penalty)


def abstain_metrics(tasks: List[dict], responses: List[str],
                    correct: Callable[[str, str], bool]) -> Dict[str, float]:
    """Accuracy / abstention / hallucination split by answerability.

    - ``accuracy``: correct answers among answerable tasks;
    - ``abstain_rate``: abstentions among answerable tasks (over-caution);
    - ``wrong_rate``: wrong answers among answerable tasks;
    - ``hallucination_rate``: answers given to unanswerable tasks;
    - ``precision``: correct / attempted on answerable tasks (how far an
      answer can be trusted when one is given).
    """
    if len(tasks) != len(responses):
        raise ValueError(f"tasks/responses length mismatch: "
                         f"{len(tasks)} vs {len(responses)}")
    ans = [(t, r) for t, r in zip(tasks, responses)
           if t["target"] is not UNANSWERABLE]
    una = [r for t, r in zip(tasks, responses)
           if t["target"] is UNANSWERABLE]
    n_abs = sum(is_abstention(r) for _, r in ans)
    n_ok = sum(not is_abstention(r) and correct(r, t["target"])
               for t, r in ans)
    attempted = len(ans) - n_abs
    out = {"n_answerable": len(ans), "n_unanswerable": len(una)}
    if ans:
        out.update(accuracy=n_ok / len(ans), abstain_rate=n_abs / len(ans),
                   wrong_rate=(attempted - n_ok) / len(ans))
    if attempted:
        out["precision"] = n_ok / attempted
    if una:
        out["hallucination_rate"] = sum(not is_abstention(r)
                                        for r in una) / len(una)
    return out
