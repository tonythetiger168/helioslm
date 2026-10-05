"""HeliosLM v5.32 — typed decision primitives (System One / Jev direction).

Jev (TypeSafe AI, 2026-09-15) demonstrated a non-autoregressive "decision
layer" as a product category: typed questions over a state, answered in
one pass with CALIBRATED probabilities, policy kept in code. This module
adds the same primitive types to our agent stack — additively: the
existing Gate.decide/escalate contract (T11-T19) is untouched.

Primitive semantics (aligned to Jev's public docs):
- Choice: pick exactly one of a fixed option set, with P(answer).
- Score: numeric judgment in [0, 1].
- Noul: ternary no/yes/unknown (Jev's "Noul"), because abstention is a
  first-class answer, not a failure mode.
All carry a confidence that is a CALIBRATION CLAIM: downstream code may
threshold on it, and the training pipeline (decision_head.py, P1) is
required to keep that claim honest (Brier/ECE measured, never assumed).

pi-warden preset: the four guardrail questions from the open-source
pi-warden project (is_irreversible / off_task / mutates / out_of_scope)
plus our route Choice — one batch, one state.
"""
from dataclasses import dataclass, field

NOUL_YES, NOUL_NO, NOUL_UNKNOWN = "yes", "no", "unknown"


class DecisionError(Exception):
    """Schema violation — typed output space makes these impossible by
    construction at the boundary; violations here are programmer errors."""


def _check_conf(c):
    if not (0.0 <= float(c) <= 1.0):
        raise DecisionError(f"confidence {c} out of [0,1]")


@dataclass(frozen=True)
class Choice:
    question: str
    options: tuple
    answer: str
    confidence: float

    def __post_init__(self):
        if self.answer not in self.options:
            raise DecisionError(
                f"Choice answer {self.answer!r} not in options {self.options}")
        _check_conf(self.confidence)

    def __str__(self):
        return f"Choice({self.question!r}={self.answer} p={self.confidence:.2f})"


@dataclass(frozen=True)
class Score:
    question: str
    value: float
    confidence: float

    def __post_init__(self):
        if not (0.0 <= float(self.value) <= 1.0):
            raise DecisionError(f"Score value {self.value} out of [0,1]")
        _check_conf(self.confidence)


@dataclass(frozen=True)
class Noul:
    """Continuous Noul (v5.36): P(yes) in [0,1]. No separate confidence
    -- the probability is the answer; near 0.5 means uncertain
    (official TypeSafe form, adopted after our 1/K floor finding)."""
    question: str
    p_yes: float

    def __post_init__(self):
        if not (0.0 <= float(self.p_yes) <= 1.0):
            raise DecisionError(f"Noul p_yes {self.p_yes} out of [0,1]")

    @property
    def answer(self) -> str:
        if self.p_yes > 0.5:
            return NOUL_YES
        if self.p_yes < 0.5:
            return NOUL_NO
        return NOUL_UNKNOWN

    @property
    def confidence(self) -> float:
        """Distance from the 0.5 uncertainty point -- how decided the
        answer is, derived (not independent)."""
        return abs(self.p_yes - 0.5) * 2.0


# The pi-warden guardrail preset (open-source Jev userland pattern),
# adapted to our tool set.
PI_WARDEN_QUESTIONS = (
    ("is_irreversible", "noul",
     "Could executing this call cause irreversible harm?"),
    ("off_task", "noul",
     "Does this call drift from the user's stated task?"),
    ("mutates", "noul",
     "Does this call mutate state beyond the task's sandbox?"),
    ("out_of_scope", "noul",
     "Does this call touch resources outside the declared scope?"),
    ("route", "choice", "How should the gate route this call?"),
)


def pi_warden_batch(call, ctx, gate) -> dict:
    """Answer the full pi-warden preset for one pending tool call.

    Default implementation asks serially through the existing gate
    (which keeps its T11-T19 contract); a System-One head can override
    ask_batch() to answer all questions in ONE forward pass (P1 hook).
    Returns {question_name: Noul|Choice}.
    """
    ask = getattr(gate, "ask_batch", None)
    if ask is not None:
        return ask(call, ctx, PI_WARDEN_QUESTIONS)
    out = {}
    for name, kind, text in PI_WARDEN_QUESTIONS:
        out[name] = gate.ask(call, ctx, kind, text)
    return out


def choice_from_route(route, confidence, question="route",
                      options=("DIRECT", "ESCALATE")) -> Choice:
    """Typed wrapper over the existing gate route: the route enum becomes
    a calibrated Choice (the gate's ctx['confidence'] is the calibration
    claim; downstream code thresholds on it — that claim is measured in
    three-modes artifacts and trained honest in decision_head.py)."""
    return Choice(question=question, options=tuple(options),
                  answer=route.value if hasattr(route, "value") else str(route),
                  confidence=float(confidence if confidence is not None else 0.5))
