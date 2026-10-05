"""HeliosLM v5.32 P2 — decision records from verified outcomes (RLCD).

Closes the loop T28 opened: the Brier calibration term only bites when
its target is an OUTCOME signal the softmax cannot memorize. Here the
outcome is produced exactly the way HeliosLM produces every other
trustworthy number: run the trajectory, then verify it against the env.

- RecordingGate wraps any gate and logs each routing decision with its
  confidence claim; records_from_runs() attaches the replay-adjacent
  outcome (env.verify on the final answer) as route__target_conf.
- rlcd_reward_adjustment() is the pure reward-shaping term for the
  future agentic-GRPO loop (milestone 13): reward -= lam * Brier(conf,
  outcome). Kept side-effect-free so it composes with any trainer.

State text (recorded deviation): the head sees task text + tool name at
decision time. The production System-One head would see the full
transcript encoding; the interface is identical.
"""
from dataclasses import dataclass, field

import torch

try:
    from .gate import Gate
    from .trajectory import text_to_ids
except ImportError:
    from gate import Gate
    from trajectory import text_to_ids


@dataclass
class RecordingGate(Gate):
    """Wraps an inner gate; logs (state, call, confidence, route)."""
    inner: Gate
    log: list = field(default_factory=list)

    def decide(self, call, context):
        route = self.inner.decide(call, context)
        self.log.append({
            "task": context.get("task", ""),
            "call_name": call.name,
            "confidence": context.get("confidence"),
            "route": route.value,
            "step": context.get("step"),
        })
        return route

    def escalate(self, call, context):
        return self.inner.escalate(call, context)

    # typed interface delegates too (v5.32 additive contract)
    def ask(self, call, context, kind, question):
        return self.inner.ask(call, context, kind, question)


def records_from_runs(gate_log, outcomes):
    """gate_log: RecordingGate.log entries. outcomes: {task_text: bool}
    from env.verify. Returns DecisionHead records with outcome-targeted
    calibration: route__target_conf = 1.0 iff the run's final answer was
    correct, else 0.0 (the RLCD signal T28 proved necessary)."""
    records = []
    for e in gate_log:
        text = f"{e['task']} [{e['call_name']}]"
        correct = bool(outcomes.get(e["task"], False))
        records.append({
            "ids": torch.tensor(text_to_ids(text)),
            "answers": {"route": e["route"],
                        "route__target_conf": 1.0 if correct else 0.0},
        })
    return records


def rlcd_reward_adjustment(rewards, confidences, outcomes, lam=1.0):
    """Pure RLCD shaping: subtract lam * Brier(confidence, outcome) from
    each reward. rewards/confidences/outcomes are parallel sequences;
    confidences may contain None (untimed claims) which are skipped."""
    out = []
    for r, c, o in zip(rewards, confidences, outcomes):
        pen = 0.0 if c is None else (float(c) - float(o)) ** 2
        out.append(float(r) - lam * pen)
    return out
