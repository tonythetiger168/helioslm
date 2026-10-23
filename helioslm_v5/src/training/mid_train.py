"""Mid-training data: reasoning traces mixed into the decay phase
(v5.121, LB5 of the 2026-10-23 leaderboard review).

K2 Horizon puts ~17% reasoning traces into pretraining and MiniCPM5
switches to curated, instruction- and reasoning-heavy data for its decay
phase. The LB2 WSD schedule already swaps the training shard at the
first decay step (``pretrain(decay_train=...)``); this module builds that
shard from a mixture manifest's ``decay`` sources:

- ``render_trace``: a D1 record (question, teacher reasoning, answer) as
  one plain-text document in the student's chat format
  (``distill_sft.format_example``, with the LB6 ``/think`` switch), so
  the base model has seen the transcript format and long chains of
  thought before SFT. Unlike SFT every token is trained on;
- ``mix_docs``: interleaves the sources' tokenised documents to their
  token budgets by smooth weighted round robin (the source furthest
  behind its budget goes next), each source cycling its documents in a
  seeded shuffle per epoch. A source that would need more than
  ``max_epochs`` passes stops early and is reported as short, never
  silently over-repeated.

No torch; ``examples/build_midtrain.py`` is the command line.
"""
import random

from .distill_sft import format_example

FORMATS = ("text", "trace")


def render_trace(rec, think_ratio=1.0, mode_switch=True):
    """One D1 record as a training document (prompt + target text)."""
    keep = "all" if think_ratio >= 1.0 else (
        "none" if think_ratio <= 0.0 else "mix")
    prompt, target = format_example(rec, keep, think_ratio, mode_switch)
    return prompt + target


def mix_docs(sources, budgets, seed=0, max_epochs=4):
    """Interleave tokenised documents to per-source token budgets.

    ``sources``: {name: [token-id list, ...]}; ``budgets``: {name: tokens}.
    Returns (order, report): ``order`` is a list of (name, ids) and
    ``report[name]`` has tokens, docs, epochs (passes started) and short
    (True when max_epochs ran out before the budget)."""
    if set(sources) != set(budgets):
        raise ValueError("sources and budgets must name the same sources")
    rng = random.Random(seed)
    state = {}
    for name in sorted(sources):
        docs = [d for d in sources[name] if len(d)]
        if budgets[name] <= 0:
            raise ValueError(f"{name}: budget must be > 0")
        if not docs:
            raise ValueError(f"{name}: no non-empty documents")
        state[name] = {"docs": docs, "perm": [], "pos": 0, "epochs": 0,
                       "tokens": 0, "n": 0, "short": False}
    order = []
    while True:
        live = [n for n, s in state.items()
                if s["tokens"] < budgets[n] and not s["short"]]
        if not live:
            break
        name = min(live, key=lambda n: (state[n]["tokens"] / budgets[n], n))
        s = state[name]
        if s["pos"] == len(s["perm"]):
            if s["epochs"] == max_epochs:
                s["short"] = True
                continue
            s["perm"] = list(range(len(s["docs"])))
            rng.shuffle(s["perm"])
            s["pos"] = 0
            s["epochs"] += 1
        ids = s["docs"][s["perm"][s["pos"]]]
        s["pos"] += 1
        s["tokens"] += len(ids)
        s["n"] += 1
        order.append((name, ids))
    report = {n: {"tokens": s["tokens"], "docs": s["n"],
                  "epochs": s["epochs"], "short": s["short"]}
              for n, s in state.items()}
    return order, report
