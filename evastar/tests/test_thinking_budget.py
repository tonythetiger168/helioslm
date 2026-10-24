"""T112 — roadmap U2: thinking-budget control, v5.73.

Pins:

1. default generate is unchanged (thinking_budget=None is bit-identical,
   greedy and sampled);
2. the budget oracle: on an untrained model (which never closes on its
   own) every row thinks exactly ``b`` tokens and the next token is the
   close marker, for every b, greedy and sampled, batched; a span opened in
   the prompt counts, rows that start outside a span are never forced,
   the open marker is not re-emitted after a close, and a close marker is
   never emitted outside a span;
3. a natural close before the budget is kept as is (state machine unit
   test), and finished (EOS) rows are not advanced;
4. truncate_thinking / random_truncation keep the prompt and the answer
   and cut the span to b tokens; effort_to_budget; validation;
5. the committed A/B artifact: both arms solve the task at the full
   budget; with truncation training accuracy is non-decreasing in the
   budget (small tolerance) and, closed one step early, far above the
   full-trace-only arm; hash.

Run from repo root: python3 evastar/tests/test_thinking_budget.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import EvastarV5Config  # noqa: E402
from evastar.src.inference.thinking_budget import (  # noqa: E402
    ThinkingBudget, effort_to_budget, random_truncation, truncate_thinking)
from evastar.src.model_v5 import EvastarV5  # noqa: E402

_ART = _REPO / "benchmarks" / "thinking_budget_ab_2026-10-17.json"
OPEN, CLOSE = 7, 8


def _model():
    torch.manual_seed(0)
    return EvastarV5(EvastarV5Config(size="lite")).eval()


def test_default_unchanged():
    m = _model()
    ids = torch.randint(10, 900, (2, 6))
    for temp in (0, 0.8):
        torch.manual_seed(1)
        a = m.generate(ids, max_new_tokens=8, temperature=temp)
        torch.manual_seed(1)
        b = m.generate(ids, max_new_tokens=8, temperature=temp,
                       thinking_budget=None)
        assert torch.equal(a, b)
    print("PASS test_default_unchanged")


def test_budget_oracle():
    m = _model()
    prompt = torch.randint(10, 900, (3, 5))
    prompt[:, -1] = OPEN                         # span opened in the prompt
    for b in (0, 1, 3, 6):
        for temp in (0, 1.0):
            torch.manual_seed(b)
            out = m.generate(prompt, max_new_tokens=b + 4, temperature=temp,
                             thinking_budget=ThinkingBudget(OPEN, CLOSE, b))
            gen = out[:, 5:].tolist()
            for row in gen:
                assert row.index(CLOSE) == b, (b, temp, row)
                assert OPEN not in row[b:], row          # no re-open
                assert row.count(CLOSE) == 1, row        # no stray close
    # rows that are not thinking are never forced
    plain = torch.randint(10, 900, (2, 5))
    out = m.generate(plain, max_new_tokens=6, temperature=0,
                     thinking_budget=ThinkingBudget(OPEN, CLOSE, 0))
    assert CLOSE not in out[:, 5:].tolist()[0] + out[:, 5:].tolist()[1]
    print("PASS test_budget_oracle")


def test_state_machine():
    tb = ThinkingBudget(OPEN, CLOSE, 5).start(torch.tensor([[1, OPEN], [1, 2]]))
    assert tb.inside.tolist() == [True, False]
    tb.update(torch.tensor([3, OPEN]))          # row 1 opens a span
    tb.update(torch.tensor([CLOSE, 4]))         # row 0 closes naturally (1 token)
    assert tb.inside.tolist() == [False, True] and tb.used.tolist() == [1, 1]
    lg = tb.process(torch.zeros(2, 12))
    assert lg[0, OPEN] == float("-inf") and torch.isfinite(lg[1]).all()
    # an inactive (finished) row is frozen
    tb.update(torch.tensor([OPEN, 4]), active=torch.tensor([False, True]))
    assert tb.inside.tolist() == [False, True] and tb.used.tolist() == [1, 2]
    # reopen allowed
    tb2 = ThinkingBudget(OPEN, CLOSE, 2, allow_reopen=True).start(
        torch.tensor([[OPEN, 3, CLOSE]]))
    assert not tb2.inside.item()
    assert torch.isfinite(tb2.process(torch.zeros(1, 12))[0, OPEN])
    print("PASS test_state_machine")


def test_truncation_helpers():
    seq = [1, 5, 6, OPEN, 20, 21, 22, 23, CLOSE, 9, 2]
    assert truncate_thinking(seq, OPEN, CLOSE, 2) == [1, 5, 6, OPEN, 20, 21, CLOSE, 9, 2]
    assert truncate_thinking(seq, OPEN, CLOSE, 0) == [1, 5, 6, OPEN, CLOSE, 9, 2]
    assert truncate_thinking(seq, OPEN, CLOSE, 99) == seq
    assert truncate_thinking([1, OPEN, 3], OPEN, CLOSE, 1) == [1, OPEN, 3]
    rng = random.Random(0)
    lens = set()
    for _ in range(200):
        t = random_truncation(seq, OPEN, CLOSE, 1.0, rng)
        assert t[:4] == seq[:4] and t[-3:] == seq[-3:]
        lens.add(t.index(CLOSE) - 4)
    assert lens == {0, 1, 2, 3, 4}
    assert random_truncation(seq, OPEN, CLOSE, 0.0, rng) == seq
    assert [effort_to_budget(e, 100) for e in ("low", "medium", "high")] == [25, 50, 100]
    for bad in (lambda: ThinkingBudget(OPEN, CLOSE, -1),
                lambda: ThinkingBudget(OPEN, CLOSE, True),
                lambda: ThinkingBudget(OPEN, OPEN, 3),
                lambda: effort_to_budget("max", 10),
                lambda: _model().generate(torch.ones(1, 2, dtype=torch.long),
                                          use_mtp=True,
                                          thinking_budget=ThinkingBudget(OPEN, CLOSE, 1))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("PASS test_truncation_helpers")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = {r["arm"]: r["acc_by_budget"] for r in art["results"]}
    n = art["task"]["digits"]
    tr = [res["truncation"][str(b)] for b in range(n + 1)]
    full = [res["full"][str(b)] for b in range(n + 1)]
    assert all(b2 >= b1 - 0.03 for b1, b2 in zip(tr, tr[1:])), tr
    assert tr[0] < 0.4 and tr[-1] > 0.9, tr
    # both arms learned the task (the comparison is meaningless otherwise)
    assert full[-1] > 0.9, full
    # closed one step early: only truncation training can still answer
    # (full-trace training never saw a forced close)
    assert tr[n - 1] - full[n - 1] > 0.5, (tr, full)
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact " + json.dumps({"truncation": tr, "full": full}))


if __name__ == "__main__":
    test_default_unchanged()
    test_budget_oracle()
    test_state_machine()
    test_truncation_helpers()
    test_ab_artifact()
