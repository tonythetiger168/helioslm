"""v5.42 — mini-benchmark harness oracles (Phase 3.3).

Run from repo root: python helioslm_v5/tests/test_mini_bench.py
"""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.eval.mini_bench import (SCORING_TAG, load_tasks,
                                         run_mini_bench)


class _BiasStub(nn.Module):
    """Content-blind per-token bias; installing the correct answer's bias
    high makes every task correct — a perfect-scoring fixture."""

    def __init__(self, vocab=64, bias=None):
        super().__init__()
        b = torch.zeros(vocab) if bias is None else bias
        self.bias = nn.Parameter(b, requires_grad=False)
        self.config = type("C", (), {"vocab_size": vocab})()

    def forward(self, ids):
        B, T = ids.shape
        return self.bias.expand(B, T, -1), None, None


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                    encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def test_load_tasks_schema_and_loud_errors():
    good = _write_jsonl([
        {"task_id": "a", "prompt_ids": [1], "choices": [2, 3],
         "answer_idx": 0, "benchmark": "math"},
        {"task_id": "b", "prompt": "Q?", "choices": ["x", "y"],
         "answer_idx": 1},
    ])
    tasks = load_tasks(good)
    assert len(tasks) == 2 and tasks[0]["benchmark"] == "math"
    bad_cases = [
        [{"task_id": "a", "choices": [2], "answer_idx": 0}],        # no prompt
        [{"task_id": "a", "prompt_ids": [1], "choices": [2],
          "answer_idx": 5}],                                        # bad idx
        [{"task_id": "a", "prompt_ids": [1], "choices": [],
          "answer_idx": 0}],                                        # empty
        [{"task_id": "a", "prompt_ids": [1], "choices": [2],
          "answer_idx": 0}] * 2,                                    # dup id
    ]
    for rows in bad_cases:
        path = _write_jsonl(rows)
        try:
            load_tasks(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad task file accepted: {rows}")
    empty = _write_jsonl([])
    try:
        load_tasks(empty)
    except ValueError:
        pass
    else:
        raise AssertionError("empty task file accepted")
    print("PASS test_load_tasks_schema_and_loud_errors schema + 5 bad cases")


def test_perfect_stub_scores_all_benchmarks():
    bias = torch.zeros(64)
    bias[3] = 10.0                      # choice token 3 always wins
    stub = _BiasStub(bias=bias)
    rows = [{"task_id": f"t{i}", "prompt_ids": [1, 2],
             "choices": [3, 4], "answer_idx": 0,
             "benchmark": "math" if i % 2 else "code"} for i in range(10)]
    rep = run_mini_bench(stub, load_tasks(_write_jsonl(rows)),
                         tag="bias-stub")
    assert rep["scoring"] == SCORING_TAG
    assert rep["overall"]["accuracy"] == 1.0 and rep["overall"]["n"] == 10
    assert rep["per_benchmark"]["math"]["accuracy"] == 1.0
    assert rep["per_benchmark"]["code"]["accuracy"] == 1.0
    print("PASS test_perfect_stub_scores_all_benchmarks 10/10, "
          "per-benchmark split exact")


def test_accuracy_is_measured_not_assumed():
    """Swap the answer so the stub's preferred token is always WRONG:
    accuracy must drop to 0.0 — the harness genuinely reads answer_idx."""
    bias = torch.zeros(64)
    bias[3] = 10.0
    stub = _BiasStub(bias=bias)
    rows = [{"task_id": f"t{i}", "prompt_ids": [1],
             "choices": [3, 4], "answer_idx": 1} for i in range(6)]
    rep = run_mini_bench(stub, load_tasks(_write_jsonl(rows)))
    assert rep["overall"]["accuracy"] == 0.0, rep
    print("PASS test_accuracy_is_measured_not_assumed wrong-answer "
          "layout -> 0.0 accuracy")


if __name__ == "__main__":
    test_load_tasks_schema_and_loud_errors()
    test_perfect_stub_scores_all_benchmarks()
    test_accuracy_is_measured_not_assumed()
    print("\n3/3 mini-bench tests passed")
