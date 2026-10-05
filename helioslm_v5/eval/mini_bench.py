"""Mini-benchmark harness (Phase 3.3) — task-level JSONL evaluation.

``eval/score_stream.py`` scores free-form (prompt, output) pairs; the
missing piece for benchmark-style claims is a TASK-level runner: a JSONL
of multiple-choice tasks, argmax-by-loglikelihood scoring (no generation
— CPU-deterministic), and per-benchmark aggregation with provenance.

Task record schema (one JSON object per line):
    {"task_id": "math-001",                       # required, unique
     "benchmark": "math",                          # optional; "" = overall only
     "prompt": "text or omitted",                  # text (char-level encode)
     "prompt_ids": [12, 13],                       # or explicit ids
     "choices": ["42", "51"],                      # text or token-id ints
     "answer_idx": 0}                              # index into choices

Honesty constraints:
  - Scoring is loglikelihood-argmax; it measures the model's own ranking,
    not agentic capability. The report carries "scoring": "choice-loglikelihood"
    so the number can never be quoted as more than that.
  - Tasks with out-of-range answer_idx or empty choices fail LOUDLY at
    load time — a benchmark that silently dropped tasks is a fake number.
"""
import json
from typing import Dict, List, Optional, Union

import torch

from helioslm_v5.eval.quant_calib import choice_confidence
from helioslm_v5.eval.score_stream import _encode

SCORING_TAG = "choice-loglikelihood"


def _to_ids(x: Union[int, str, list], vocab_size: int) -> List[int]:
    if isinstance(x, int):
        return [x]
    if isinstance(x, str):
        return _encode(x, vocab_size)
    ids = list(x)
    if not ids:
        raise ValueError("empty id list in task field")
    return ids


def load_tasks(path: str) -> List[dict]:
    tasks = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            for field in ("task_id", "choices", "answer_idx"):
                if field not in t:
                    raise ValueError(
                        f"task line {lineno}: missing {field!r}")
            if t["task_id"] in seen:
                raise ValueError(
                    f"task line {lineno}: duplicate task_id "
                    f"{t['task_id']!r}")
            seen.add(t["task_id"])
            if not isinstance(t["choices"], list) or not t["choices"]:
                raise ValueError(
                    f"task {t['task_id']!r}: choices must be a non-empty "
                    f"list")
            if not (0 <= int(t["answer_idx"]) < len(t["choices"])):
                raise ValueError(
                    f"task {t['task_id']!r}: answer_idx "
                    f"{t['answer_idx']} out of range for "
                    f"{len(t['choices'])} choices")
            if "prompt_ids" not in t and "prompt" not in t:
                raise ValueError(
                    f"task {t['task_id']!r}: needs prompt or prompt_ids")
            tasks.append(t)
    if not tasks:
        raise ValueError(f"no tasks in {path}")
    return tasks


def run_mini_bench(model, tasks: List[dict], tag: str = "unknown",
                   vocab_size: Optional[int] = None) -> Dict:
    """Score every task, aggregate overall + per-benchmark."""
    if vocab_size is None:
        vocab_size = int(getattr(getattr(model, "config", None),
                                 "vocab_size", 0))
        if vocab_size <= 0:
            raise ValueError("run_mini_bench: pass vocab_size explicitly "
                             "(model config has none)")
    per_bench: Dict[str, Dict[str, int]] = {}
    overall = {"correct": 0, "n": 0}
    for t in tasks:
        p_ids = (_to_ids(t["prompt_ids"], vocab_size)
                 if "prompt_ids" in t
                 else _to_ids(t["prompt"], vocab_size))
        c_ids = [_to_ids(c, vocab_size) for c in t["choices"]]
        probs, _ = choice_confidence(model, p_ids, c_ids)
        ok = int(int(probs.argmax()) == int(t["answer_idx"]))
        bench = t.get("benchmark", "")
        row = per_bench.setdefault(bench, {"correct": 0, "n": 0})
        row["correct"] += ok
        row["n"] += 1
        overall["correct"] += ok
        overall["n"] += 1
    return {
        "tag": tag,
        "scoring": SCORING_TAG,
        "overall": {"accuracy": overall["correct"] / overall["n"],
                    "n": overall["n"]},
        "per_benchmark": {
            k: {"accuracy": v["correct"] / v["n"], "n": v["n"]}
            for k, v in per_bench.items()},
    }
