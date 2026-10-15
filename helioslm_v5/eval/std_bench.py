"""Standard-benchmark harness (v5.55, roadmap Phase A).

The calibration board can only compare runs on our own 12-task probe
suite. This module adds two public, generation-based benchmarks so
HeliosLM numbers can be compared with anyone's:

- GSM8K   : grade-school math word problems, exact numeric match on the
            final answer (``#### <n>`` marker, else the last number).
- HumanEval: Python function synthesis, scored by executing the
            upstream unit tests (pass/fail per task).

Data lives in ``data/std_bench/`` (pinned upstream commits, sha256 in
MANIFEST.json; ``scripts/fetch_std_bench.py`` fetches the full splits).

The harness is model-agnostic: a run is ``generate_fn(prompt, i)`` ->
``text`` or ``(text, conf)``. Results are written in the shared probe
schema (``results[*].correct`` + ``conf``) so every run can go straight
onto the calibration board.

Execution safety: HumanEval completions are untrusted code. Each one
runs in a fresh subprocess (``python -I``) in a temporary directory with
a wall-clock timeout and, on POSIX, CPU / address-space limits. That
stops runaway loops and memory blowups, NOT a hostile program — run
untrusted models' completions inside a container.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA_ID = "std-bench/1"
BENCHES = ("gsm8k", "humaneval")
EXEC_TIMEOUT_S = 10
EXEC_MEMORY_BYTES = 1 << 30   # 1 GiB address space per check
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


class BenchSchemaError(ValueError):
    """Raised for any malformed dataset record. Never swallowed."""


# ------------------------------------------------------------------ loading

def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise BenchSchemaError(f"dataset not found: {path}")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise BenchSchemaError(f"{path.name}:{i + 1}: not JSON: {e}") from e
    if not rows:
        raise BenchSchemaError(f"{path.name}: no records")
    return rows


def load_gsm8k(path):
    """GSM8K records -> [{task_id, question, answer}] (answer normalized)."""
    out = []
    for i, row in enumerate(_read_jsonl(path)):
        q, a = row.get("question"), row.get("answer")
        if not isinstance(q, str) or not isinstance(a, str) or "####" not in a:
            raise BenchSchemaError(
                f"gsm8k[{i}]: needs question and an answer with '####'")
        gold = normalize_number(a.split("####")[-1])
        if gold is None:
            raise BenchSchemaError(f"gsm8k[{i}]: gold answer is not a number")
        out.append({"task_id": f"gsm8k/{i}", "question": q, "answer": gold})
    return out


def load_humaneval(path):
    """HumanEval records, with the five upstream keys checked."""
    out = []
    for i, row in enumerate(_read_jsonl(path)):
        for key in ("task_id", "prompt", "entry_point", "canonical_solution",
                    "test"):
            if not isinstance(row.get(key), str):
                raise BenchSchemaError(f"humaneval[{i}]: missing {key!r}")
        out.append(row)
    return out


# ------------------------------------------------------------------- GSM8K

def normalize_number(text):
    """'1,234.50' / '$18' / '18.0' -> canonical string, else None."""
    m = _NUMBER.search(text or "")
    if not m:
        return None
    raw = m.group(0).replace(",", "").replace("$", "")
    try:
        d = Decimal(raw)
    except InvalidOperation:
        return None
    d = d.normalize()
    if d == d.to_integral():
        d = d.quantize(Decimal(1))
    return format(d, "f")


def extract_gsm8k_answer(text):
    """The '#### n' marker wins; otherwise the LAST number in the text."""
    text = text or ""
    if "####" in text:
        return normalize_number(text.split("####")[-1])
    nums = _NUMBER.findall(text)
    return normalize_number(nums[-1]) if nums else None


def gsm8k_prompt(task):
    return ("Solve the following grade-school math problem. Show your "
            "reasoning, then give the final answer on its own line as "
            "'#### <number>'.\n\n"
            f"Question: {task['question']}\nAnswer:")


# --------------------------------------------------------------- HumanEval

def humaneval_prompt(task):
    return ("Complete the following Python function. Reply with the full "
            "function in a single ```python code block.\n\n"
            f"```python\n{task['prompt']}```")


def extract_program(completion, task):
    """Turn a completion into a runnable program.

    - a fenced code block is unwrapped (the first python block wins);
    - if the code defines the entry point, it is used as-is (with the
      prompt's imports prepended);
    - otherwise it is treated as a function BODY and appended to the
      prompt — the upstream HumanEval completion format.
    """
    text = completion or ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    if re.search(rf"^\s*def\s+{re.escape(task['entry_point'])}\s*\(",
                 code, re.M):
        header = "\n".join(ln for ln in task["prompt"].splitlines()
                           if ln.startswith(("import ", "from ")))
        return (header + "\n\n" + code) if header else code
    return task["prompt"] + code


def _limit_resources():   # pragma: no cover - runs in the child process
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_S, EXEC_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS,
                       (EXEC_MEMORY_BYTES, EXEC_MEMORY_BYTES))


def run_humaneval_check(task, program, timeout=EXEC_TIMEOUT_S):
    """Execute program + upstream tests in a sandboxed subprocess.

    Returns (passed, detail) where detail is 'passed', 'timeout', or
    the last line of stderr."""
    source = (program + "\n\n" + task["test"] + "\n\n"
              + f"check({task['entry_point']})\n")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=td,
                               capture_output=True, text=True,
                               timeout=timeout, env={}, **kwargs)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if r.returncode == 0:
        return True, "passed"
    tail = (r.stderr.strip().splitlines() or ["exit code %d" % r.returncode])
    return False, tail[-1][:200]


# --------------------------------------------------------------------- runs

def _call(generate_fn, prompt, i):
    out = generate_fn(prompt, i)
    if isinstance(out, tuple):
        text, conf = out
    else:
        text, conf = out, None
    if conf is not None and not 0.0 <= float(conf) <= 1.0:
        raise BenchSchemaError(f"generate_fn returned conf {conf} outside [0, 1]")
    return ("" if text is None else str(text)), conf


def run_gsm8k(tasks, generate_fn, n_samples=1):
    """Majority answer over n samples; conf = model conf of the first
    sample if given, else self-consistency (n_samples > 1), else None."""
    results = []
    for t in tasks:
        prompt = gsm8k_prompt(t)
        outs = [_call(generate_fn, prompt, i) for i in range(n_samples)]
        answers = [extract_gsm8k_answer(text) for text, _ in outs]
        voted = [a for a in answers if a is not None]
        final, conf, conf_type = None, None, None
        if voted:
            final, cnt = Counter(voted).most_common(1)[0]
        if outs[0][1] is not None:
            conf, conf_type = float(outs[0][1]), "model"
        elif n_samples > 1:
            conf = (cnt / n_samples) if voted else 0.0
            conf_type = "self_consistency"
        results.append({
            "family": "gsm8k", "task": t["task_id"],
            "expected": t["answer"], "final": final,
            "correct": final is not None and final == t["answer"],
            "conf": conf, "conf_type": conf_type, "n_samples": n_samples,
        })
    return results


def run_humaneval(tasks, generate_fn, n_samples=1, timeout=EXEC_TIMEOUT_S):
    """``correct`` is sample 0 passing; ``pass_at_1`` is the unbiased
    c/n estimate over all samples; conf is the model's own (if any)."""
    results = []
    for t in tasks:
        prompt = humaneval_prompt(t)
        outs = [_call(generate_fn, prompt, i) for i in range(n_samples)]
        checks = [run_humaneval_check(t, extract_program(text, t), timeout)
                  for text, _ in outs]
        passed = [ok for ok, _ in checks]
        results.append({
            "family": "humaneval", "task": t["task_id"],
            "expected": "tests pass", "final": checks[0][1],
            "correct": passed[0],
            "pass_at_1": sum(passed) / n_samples,
            "conf": outs[0][1],
            "conf_type": "model" if outs[0][1] is not None else None,
            "n_samples": n_samples,
        })
    return results


def summarize(results):
    n = len(results)
    if not n:
        raise BenchSchemaError("summarize: no results")
    c = sum(1 for r in results if r["correct"])
    out = {"n": n, "correct": f"{c}/{n}", "accuracy": round(c / n, 4)}
    if results[0].get("family") == "humaneval":
        out["pass_at_1"] = round(sum(r["pass_at_1"] for r in results) / n, 4)
    return out
