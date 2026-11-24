"""Route A: verified trajectories from fresh, eval-disjoint agentic tasks.

This module is deliberately separate from the formal benchmark runners.  It
creates parameterized *training* tasks for terminal, SWE, long-horizon and
tool-recovery protocols, pairs each teacher reply with the exact candidate-
facing prompt for that step, and exports only verifier-accepted trajectories.
Existing reference
artifacts are not valid input: they describe evaluation episodes and do not
carry this module's task provenance and per-step text.

Teacher output and benchmark-derived code are untrusted.  Shell tasks must be
collected in a disposable container; the CLI requires an explicit container
confirmation.  Export applies the structured-data quality heuristics and the
13-gram evaluation-overlap gate before writing anything for ``sft_hf.py``.
No model training is started here.
"""
import hashlib
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

from ...agent.gate import FixedGate, Route
from ...agent.loop import AgentLoop
from ...agent.oracle_guard import OracleGuard
from ...agent.schema import (ToolCall, ToolCallError, ToolRegistry, ToolSpec,
                             parse_tool_call)
from ...agent.tools import build_default_registry
from ...agent.trajectory import ReplayMismatch, ids_to_text, verify_replay
from ...eval import long_horizon_bench as lh
from ...eval import term_bench as tb
from ...eval.decontam import EvalIndex
from .data_quality import heuristic_check

TRAIN_TASK_SET = "agentic-distill-disjoint-v1"
PROTOCOLS = ("terminal", "swe", "long_horizon", "tool_recovery")
QUALITY_RULES = {"min_units": 8, "max_units": 50_000}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def task_sha256(task):
    return hashlib.sha256(_canonical(task).encode("utf-8")).hexdigest()


def _base(task_id, protocol, seed):
    return {"id": task_id, "protocol": protocol, "task_set": TRAIN_TASK_SET,
            "synthetic": True, "seed": seed, "requires_container": True}


def _terminal_task(rng, seed, i):
    stores = ["北門", "西門", "松山", "板橋", "淡水", "中壢"]
    rows = [(rng.choice(stores), rng.randint(11, 999))
            for _ in range(rng.randint(9, 18))]
    store = rng.choice(sorted({s for s, _ in rows}))
    csv = "門市,金額\n" + "".join(f"{s},{amount}\n" for s, amount in rows)
    total = sum(amount for s, amount in rows if s == store)
    task = _base(f"train-terminal-{seed}-{i}", "terminal", seed)
    task.update({
        "instruction": f"sales.csv 是門市銷售資料。請計算「{store}」列的金額總和，"
                       "並只把該整數寫入 answer.txt。完成後回報 DONE。",
        "files": {"sales.csv": csv},
        "check": {"type": "file_equals", "path": "answer.txt",
                  "value": str(total)},
    })
    return task


def _swe_task(rng, seed, i):
    # A fresh bounded-moving-average repair task.  The visible test misses
    # the empty-window and non-truncating cases; the hidden test covers both.
    size = rng.randint(2, 6)
    values = [rng.randint(-20, 40) for _ in range(size + 3)]
    task = _base(f"train-swe-{seed}-{i}", "swe", seed)
    source = (
        "def moving_mean(values, width):\n"
        "    if width <= 0:\n"
        "        raise ValueError('width must be positive')\n"
        "    return [sum(values[j:j + width]) // width\n"
        "            for j in range(len(values) - width)]\n")
    visible = (
        "from metric import moving_mean\n"
        "assert moving_mean([2, 4, 6], 2) == [3.0, 5.0]\n")
    hidden = (
        "from metric import moving_mean\n"
        f"values = {values!r}\n"
        f"width = {size}\n"
        "want = [sum(values[j:j + width]) / width\n"
        "        for j in range(len(values) - width + 1)]\n"
        "assert moving_mean(values, width) == want\n"
        "assert moving_mean([7], 1) == [7.0]\n"
        "try:\n"
        "    moving_mean([1], 0)\n"
        "    raise AssertionError('zero width accepted')\n"
        "except ValueError:\n"
        "    pass\n")
    task.update({
        "instruction": "這個 Python 專案的 moving_mean 有邊界與整數除法錯誤。"
                       "請只修改 metric.py，讓公開及隱藏測試通過；不可修改 tests/。"
                       "輸出最後一個視窗的平均，並對零寬度 raising ValueError。",
        "files": {"metric.py": source, "tests/test_visible.py": visible},
        "check": {"type": "python_tests", "visible": ["tests/test_visible.py"],
                  "hidden": {"_route_a_hidden_test.py": hidden}},
    })
    return task


def _long_horizon_task(rng, seed, i):
    a, b, c = (rng.randint(3, 15) for _ in range(3))
    product, offset = a * b, b + c
    answer = str(product + offset)
    task = _base(f"train-long-horizon-{seed}-{i}", "long_horizon", seed)
    task.update({
        "scenario": {
            "task": f"First evaluate the product {a} * {b}; save that number "
                    f"in product.txt. Next evaluate {b} + {c}; save that number "
                    "in offset.txt. After reading both outputs, calculate their "
                    "sum and store it in total.txt. Read total.txt once more, "
                    "then finish with exactly that value. Never access oracle/.", 
            "answer": answer, "budget": 12,
            "files": {"oracle/answer.txt": answer},
            "expect_files": {"product.txt": str(product),
                             "offset.txt": str(offset), "total.txt": answer},
            "protected": ["oracle"],
        }
    })
    return task


TRAIN_TOOL_REGISTRY = ToolRegistry([
    ToolSpec("lookup", "look up a training record", {"key": str}),
    ToolSpec("transform", "transform text", {"path": str, "count": int,
                                             "strict": bool}),
])


def _training_calls(rng):
    pool = [
        ToolCall("lookup", {"key": rng.choice(["alpha-train", "β-train",
                                                "key with spaces"])}),
        ToolCall("transform", {"path": rng.choice(["train/a.txt", "train/b c.txt"]),
                               "count": rng.randint(0, 9),
                               "strict": rng.random() < 0.5}),
    ]
    return rng.sample(pool, rng.randint(1, 2))


def _corrupt_training_call(calls, kind):
    body = json.dumps({"calls": [{"name": c.name, "args": c.args}
                                 for c in calls]}, ensure_ascii=False)
    if kind == "markdown_fence":
        return "訓練用損壞輸出：\n```json\n" + body + "\n```"
    if kind == "openai_arguments":
        return "@@tool@@" + json.dumps({"calls": [
            {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}
            for c in calls]}, ensure_ascii=False) + "@@end@@"
    if kind == "bare_list":
        return "@@tool@@" + json.dumps([{"name": c.name, "args": c.args}
                                        for c in calls], ensure_ascii=False) + "@@end@@"
    if kind == "missing_close":
        return "@@tool@@" + body
    raise ValueError(f"unknown training corruption {kind!r}")


def _tool_recovery_task(rng, seed, i):
    kinds = ("markdown_fence", "openai_arguments", "bare_list", "missing_close")
    kind = kinds[i % len(kinds)]
    calls = _training_calls(rng)
    malformed = _corrupt_training_call(calls, kind)
    task = _base(f"train-tool-recovery-{seed}-{i}", "tool_recovery", seed)
    task.update({
        "prompt": "請修復下列訓練用工具呼叫，只輸出符合 schema 的完整工具區塊，"
                  "不要解釋，也不要真的執行工具。\n\n" + malformed,
        "check": {"type": "tool_calls",
                  "calls": [{"name": c.name, "args": c.args} for c in calls]},
        "corruption": kind,
    })
    return task


def make_tasks(per_protocol=4, seed=0):
    """Deterministic fresh tasks; no formal eval task objects are reused."""
    if type(per_protocol) is not int or per_protocol < 1:
        raise ValueError("per_protocol must be a positive integer")
    rng = random.Random(seed)
    tasks = []
    makers = {"terminal": _terminal_task, "swe": _swe_task,
              "long_horizon": _long_horizon_task,
              "tool_recovery": _tool_recovery_task}
    for i in range(per_protocol):
        for protocol in PROTOCOLS:
            tasks.append(makers[protocol](rng, seed, i))
    return tasks


def task_text(task):
    """Task-specific payload for overlap checks (not shared system boilerplate)."""
    parts = [task.get("instruction", ""), task.get("prompt", "")]
    parts.extend(task.get("files", {}).values())
    if "scenario" in task:
        sc = task["scenario"]
        parts += [sc["task"], *sc.get("files", {}).values()]
    if task["protocol"] == "tool_recovery":
        parts.append(_canonical(task["check"]["calls"]))
    return "\n".join(parts)


def builtin_eval_items():
    """Task-specific text from the four formal agentic evaluation sets.

    Shared protocol instructions are intentionally excluded: eval questions,
    fixtures and malformed payloads are the contamination units, not the shell
    marker documentation common to every episode.
    """
    from ...eval import swe_bench as sb

    items = {}
    for task in tb.load_tasks():
        items[f"terminal:{task['id']}"] = task["instruction"] + "\n" + "\n".join(
            task.get("files", {}).values())
    for task in sb.TASKS:
        # Include the hidden oracle text as well as the visible issue/files;
        # load_tasks() intentionally strips it from the model-facing adapter.
        items[f"swe:{task['id']}"] = "\n".join(
            [task["issue"], *task.get("files", {}).values(), task["hidden"]])
    for sc in lh.SCENARIOS:
        items[f"long_horizon:{sc.id}"] = sc.task + "\n" + "\n".join(sc.files.values())

    # Keep this byte-compatible with agentic_candidate_run.run_tool_recovery,
    # but import lazily so production data export does not need test modules.
    from ...tests.test_tool_recovery import REG, REPAIRABLE, _sample_calls
    del REG  # only the corpus and its strict parser contract are needed here
    rng = random.Random(20261123)
    for cls, (corrupt, _repair) in sorted(REPAIRABLE.items()):
        for i in range(10):
            calls = _sample_calls(rng)
            items[f"tool_recovery:{cls}:{i}"] = corrupt(calls)
    return items


def combined_eval_items(extra=None):
    items = builtin_eval_items()
    for key, value in (extra or {}).items():
        if key in items and items[key] != value:
            raise ValueError(f"duplicate eval item id with different text: {key}")
        items[key] = value
    return items


def assert_tasks_disjoint(tasks, eval_items):
    if not eval_items:
        raise ValueError("a non-empty eval index is required before collection")
    index = EvalIndex(eval_items)
    hits = {task["id"]: sorted(index.scan(task_text(task))) for task in tasks}
    hits = {key: value for key, value in hits.items() if value}
    if hits:
        raise ValueError(f"training tasks overlap formal evaluation items: {hits}")
    return index.report(task_text(task) for task in tasks)


def _record(task, verified, steps, verification):
    return {"id": task["id"], "protocol": task["protocol"],
            "kind": task["protocol"], "task_set": task["task_set"],
            "synthetic": True, "task_sha256": task_sha256(task),
            "verified": bool(verified), "steps": steps,
            "verification": verification}


def _file_check(task, root):
    check = task["check"]
    path = Path(root) / check["path"]
    got = path.read_text(encoding="utf-8") if path.is_file() else None
    ok = got is not None and got.strip() == check["value"].strip()
    return ok, {"type": "file_equals", "path": check["path"], "passed": ok}


def _swe_check(task, root, command_runner):
    check = task["check"]
    with tempfile.TemporaryDirectory(prefix="route_a_swe_verify_") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(root, work)
        for rel in check["visible"]:
            original = task["files"][rel]
            got = (Path(root) / rel).read_text(encoding="utf-8") \
                if (Path(root) / rel).is_file() else None
            if got != original:
                return False, {"type": "python_tests", "passed": False,
                               "reason": f"visible test modified: {rel}"}
            (work / rel).write_text(original, encoding="utf-8", newline="\n")
        for rel, text in check["hidden"].items():
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
        commands = [f"python3 {rel}" for rel in
                    list(check["visible"]) + list(check["hidden"])]
        for command in commands:
            code, output = command_runner(command, work)
            if code != 0:
                return False, {"type": "python_tests", "passed": False,
                               "failed_command": command,
                               "output_tail": output.strip()[-200:]}
    return True, {"type": "python_tests", "passed": True,
                  "visible": len(check["visible"]), "hidden": len(check["hidden"])}


def collect_shell_task(task, teacher, sample=0, max_steps=tb.MAX_STEPS,
                       command_runner=None):
    """Collect one terminal/SWE episode, retaining exact prompts and replies."""
    if task["protocol"] not in ("terminal", "swe"):
        raise ValueError("collect_shell_task requires terminal or swe")
    run_command = command_runner or tb.run_command
    runtime = {"id": task["id"], "instruction": task["instruction"]}
    with tempfile.TemporaryDirectory(prefix="route_a_shell_") as tmp:
        root = Path(tmp)
        tb.seed_sandbox({"id": task["id"], "files": task["files"]}, root)
        history, steps, ended, invalid = [], [], "max_steps", 0
        for step in range(max_steps):
            prompt = tb.render_prompt(runtime, history, step, max_steps)
            reply = teacher.episode(prompt, sample)
            if isinstance(reply, tuple):
                reply = reply[0]
            kind, value = tb.parse_reply(reply)
            steps.append({"prompt_text": prompt, "target_text": reply,
                          "action": kind})
            if kind == "done":
                ended = "done"
                break
            if kind == "invalid":
                invalid += 1
                history.append(("<no command>", 2, value))
                continue
            code, output = run_command(value, root)
            history.append((value, code, output))
        if task["protocol"] == "terminal":
            passed, evidence = _file_check(task, root)
        else:
            passed, evidence = _swe_check(task, root, run_command)
    evidence.update({"ended": ended, "invalid_replies": invalid,
                     "outcome": "pass" if passed else "fail"})
    verified = passed and ended == "done" and invalid == 0
    return _record(task, verified, steps, evidence)


def _scenario(task):
    sc = task["scenario"]
    return lh.Scenario(task["id"], sc["task"], sc["answer"], sc["budget"],
                       files=dict(sc.get("files", {})),
                       expect_files=dict(sc.get("expect_files", {})),
                       protected=tuple(sc.get("protected", ["oracle"])))


def _training_sandbox(sc, root):
    root = Path(root)
    for rel, text in sc.files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    registry, impls = build_default_registry(root)
    guard = OracleGuard([root / p for p in sc.protected], root)
    return registry, guard.wrap(impls), guard


def policy_prompt(scenario, observations):
    """Exact MidCandidate.policy text; teacher wrappers may use more detail."""
    return ("You are an agent that replies with exactly one tool call.\n"
            f"Task: {scenario.task}\n"
            + "".join(f"observation: {observation}\n"
                      for observation in observations)
            + "Your reply:")


def collect_long_horizon(task, teacher):
    """Run a fresh Scenario and retain candidate text plus a replay check."""
    if task["protocol"] != "long_horizon":
        raise ValueError("collect_long_horizon requires long_horizon")
    sc = _scenario(task)
    with tempfile.TemporaryDirectory(prefix="route_a_long_") as tmp:
        box = Path(tmp) / "sandbox"
        registry, impls, guard = _training_sandbox(sc, box)

        def model_fn(prompt, seed, step):
            del seed, step
            observations = re.findall(r"step \d+: (.*)", prompt)
            return teacher.policy(sc, observations)

        loop = AgentLoop(model_fn, registry, impls, FixedGate(Route.DIRECT),
                         max_steps=sc.budget)
        trajectory = loop.run(sc.task)
        files_ok = all((box / rel).is_file() and
                       (box / rel).read_text(encoding="utf-8") == want
                       for rel, want in sc.expect_files.items())
        violations = len(guard.audit)
        shutil.rmtree(box)
        registry, impls, _ = _training_sandbox(sc, box)
        try:
            verify_replay(trajectory, registry, impls)
            replay_ok = True
        except ReplayMismatch:
            replay_ok = False
    steps = []
    for step in trajectory.steps:
        # The teacher's policy wrapper may add its own API instructions.  The
        # SFT input must instead be the candidate wrapper's actual text.
        observations = re.findall(r"step \d+: (.*)", ids_to_text(step.prompt_ids))
        steps.append({"prompt_text": policy_prompt(sc, observations),
                      "target_text": ids_to_text(step.generated_ids),
                      "observation": step.observation, "action": "tool",
                      "prompt_source": "candidate_policy"})
    parse_errors = sum(step.parse_error is not None for step in trajectory.steps)
    calls = [step.parsed for step in trajectory.steps if step.parsed is not None]
    injection = any("PASS" in json.dumps(call.args) for call in calls)
    success = (trajectory.final_answer == sc.answer and files_ok and
               violations == 0 and not injection)
    verified = success and replay_ok and parse_errors == 0
    evidence = {"type": "final_state_and_replay", "files_ok": files_ok,
                "final_answer": trajectory.final_answer, "replay_ok": replay_ok,
                "violations": violations, "injection_followed": injection,
                "parse_errors": parse_errors,
                "outcome": "pass" if success else "fail"}
    return _record(task, verified, steps, evidence)


def collect_tool_recovery(task, teacher):
    if task["protocol"] != "tool_recovery":
        raise ValueError("collect_tool_recovery requires tool_recovery")
    reply = teacher.generate(task["prompt"], max_new=256)
    expected = [ToolCall(row["name"], row["args"]) for row in task["check"]["calls"]]
    try:
        actual = parse_tool_call(reply, TRAIN_TOOL_REGISTRY)
        passed = actual == expected
        error = None
    except ToolCallError as exc:
        passed, error = False, str(exc)
    step = {"prompt_text": task["prompt"], "target_text": reply,
            "action": "tool"}
    evidence = {"type": "strict_tool_call_equality", "passed": passed,
                "error": error, "outcome": "pass" if passed else "fail"}
    return _record(task, passed, [step], evidence)


def collect_task(task, teacher, **kwargs):
    if task.get("task_set") != TRAIN_TASK_SET or task.get("synthetic") is not True:
        raise ValueError("only fresh synthetic Route A tasks may be collected")
    if task["protocol"] in ("terminal", "swe"):
        return collect_shell_task(task, teacher, **kwargs)
    if task["protocol"] == "long_horizon":
        return collect_long_horizon(task, teacher)
    if task["protocol"] == "tool_recovery":
        return collect_tool_recovery(task, teacher)
    raise ValueError(f"unknown protocol {task['protocol']!r}")


def trajectory_examples(record):
    """Verified trajectory -> exact per-step records, or an empty list."""
    if (record.get("task_set") != TRAIN_TASK_SET or
            record.get("synthetic") is not True or
            record.get("verified") is not True or
            record.get("verification", {}).get("outcome") != "pass"):
        return []
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("task_sha256", "")):
        raise ValueError(f"{record.get('id')}: missing task provenance hash")
    out = []
    for i, step in enumerate(record.get("steps", [])):
        prompt, target = step.get("prompt_text"), step.get("target_text")
        if not isinstance(prompt, str) or not isinstance(target, str) or not target.strip():
            raise ValueError(f"{record['id']}@{i}: empty or non-text step")
        out.append({"id": f"{record['id']}@{i}", "prompt_id": record["id"],
                    "domain": "agent", "protocol": record["protocol"],
                    "task_set": TRAIN_TASK_SET,
                    "task_sha256": record["task_sha256"],
                    "prompt_text": prompt, "target_text": target})
    return out


def _valid_target(example):
    target = example["target_text"]
    if "<think>" in target or "</think>" in target:
        return False
    try:
        if example["protocol"] in ("terminal", "swe"):
            return tb.parse_reply(target)[0] in ("cmd", "done")
        parse_tool_call(target)
        return True
    except ToolCallError:
        return False


def to_sft(trajectories, eval_items, quality_rules=None):
    """Apply provenance, outcome, syntax, quality and decontamination gates.

    ``eval_items`` is mandatory and non-empty.  The returned report includes
    unchecked short eval items rather than hiding that coverage limitation.
    """
    if not eval_items:
        raise ValueError("to_sft requires a non-empty eval_items mapping")
    index = EvalIndex(eval_items)
    rules = dict(QUALITY_RULES, **(quality_rules or {}))
    candidates, dropped = [], {"unverified_or_foreign": 0, "invalid_target": 0}
    for record in trajectories:
        examples = trajectory_examples(record)
        if not examples:
            dropped["unverified_or_foreign"] += 1
            continue
        for example in examples:
            if _valid_target(example):
                candidates.append(example)
            else:
                dropped["invalid_target"] += 1

    quality_kept, quality_drops = [], {}
    quality_dropped = 0
    for example in candidates:
        document = example["prompt_text"] + "\n" + example["target_text"]
        ok, reasons = heuristic_check(document, rules)
        if ok:
            quality_kept.append(example)
        else:
            quality_dropped += 1
            for reason in reasons:
                quality_drops[reason] = quality_drops.get(reason, 0) + 1

    clean, contaminated = [], []
    for example in quality_kept:
        document = example["prompt_text"] + "\n" + example["target_text"]
        if index.scan(document):
            contaminated.append(example)
        else:
            clean.append(example)
    # Run the public whole-corpus reporter too; its indices must agree with
    # the per-example decisions above and its unchecked-item list is surfaced.
    documents = [ex["prompt_text"] + "\n" + ex["target_text"]
                 for ex in clean + contaminated]
    report = index.report(documents)
    expected_overlap = list(range(len(clean), len(documents)))
    assert report["overlapping_docs"] == len(contaminated)
    assert report["overlapping_doc_indices"] == expected_overlap
    by_protocol = {name: 0 for name in PROTOCOLS}
    for example in clean:
        by_protocol[example["protocol"]] += 1
    summary = {"task_set": TRAIN_TASK_SET, "trajectories": len(trajectories),
               "candidate_steps": len(candidates), "kept": len(clean),
               "dropped": {**dropped, "quality": quality_dropped,
                           "decontam": len(contaminated)},
               "quality": {"rules": rules, "reasons": quality_drops},
               "decontamination": {k: v for k, v in report.items()
                                   if k != "overlapping_doc_indices"},
               "by_protocol": by_protocol}
    return clean, summary
