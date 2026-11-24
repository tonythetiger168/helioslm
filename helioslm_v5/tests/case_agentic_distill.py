"""Route A: fresh four-protocol teacher data, kept disjoint from formal eval."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from helioslm_v5.src.training import agentic_distill as ad
from helioslm_v5.src.training import tool_recovery_bench as tr
from helioslm_v5.src.training.agentic_index import VALIDATORS
from helioslm_v5.src.training.distill_sft import heuristic_check
from helioslm_v5.src.training.terminal_bench import TerminalTask
from helioslm_v5.src.training import terminal_bench as tb
from helioslm_v5.src.eval import agentic_bench as ab


def test_fresh_tasks_and_contamination_guard():
    eval_items = ad.builtin_eval_items()
    counts = {name: 0 for name in ad.PROTOCOLS}
    for item in eval_items:
        counts[item["protocol"]] += 1
    assert counts == {"terminal": 50, "swe": 20,
                      "long_horizon": 5, "tool_recovery": 70}
    for seed in (0, 41):
        tasks = ad.make_tasks(per_protocol=4, seed=seed)
        assert len(tasks) == 16
        ad.assert_tasks_disjoint(tasks, eval_items)
        assert len({t["task_sha256"] for t in tasks}) == 16
    foreign = ad.make_tasks(per_protocol=1)[0]
    foreign["text"] = eval_items[0]["text"]
    try:
        ad.assert_tasks_disjoint([foreign], eval_items)
    except ValueError as exc:
        assert "formal-eval" in str(exc)
    else:
        raise AssertionError("formal eval copy accepted as a training task")
    for task in ad.make_tasks(per_protocol=1):
        assert task["task_set"] == ad.TRAIN_TASK_SET
        assert task["provenance"] == ad.PROVENANCE


def test_shell_collection_and_final_state_verifier():
    task = next(t for t in ad.make_tasks(per_protocol=1)
                if t["protocol"] == "terminal")
    replies = iter([
        "cat > answer.txt <<'EOF'\n"
        f"{task['payload']['expected']}\nEOF\nDONE",
        "unexpected second call",
    ])

    def teacher(prompt, **kwargs):
        assert "Create answer.txt" in prompt
        return next(replies), {"cost": 0.0}

    record = ad.collect_task(task, teacher)
    assert record["outcome"] == "verified"
    assert record["steps"][0]["action"] == "done"
    assert "answer.txt" in record["steps"][0]["observation"]
    assert record["metrics"]["steps"] == 1
    examples = ad.trajectory_examples(record)
    assert examples[0]["target"].endswith("EOF\nDONE")
    sft = ad.to_sft([record], ad.builtin_eval_items(),
                    quality_rules={"min_units": 1})
    assert sft["report"]["kept"] == 1

    bad = ad.collect_task(task, lambda prompt, **kw: (
        "not a shell command", {"cost": 0.0}))
    assert bad["outcome"] == "unverified"
    assert not ad.to_sft([bad], ad.builtin_eval_items())["examples"]


def test_long_horizon_text_capture_and_replay():
    from helioslm_v5.src.training.agentic_harness import (
        AgentHarness, AgentLoop, ToolCall, parse_tool_call)
    from helioslm_v5.src.eval.long_horizon_bench import FileEnv
    import torch

    task = next(t for t in ad.make_tasks(per_protocol=1)
                if t["protocol"] == "long_horizon")
    sc = ab.LongScenario(**task["payload"])
    captured = []

    class Teacher:
        def reset(self):
            pass

        def act(self, observation):
            captured.append(observation)
            if not any(parse_tool_call(x).name == "write_file"
                       for x in captured):
                text = sc.files[sc.target_file]
            else:
                text = "answer: " + str(sc.answer)
            call = ToolCall("write_file",
                            {"path": "/output/answer.txt", "content": text})
            return ab._ids(call.render())

    env = FileEnv(sc)
    with tempfile.TemporaryDirectory() as tmp:
        harness = AgentHarness(env, ToolCall, parse_tool_call, tmp,
                               final_file="answer.txt")
        loop = AgentLoop(Teacher(), harness, max_steps=4,
                         tokenizer=ab._SimpleTokenizer())
        record = ad.collect_task(task, loop)
    assert record["outcome"] == "verified"
    assert record["metrics"]["file_correct"] is True
    assert record["metrics"]["replay_ok"] is True
    assert len(record["steps"]) == 2
    observations = []
    for step, observation in zip(record["steps"], captured):
        assert step["prompt_text"] == ad.policy_prompt(sc, observations)
        assert step["prompt_source"] == "candidate_policy"
        assert observation in step["prompt_text"]
        assert "private API" not in step["prompt_text"]
        assert step["target_text"] == parse_tool_call(observation).render()
        observations.append(observation)
    examples = ad.trajectory_examples(record)
    assert len(examples) == 2
    assert examples[-1]["target"].startswith("{\"name\": \"write_file\"")
    assert ad.to_sft([record], ad.builtin_eval_items(),
                     quality_rules={"min_units": 1})["report"]["kept"] == 2

    class FalseClaim(Teacher):
        def act(self, observation):
            call = ToolCall("write_file",
                            {"path": "/output/answer.txt",
                             "content": "answer: wrong"})
            return ab._ids(call.render())

    with tempfile.TemporaryDirectory() as tmp:
        harness = AgentHarness(FileEnv(sc), ToolCall, parse_tool_call, tmp,
                               final_file="answer.txt")
        bad = ad.collect_task(task, AgentLoop(
            FalseClaim(), harness, max_steps=2,
            tokenizer=ab._SimpleTokenizer()))
    assert bad["outcome"] == "unverified"
    assert bad["metrics"]["file_correct"] is False
    assert not ad.to_sft([bad], ad.builtin_eval_items())["examples"]


def test_tool_recovery_strict_equality():
    task = next(t for t in ad.make_tasks(per_protocol=1)
                if t["protocol"] == "tool_recovery")
    calls = task["payload"]["calls"]
    rendered = tr.render_tool_call(calls, style="xml")
    record = ad.collect_task(
        task, lambda prompt, **kw: (rendered, {"cost": 0.0}))
    assert record["outcome"] == "verified"
    assert record["steps"][0]["target_text"] == rendered
    assert record["metrics"]["exact_calls"] is True
    assert len(ad.to_sft([record], ad.builtin_eval_items(),
                         quality_rules={"min_units": 1})["examples"]) == 1
    for reply in (rendered + " extra", rendered.replace("}", "}", 1) + "\n garbage"):
        bad = ad.collect_task(task, lambda prompt, _r=reply, **kw: (
            _r, {"cost": 0.0}))
        assert bad["outcome"] == "unverified"


def test_export_gates_and_cli():
    task = next(t for t in ad.make_tasks(per_protocol=1)
                if t["protocol"] == "tool_recovery")
    target = tr.render_tool_call(task["payload"]["calls"], style="xml")
    good = {
        "task": task, "provenance": ad.PROVENANCE,
        "task_sha256": task["task_sha256"], "outcome": "verified",
        "steps": [{"prompt_text": "A sufficient synthetic training prompt "
                                  "with enough distinct words for quality checks.",
                   "target_text": target, "action": "final"}],
    }
    result = ad.to_sft([good], ad.builtin_eval_items())
    assert result["report"]["kept"] == 1
    assert result["report"]["training_started"] is False
    for mutation in ("foreign", "unverified", "think", "contamination"):
        row = json.loads(json.dumps(good))
        if mutation == "foreign":
            row["provenance"] = "external-reference-eval"
        elif mutation == "unverified":
            row["outcome"] = "unverified"
        elif mutation == "think":
            row["steps"][0]["target_text"] = "<think>secret</think>"
        else:
            text = ad.builtin_eval_items()[0]["text"]
            row["steps"][0] = {"prompt_text": text, "target_text": text,
                               "action": "final"}
        rejected = ad.to_sft([row], ad.builtin_eval_items())
        assert rejected["report"]["kept"] == 0, mutation
    try:
        ad.to_sft([good], [])
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty formal eval index accepted")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tasks_path = root / "tasks.jsonl"
        tasks_path.write_text("".join(
            json.dumps(t) + "\n" for t in ad.make_tasks(per_protocol=1)),
            encoding="utf-8")
        cmd = [sys.executable, "examples/agentic_distill.py", "export",
               "--trajectories", str(root / "traj.jsonl"),
               "--out", str(root / "sft.jsonl")]
        (root / "traj.jsonl").write_text(
            json.dumps(good) + "\n", encoding="utf-8")
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        report = json.loads((root / "sft.report.json").read_text())
        assert report["kept"] == 1
        assert report["output_sha256"]
        (root / "traj.jsonl").write_text("", encoding="utf-8")
        refused = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert refused.returncode != 0
        assert "No SFT rows survived" in refused.stderr


def test_agentic_distill_suite():
    test_fresh_tasks_and_contamination_guard()
    test_shell_collection_and_final_state_verifier()
    test_long_horizon_text_capture_and_replay()
    test_tool_recovery_strict_equality()
    test_export_gates_and_cli()
