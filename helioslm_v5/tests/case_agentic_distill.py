"""Route A fresh-task trajectory distillation oracles (daily gate).

No network, model weights, teacher endpoint or benchmark-generated code is
executed.  Shell execution is replaced by a deterministic fixture; the real
verifier orchestration, final-state checks, replay and export gates run.
"""
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

from helioslm_v5.agent.schema import ToolCall, render_tool_call
from helioslm_v5.agent.tool_recovery import recover_tool_call
from helioslm_v5.src.training import agentic_distill as ad
from helioslm_v5.tests.case_common import _pass

_REPO = Path(__file__).resolve().parent.parent.parent


def _c(name, **args):
    return render_tool_call([ToolCall(name, args)])


class ShellTeacher:
    def __init__(self, done_immediately=False):
        self.done_immediately = done_immediately

    def episode(self, prompt, sample):
        del sample
        if self.done_immediately or "Commands so far:" in prompt:
            return "DONE confidence=0.99"
        return "```bash\nsolve\n```"


class LongTeacher:
    def policy(self, scenario, observations):
        n = len(observations)
        product = scenario.expect_files["product.txt"]
        offset = scenario.expect_files["offset.txt"]
        total = scenario.expect_files["total.txt"]
        plan = [
            _c("calc", expr=f"{product} // 1"),
            _c("file_write", path="product.txt", content=product),
            _c("calc", expr=f"{offset} // 1"),
            _c("file_write", path="offset.txt", content=offset),
            _c("file_read", path="product.txt"),
            _c("file_read", path="offset.txt"),
            _c("calc", expr=f"{product} + {offset}"),
            _c("file_write", path="total.txt", content=total),
            _c("file_read", path="total.txt"),
            _c("finish", answer=observations[-1] if n == 9 else total),
        ]
        return plan[n]


class RecoveryTeacher:
    def __init__(self, tasks, wrong=False):
        self.tasks = {task["prompt"]: task for task in tasks}
        self.wrong = wrong

    def generate(self, prompt, max_new=256):
        del max_new
        if self.wrong:
            return "not a tool call"
        task = self.tasks[prompt]
        calls = [ToolCall(row["name"], row["args"])
                 for row in task["check"]["calls"]]
        return render_tool_call(calls)


def _shell_runner(task):
    def run(command, root):
        root = Path(root)
        if command == "solve":
            if task["protocol"] == "terminal":
                (root / task["check"]["path"]).write_text(
                    task["check"]["value"], encoding="utf-8")
            else:
                (root / "metric.py").write_text(
                    "def moving_mean(values, width):\n"
                    "    if width <= 0:\n"
                    "        raise ValueError('width must be positive')\n"
                    "    return [sum(values[j:j + width]) / width\n"
                    "            for j in range(len(values) - width + 1)]\n",
                    encoding="utf-8")
            return 0, "fixture applied"
        if command.startswith("python3 "):
            source = (root / "metric.py").read_text(encoding="utf-8")
            good = "/ width" in source and "len(values) - width + 1" in source
            return (0, "visible + hidden fixture pass") if good else (1, "repair missing")
        raise AssertionError(f"unexpected fixture command: {command}")
    return run


def _verified_records():
    tasks = {task["protocol"]: task for task in ad.make_tasks(1, seed=17)}
    terminal = ad.collect_shell_task(tasks["terminal"], ShellTeacher(),
                                     command_runner=_shell_runner(tasks["terminal"]))
    swe = ad.collect_shell_task(tasks["swe"], ShellTeacher(),
                                command_runner=_shell_runner(tasks["swe"]))
    long_horizon = ad.collect_long_horizon(tasks["long_horizon"], LongTeacher())
    recovery = ad.collect_tool_recovery(
        tasks["tool_recovery"], RecoveryTeacher([tasks["tool_recovery"]]))
    return [terminal, swe, long_horizon, recovery]


def test_fresh_tasks_and_contamination_guard():
    a = ad.make_tasks(3, seed=5)
    assert a == ad.make_tasks(3, seed=5)
    assert len(a) == 12 and len({task["id"] for task in a}) == 12
    assert {task["protocol"] for task in a} == set(ad.PROTOCOLS)
    assert all(task["task_set"] == ad.TRAIN_TASK_SET and task["synthetic"]
               for task in a)
    eval_items = ad.builtin_eval_items()
    assert len(eval_items) == 145       # 50 terminal + 20 SWE + 5 LH + 70 recovery
    report = ad.assert_tasks_disjoint(a, eval_items)
    assert report["overlapping_docs"] == 0
    polluted = dict(eval_items)
    polluted["explicit-copy"] = ad.task_text(a[0])
    try:
        ad.assert_tasks_disjoint(a, polluted)
    except ValueError as exc:
        assert a[0]["id"] in str(exc) and "explicit-copy" in str(exc)
    else:
        raise AssertionError("an exact training/eval copy passed the guard")
    print("PASS test_fresh_tasks_and_contamination_guard")


def test_shell_collection_and_final_state_verifier():
    tasks = {task["protocol"]: task for task in ad.make_tasks(1, seed=9)}
    for protocol in ("terminal", "swe"):
        task = tasks[protocol]
        rec = ad.collect_shell_task(task, ShellTeacher(),
                                    command_runner=_shell_runner(task))
        assert rec["verified"] and rec["verification"]["ended"] == "done"
        assert len(rec["steps"]) == 2
        assert rec["steps"][0]["target_text"] == "```bash\nsolve\n```"
        assert rec["steps"][1]["target_text"].startswith("DONE")
        bad = ad.collect_shell_task(task, ShellTeacher(done_immediately=True),
                                    command_runner=_shell_runner(task))
        assert not bad["verified"] and bad["verification"]["outcome"] == "fail"
    print("PASS test_shell_collection_and_final_state_verifier")


def test_long_horizon_text_capture_and_replay():
    task = next(t for t in ad.make_tasks(1, seed=11)
                if t["protocol"] == "long_horizon")
    rec = ad.collect_long_horizon(task, LongTeacher())
    assert rec["verified"] and rec["verification"]["replay_ok"]
    assert rec["verification"]["files_ok"] and not rec["verification"]["violations"]
    assert len(rec["steps"]) == 10
    examples = ad.trajectory_examples(rec)
    assert len(examples) == 10
    assert examples[0]["prompt_text"] == ad.policy_prompt(
        ad._scenario(task), [])
    assert examples[0]["prompt_text"].startswith(
        "You are an agent that replies with exactly one tool call.")
    assert examples[-1]["target_text"].endswith("@@end@@")
    assert "oracle/answer.txt" not in examples[-1]["target_text"]

    class FalseClaim:
        def policy(self, scenario, observations):
            del observations
            return _c("finish", answer=scenario.answer)

    false = ad.collect_long_horizon(task, FalseClaim())
    assert not false["verified"] and not false["verification"]["files_ok"]
    print("PASS test_long_horizon_text_capture_and_replay")


def test_tool_recovery_strict_equality():
    tasks = [t for t in ad.make_tasks(4, seed=13) if t["protocol"] == "tool_recovery"]
    for task in tasks:
        # The fixture is genuinely malformed but independently repairable.
        calls, repairs = recover_tool_call(task["prompt"].split("\n\n", 1)[1],
                                           ad.TRAIN_TOOL_REGISTRY)
        expected = [ToolCall(row["name"], row["args"])
                    for row in task["check"]["calls"]]
        assert calls == expected and repairs
        rec = ad.collect_tool_recovery(task, RecoveryTeacher([task]))
        assert rec["verified"] and rec["verification"]["passed"]
        bad = ad.collect_tool_recovery(task, RecoveryTeacher([task], wrong=True))
        assert not bad["verified"] and bad["verification"]["error"]
    print("PASS test_tool_recovery_strict_equality")


def test_export_gates_and_cli(tmp_path=None):
    records = _verified_records()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        eval_items = ad.combined_eval_items()
        rows, report = ad.to_sft(records, eval_items)
        assert len(rows) == sum(len(r["steps"]) for r in records)
        assert all(report["by_protocol"][p] > 0 for p in ad.PROTOCOLS)
        assert report["decontamination"]["overlapping_docs"] == 0
        assert all("task_sha256" in row for row in rows)

        foreign = copy.deepcopy(records[-1])
        foreign["task_set"] = "formal-eval-reference"
        unverified = copy.deepcopy(records[-1])
        unverified["verified"] = False
        invalid = copy.deepcopy(records[-1])
        invalid["steps"][0]["target_text"] = "@@tool@@broken@@end@@"
        bad_rows, bad_report = ad.to_sft([foreign, unverified, invalid], eval_items)
        assert bad_rows == []
        assert bad_report["dropped"]["unverified_or_foreign"] == 2
        assert bad_report["dropped"]["invalid_target"] == 1

        contaminated = dict(eval_items)
        contaminated["test-copy"] = records[-1]["steps"][0]["prompt_text"]
        kept, rep = ad.to_sft([records[-1]], contaminated)
        assert kept == [] and rep["dropped"]["decontam"] == 1
        quality_rows, quality_report = ad.to_sft(
            [records[-1]], eval_items, quality_rules={"min_units": 100_000})
        assert quality_rows == []
        assert quality_report["dropped"]["quality"] == 1
        assert quality_report["quality"]["reasons"]["too_short"] == 1
        try:
            ad.to_sft(records, {})
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("empty eval index bypassed decontamination")

        sys.path.insert(0, str(_REPO / "examples"))
        import agentic_distill as cli
        task_path = root / "tasks.jsonl"
        cli.main(["tasks", "--per-protocol", "2", "--seed", "3",
                  "--out", str(task_path)])
        loaded = [json.loads(line) for line in
                  task_path.read_text(encoding="utf-8").splitlines()]
        assert len(loaded) == 8
        trajectories = root / "trajectories.jsonl"
        trajectories.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                        for r in records), encoding="utf-8")
        output = root / "sft.jsonl"
        cli.main(["export", "--trajectories", str(trajectories),
                  "--out", str(output)])
        saved = [json.loads(line) for line in
                 output.read_text(encoding="utf-8").splitlines()]
        assert len(saved) == len(rows)
        metadata = json.loads(Path(str(output) + ".report.json").read_text(
            encoding="utf-8"))
        assert metadata["kept"] == len(rows) and metadata["training_started"] is False
    print("PASS test_export_gates_and_cli")


def test_agentic_distill_suite():
    test_fresh_tasks_and_contamination_guard()
    test_shell_collection_and_final_state_verifier()
    test_long_horizon_text_capture_and_replay()
    test_tool_recovery_strict_equality()
    test_export_gates_and_cli()
    _pass("test_agentic_distill_suite",
          "Route A: fresh disjoint four-protocol tasks, verified step capture, "
          "replay/final-state oracles, quality + mandatory decontam SFT export")


if __name__ == "__main__":
    test_agentic_distill_suite()
