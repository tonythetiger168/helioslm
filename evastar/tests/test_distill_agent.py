"""D4 (v5.114): agent-trajectory distillation through hsh
(src/training/distill_agent.py).

Pins:
1. make_tasks is deterministic, covers the five kinds, and every check
   fails on the untouched workspace and passes on the reference solution;
   fix_bug also fails when the test file is edited;
2. run_task with a scripted teacher that uses the minimal preset's bash
   tool: the trajectory is verified, its log is kept, the reasoning is
   logged but never reaches the student examples;
3. trajectory_examples are exactly the student protocol: a TextModelAdapter
   that answers each prompt_text with its target_text replays the task
   through hsh and passes the same check;
4. a wrong answer or a run past max_steps is not verified;
5. run_tasks resumes by id, summarises per kind and refuses a teacher
   whose licence is not pinned; the CLI runs tasks -> run -> to-sft and
   D3's format_example passes the step records through.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.hsh import ScriptedAdapter, TextModelAdapter  # noqa: E402
from evastar.src.training import distill_agent as da  # noqa: E402
from evastar.src.training.distill_sft import format_example  # noqa: E402

FIXED = {"average": "def average(xs):\n    return sum(xs) / len(xs)\n",
         "largest": "def largest(xs):\n    return max(xs)\n",
         "evens": "def evens(xs):\n    return [x for x in xs if x % 2 == 0]\n",
         "reverse": "def reverse(s):\n    return s[::-1]\n"}


def heredoc(path, text):
    return f"mkdir -p \"$(dirname {path})\" && cat > {path} <<'EOF_X'\n{text}EOF_X"


def solution(task):
    """(bash command that solves it, final reply)."""
    k, c = task["kind"], task["check"]
    if k == "csv_sum":
        return f"printf '%s' {c['value']} > answer.txt", "已把總和寫進 answer.txt。"
    if k == "fix_bug":
        fn = task["prompt"].split(" 裡的 ")[1].split(" ")[0]
        return heredoc("lib.py", FIXED[fn]), "已修正 lib.py，測試通過。"
    if k == "term_replace":
        return heredoc("notes.md", c["value"]), "已改成台灣用語。"
    if k == "find_file":
        code = task["prompt"].split("代號 ")[1].split("？")[0]
        return f"grep -rl {code} docs", f"是 docs/{c['value']}。"
    if k == "json_filter":
        return heredoc("adults.json", json.dumps(c["value"], ensure_ascii=False) + "\n"), \
            "已建立 adults.json。"
    raise AssertionError(k)


def call(cmd, i):
    return {"reasoning": "先看看檔案。", "content": "",
            "tool_calls": [{"id": f"c{i}", "name": "bash",
                            "arguments": json.dumps({"description": "run", "command": cmd},
                                                    ensure_ascii=False)}]}


def teacher(tasks, wrong=False, loop=False):
    by_prompt = {t["prompt"]: t for t in tasks}

    def fn(request, i):
        user = next(m["content"] for m in request["messages"] if m["role"] == "user")
        task = by_prompt[user]
        done = sum(m["role"] == "assistant" for m in request["messages"])
        cmd, final = solution(task)
        if wrong:
            cmd = "echo 0 > answer.txt"
        plan = [call("ls -R", 0), call(cmd, 1), call("ls", 2),
                {"reasoning": "完成了。", "content": final}]
        if loop:
            return call("ls", done)
        return plan[min(done, 3)]
    return ScriptedAdapter(fn=fn)


CFG = da.agent_config("deepseek-ai/DeepSeek-V4.1-Flash")


def test_tasks_and_checks(tmp_path):
    a, b = da.make_tasks(10, seed=3), da.make_tasks(10, seed=3)
    assert a == b and {t["kind"] for t in a} == {"csv_sum", "fix_bug", "term_replace",
                                                  "find_file", "json_filter"}
    assert len({t["id"] for t in a}) == 10
    import subprocess
    for t in a:
        d = tmp_path / t["id"]
        da.setup_workspace(t, d)
        assert not da.check_task(t, d, "")
        cmd, final = solution(t)
        if t["kind"] != "find_file":
            subprocess.run(["bash", "-c", cmd], cwd=d, check=True)
        assert da.check_task(t, d, final), t["id"]
        if t["kind"] == "fix_bug":
            (d / "test_lib.py").write_text("print('ok')\n")
            assert not da.check_task(t, d, final)
        if t["kind"] == "find_file":
            assert not da.check_task(t, d, final + " " + t["check"]["others"][0])


def test_run_task_and_student_replay():
    tasks = da.make_tasks(5, seed=1)
    ad = teacher(tasks)
    for t in tasks:
        rec = da.run_task(t, ad, CFG)
        assert rec["verified"] and rec["reason"] == "completed" and rec["steps"] == 4, t["id"]
        assert any(e["type"] == "tool/result" for e in rec["events"])
        ex = da.trajectory_examples(rec)
        assert len(ex) == 4
        assert ex[0]["prompt_text"].startswith("##tools##")
        assert ex[0]["prompt_text"].endswith("##assistant##")
        assert ex[1]["target_text"].startswith("@@tool@@") and ex[1]["target_text"].endswith("@@end@@")
        assert ex[3]["target_text"] == solution(t)[1]
        assert all("先看看檔案" not in e["prompt_text"] + e["target_text"] for e in ex)
        assert format_example(ex[2]) == (ex[2]["prompt_text"], ex[2]["target_text"])

        table = {e["prompt_text"]: e["target_text"] for e in ex}
        student = TextModelAdapter(lambda prompt: table[prompt])
        replay = da.run_task(t, student, CFG)
        assert replay["verified"], t["id"]


def test_failures_not_verified():
    t = da.make_tasks(1, seed=2)[0]
    assert t["kind"] == "csv_sum"
    assert not da.run_task(t, teacher([t], wrong=True), CFG)["verified"]
    rec = da.run_task(t, teacher([t], loop=True), CFG, max_steps=3)
    assert rec["reason"] == "max-steps" and not rec["verified"]


def test_run_tasks_resume_and_cli(tmp_path, monkeypatch):
    tasks = da.make_tasks(5, seed=4)
    out = str(tmp_path / "traj.jsonl")
    s = da.run_tasks(tasks[:3], out, adapter=teacher(tasks), workers=1)
    assert s["n"] == 3 and s["verified"] == 3
    s = da.run_tasks(tasks, out, adapter=teacher(tasks), workers=1)
    assert s["n"] == 5 and s["verified_rate"] == 1.0
    assert s["by_kind"]["csv_sum"]["mean_steps"] == 4.0
    with pytest.raises(ValueError):
        da.run_tasks(tasks, out, teacher="gpt-x", adapter=teacher(tasks))

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import distill_agent as cli
    tf = tmp_path / "tasks.jsonl"
    cli.main(["tasks", "--n", "5", "--seed", "4", "--out", str(tf)])
    assert [json.loads(x)["id"] for x in tf.read_text().splitlines()] == [t["id"] for t in tasks]
    sft = tmp_path / "sft.jsonl"
    cli.main(["to-sft", "--trajectories", out, "--out", str(sft)])
    rows = [json.loads(x) for x in sft.read_text().splitlines()]
    assert len(rows) == 20 and {r["domain"] for r in rows} == {"agent"}


def test_bash_tool_never_runs_stale_bytecode(tmp_path):
    """Edit-and-rerun within one second at the same file size: without
    PYTHONDONTWRITEBYTECODE the second run imports the first run's .pyc."""
    t = {"id": "pyc", "kind": "fix_bug", "files": {}, "prompt": "p",
         "check": {"type": "file_equals", "path": "out.txt", "value": "0"}}
    run = "python3 -c 'import lib; print(lib.f())' > out.txt"
    steps = [call(heredoc("lib.py", "def f():\n    return 1 % 2\n"), 0), call(run, 1),
             call(heredoc("lib.py", "def f():\n    return 0 % 2\n"), 2), call(run, 3),
             {"content": "done"}]
    rec = da.run_task(t, ScriptedAdapter(steps), CFG)
    assert rec["verified"]
