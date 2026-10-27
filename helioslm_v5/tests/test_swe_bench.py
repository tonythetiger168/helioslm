"""T97 — roadmap Phase AC, AC1: mini SWE task set.

Pins:

1. every task's three reference gates: the untouched repository fails,
   the reference patch passes, and the trap patch — which passes the
   VISIBLE tests — fails the hidden ones;
2. grading guards: weakening a visible test does not help (the verifier
   restores it), the hidden test never appears in the sandbox, and
   grading never mutates the agent's sandbox;
3. a scripted multi-turn episode through term_bench's loop yields a
   ``family='swe'`` probe-schema row with verbalized confidence;
4. the runner's reference artifact carries the 4 swe gates.

Run from repo root: python3 helioslm_v5/tests/test_swe_bench.py
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import swe_bench as sb  # noqa: E402
from helioslm_v5.eval import term_bench as tb  # noqa: E402

_RAW = {t["id"]: t for t in sb.TASKS}
_TASKS = {t["id"]: t for t in sb.load_tasks()}


def _grade(tid, edits=None, extra=()):
    task, raw = _TASKS[tid], _RAW[tid]
    with tempfile.TemporaryDirectory() as root:
        tb.seed_sandbox(task, root)
        if edits:
            code, out = tb.run_command(sb.patch_command(edits), root)
            assert code == 0, (tid, out)
        for cmd in extra:
            tb.run_command(cmd, root)
        visible = all(tb.run_command(f"python3 {v}", root)[0] == 0
                      for v in raw["visible"])
        ok, _ = task["verify"](root)
    return ok, visible


def test_reference_gates_every_task():
    assert len(_TASKS) == 20
    for tid, raw in _RAW.items():
        ok, vis = _grade(tid)
        assert not ok and vis, (tid, "untouched repo must fail hidden only")
        ok, _ = _grade(tid, raw["fix"])
        assert ok, (tid, "reference patch must pass")
        ok, vis = _grade(tid, raw["trap"])
        assert not ok and vis, (tid, "trap must pass visible, fail hidden")
    print(f"PASS test_reference_gates_every_task ({len(_TASKS)} tasks)")


def test_grading_guards():
    tid = "swe/01-paginate"
    raw = _RAW[tid]
    # the agent "fixes" the bug by deleting the visible test's asserts and
    # dropping its own copy of the hidden test: still graded as a failure
    ok, _ = _grade(tid, extra=["printf 'print(1)\\n' > tests/test_pager.py",
                               f"printf 'print(1)\\n' > {sb.HIDDEN_NAME}"])
    assert not ok
    with tempfile.TemporaryDirectory() as root:
        tb.seed_sandbox(_TASKS[tid], root)
        tb.run_command(sb.patch_command(raw["fix"]), root)
        before = {p.relative_to(root): p.read_bytes()
                  for p in Path(root).rglob("*") if p.is_file()}
        assert _TASKS[tid]["verify"](root)[0]
        after = {p.relative_to(root): p.read_bytes()
                 for p in Path(root).rglob("*") if p.is_file()}
        assert before == after, "grading mutated the sandbox"
        assert not (Path(root) / sb.HIDDEN_NAME).exists()
    for t in sb.TASKS:
        assert sb.HIDDEN_NAME not in t["files"]
        assert "Hidden tests" in _TASKS[t["id"]]["instruction"]
    print("PASS test_grading_guards")


def test_episode_row():
    tid = "swe/17-flatten"
    fix = sb.patch_command(_RAW[tid]["fix"])
    replies = iter(["```bash\ncat nested.py\n```",
                    f"```bash\n{fix}\n```",
                    "```bash\npython3 tests/test_nested.py\n```",
                    "DONE confidence=0.8"])
    rows = sb.run_swe_bench([_TASKS[tid]], lambda p, i: next(replies))
    row = rows[0]
    assert row["family"] == "swe" and row["correct"] and row["steps"] == 3
    assert row["conf"] == 0.8 and row["conf_type"] == "verbalized"
    assert row["pass_at_1"] == 1.0
    print("PASS test_episode_row")


def test_reference_artifact_has_swe_gates():
    report = json.loads((_REPO / "benchmarks"
                         / "std_bench_reference_2026-10-15.json").read_text())
    swe = {k: v for k, v in report["gates"].items() if k.startswith("swe/")}
    assert set(swe) == {"swe/reference_solution", "swe/done_immediately",
                        "swe/prose_no_command", "swe/trap_patch"}
    assert all(v["accuracy"] == v["required"] for v in swe.values())
    assert "helioslm_v5/eval/swe_bench.py" in report["inputs"]
    print("PASS test_reference_artifact_has_swe_gates")


if __name__ == "__main__":
    test_reference_gates_every_task()
    test_grading_guards()
    test_episode_row()
    test_reference_artifact_has_swe_gates()
