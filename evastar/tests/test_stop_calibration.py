"""T98 — roadmap Phase AC, AC5 (知止): stop-decision metrics.

Pins:

1. hand-built rows with known answers: false_done, the confident subset,
   ran_out, tested-before-done share and the accuracy split;
2. the test-command detector (python3 tests/x.py, pytest, -m pytest;
   not 'cat test_x.py' or 'echo pytest-ish');
3. the defining case on the real SWE tasks: a trap patch whose visible
   tests pass is still a FALSE DONE when the agent claims it, while the
   careful agent (fix, run tests, DONE) has none;
4. rows without the episode fields are refused, not guessed.

Run from repo root: python3 evastar/tests/test_stop_calibration.py
"""
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval import stop_calibration as sc  # noqa: E402
from evastar.eval import swe_bench as sb  # noqa: E402


def _row(correct, ended="done", conf=0.9, commands=()):
    return {"correct": correct, "ended": ended, "conf": conf,
            "commands": list(commands)}


def test_known_rows():
    rows = [_row(True, commands=["python3 tests/t.py"]),
            _row(False, conf=0.95, commands=["sed -i x f", "python3 tests/t.py"]),
            _row(False, conf=0.3),
            _row(True, conf=0.6),
            _row(False, ended="max_steps", conf=None)]
    r = sc.stop_report(rows)
    assert (r["n"], r["done"], r["ran_out"]) == (5, 4, 1)
    assert (r["false_done"], r["false_done_confident"]) == (2, 1)
    assert r["false_done_rate"] == 0.5
    assert r["tested_before_done"] == 0.5
    assert r["accuracy_if_tested"] == 0.5 and r["accuracy_if_untested"] == 0.5
    assert r["n_conf"] == 4 and r["accuracy"] == 0.4
    # conf order: wrong 0.95 > right 0.9 > right 0.6 > wrong 0.3 -> 2/4 wins
    assert r["auroc"] == 0.5
    print("PASS test_known_rows")


def test_test_command_detector():
    yes = ["python3 tests/test_a.py", "cd x && python test_b.py",
           "pytest -q", "python3 -m pytest tests"]
    no = ["cat tests/test_a.py", "echo pytest-ish", "ls tests", "vim test.py"]
    assert all(sc.ran_tests([c]) for c in yes), yes
    assert not any(sc.ran_tests([c]) for c in no), no
    print("PASS test_test_command_detector")


def _agent(raw, kind):
    def gen(prompt, i):
        m = re.search(r"\[(swe/[^,]+), step (\d+)", prompt)
        t, k = raw[m.group(1)], int(m.group(2)) - 1
        patch = sb.patch_command(t["fix" if kind == "careful" else "trap"])
        steps = [patch, "python3 " + t["visible"][0]]
        return (f"```bash\n{steps[k]}\n```" if k < len(steps)
                else "DONE confidence=0.9")
    return gen


def test_trap_is_false_done():
    raw = {t["id"]: t for t in sb.TASKS}
    tasks = sb.load_tasks()[:5]
    careful = sc.stop_report(sb.run_swe_bench(tasks, _agent(raw, "careful")))
    trap = sc.stop_report(sb.run_swe_bench(tasks, _agent(raw, "trap")))
    assert careful["false_done"] == 0 and careful["tested_before_done"] == 1.0
    assert trap["false_done"] == 5 and trap["false_done_confident"] == 5
    assert trap["tested_before_done"] == 1.0 and trap["accuracy_if_tested"] == 0.0
    print("PASS test_trap_is_false_done")


def test_refuses_old_rows():
    try:
        sc.stop_report([{"correct": True, "conf": 0.5}])
    except ValueError as e:
        assert "commands" in str(e)
    else:
        raise AssertionError("old rows must be refused")
    print("PASS test_refuses_old_rows")


if __name__ == "__main__":
    test_known_rows()
    test_test_command_detector()
    test_trap_is_false_done()
    test_refuses_old_rows()
