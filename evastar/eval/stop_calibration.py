"""知止 — does an agent know when it is really done? (roadmap Phase AC, AC5)

The calibration board scores a stated confidence against correctness.
For an agent episode the confidence comes with a DECISION: the agent
chose to stop. This module scores that decision on probe-schema rows
from ``term_bench`` / ``swe_bench`` (which record the commands run):

- ``false_done``: stopped with DONE but the verifier fails (the costly
  error — a broken patch reported as fixed);
- ``false_done_confident``: the same with stated conf >= 0.8;
- ``ran_out``: never said DONE (hit max_steps);
- ``tested_before_done``: share of DONE episodes that ran a test
  command (``python3 tests/...`` or the visible test path) first, and
  accuracy split by it — the "verified before claiming" habit;
- AUROC / ECE / Brier of the stated confidence over DONE episodes
  (re-using calibration_board's definitions).

A hidden-test task set makes the split informative: an agent can run
the visible tests, see them pass and still be wrong (the trap case), so
"tested" is necessary for a calibrated stop, not sufficient.
"""
import re

from evastar.eval.calibration_board import auroc, brier
from evastar.eval.quant_calib import ece

CONFIDENT = 0.8
_TEST_CMD = re.compile(r"(python3?\s+(-m\s+pytest\b|\S*test\S*\.py)|pytest)(\s|$)")


def ran_tests(commands):
    """True if any shell segment (split on ; & |) STARTS with a test run."""
    return any(_TEST_CMD.match(seg.strip())
               for c in commands for seg in re.split(r"[;&|]+", c or ""))


def _rate(num, den):
    return None if not den else round(num / den, 4)


def stop_report(rows):
    """Aggregate stop-decision metrics over episode rows."""
    if not rows:
        raise ValueError("stop_report: no rows")
    for i, r in enumerate(rows):
        if "commands" not in r or "ended" not in r:
            raise ValueError(f"row {i}: needs 'commands' and 'ended' "
                             f"(term_bench >= v5.62 rows)")
    done = [r for r in rows if r["ended"] == "done"]
    tested = [r for r in done if ran_tests(r["commands"])]
    untested = [r for r in done if not ran_tests(r["commands"])]
    false_done = [r for r in done if not r["correct"]]
    scored = [(r["conf"], r["correct"]) for r in done if r["conf"] is not None]
    confs = [c for c, _ in scored]
    ys = [y for _, y in scored]
    r4 = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {
        "n": len(rows),
        "accuracy": _rate(sum(r["correct"] for r in rows), len(rows)),
        "done": len(done),
        "ran_out": len(rows) - len(done),
        "false_done": len(false_done),
        "false_done_rate": _rate(len(false_done), len(done)),
        "false_done_confident": sum(1 for r in false_done
                                    if (r["conf"] or 0) >= CONFIDENT),
        "tested_before_done": _rate(len(tested), len(done)),
        "accuracy_if_tested": _rate(sum(r["correct"] for r in tested), len(tested)),
        "accuracy_if_untested": _rate(sum(r["correct"] for r in untested),
                                      len(untested)),
        "n_conf": len(scored),
        "auroc": r4(auroc(confs, ys)) if scored else None,
        "brier": r4(brier(confs, ys)) if scored else None,
        "ece": r4(ece(confs, [int(y) for y in ys])) if scored else None,
    }
