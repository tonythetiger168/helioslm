"""eval_handover_resume.py — v5.54 / M4 handover-resume oracle.

Runs the constructive interruption/resume suite and writes a versioned
benchmark artifact. Faithful fresh-session resumes must:

- validate goal/state/trajectory hashes before acting;
- execute exactly the remaining oracle plan;
- repeat zero completed prefix calls;
- finish with the oracle answer;
- reject stale-state, goal-tampered, and truncated handoffs loudly.

Run from repo root:
    python examples/eval_handover_resume.py
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.handover_resume import build_report  # noqa: E402

DEFAULT_OUT = "benchmarks/handover_resume_2026-10-14.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    report = build_report()
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = report["results"]["summary"]
    print(json.dumps({
        "faithful":
            f"{summary['faithful_resumes']}/{summary['handover_tasks']}",
        "correct_finals":
            f"{summary['correct_finals']}/{summary['handover_tasks']}",
        "prefix_repeated_steps": summary["prefix_repeated_steps"],
        "goal_drift_events": summary["goal_drift_events"],
        "tamper_controls": (
            f"{summary['tamper_controls_caught']}/"
            f"{summary['tamper_controls']}"),
        "results_sha256": report["results_sha256"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
