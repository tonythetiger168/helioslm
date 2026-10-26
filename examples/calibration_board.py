"""calibration_board.py — roadmap Phase A: calibration leaderboard.

Scores every committed eval artifact in the shared probe schema on
"does the model know when it is wrong" (AUROC / Brier / ECE /
overconfident-wrong) and writes a ranked board.

New API rows come from ``examples/probe_api.py`` (needs OPENAI_API_KEY):
run it, copy the output into benchmarks/, register its tier in
benchmarks/verification_tiers.json, add it to ARTIFACTS below and
re-run this script.

Run from repo root:
    python examples/calibration_board.py
    python examples/calibration_board.py --check   # verify committed board
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.calibration_board import (  # noqa: E402
    build_board, render_markdown)

DATE = "2026-10-15"
# mid_agent_eval_v4_copycurr.json is left out: it is byte-identical to
# mid_agent_eval_v4.json (same sha256), so it would count one run twice.
ARTIFACTS = [
    "benchmarks/api_probe_deepseek_2026-10-04.json",
    "benchmarks/api_probe_luna_2026-10-04.json",
    "benchmarks/mid_agent_eval_v2.json",
    "benchmarks/mid_agent_eval_v4.json",
    "benchmarks/mid_agent_eval_v4_mismatch_probe.json",
    "benchmarks/mid_agent_eval_v4_trilogy.json",
    "benchmarks/mid_agent_eval_v5.json",
    "benchmarks/mid_agent_eval_v5_trilogy.json",
]
OUT_JSON = "benchmarks/calibration_board_2026-10-15.json"
OUT_MD = "benchmarks/calibration_board_2026-10-15.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild and fail if the committed board differs")
    args = ap.parse_args()
    board = build_board([_REPO / a for a in ARTIFACTS], DATE, _REPO)
    text = json.dumps(board, indent=2, ensure_ascii=False) + "\n"
    md = render_markdown(board)
    if args.check:
        stale = [p for p, want in ((OUT_JSON, text), (OUT_MD, md))
                 if (_REPO / p).read_text(encoding="utf-8") != want]
        if stale:
            sys.exit(f"stale calibration board: {', '.join(stale)}")
        print("calibration board up to date")
        return
    (_REPO / OUT_JSON).write_text(text, encoding="utf-8")
    (_REPO / OUT_MD).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
