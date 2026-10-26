"""eval_clarify_missing_param.py — v5.53 / M3 TrustGate CLARIFY oracle.

Runs the constructive missing-parameter suite and writes a versioned
benchmark artifact. The expected behavior is three-state:

- DIRECT only after the missing parameter is supplied and the wrapped gate
  trusts the call;
- CLARIFY for a syntactically valid call with a required parameter absent;
- ESCALATE for complete-but-distrusted or schema-invalid calls.

Run from repo root:
    python examples/eval_clarify_missing_param.py
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.clarify_missing_param import build_report  # noqa: E402

DEFAULT_OUT = "benchmarks/clarify_missing_param_2026-10-13.json"


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
        "clarified": f"{summary['clarified']}/{summary['missing_tasks']}",
        "direct_controls":
            f"{summary['direct_after_completion']}/{summary['complete_controls']}",
        "escalated_low_trust": summary["escalated_low_trust"],
        "escalated_invalid_schema": summary["escalated_invalid_schema"],
        "results_sha256": report["results_sha256"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
