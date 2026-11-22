"""docs/cost_per_verified_decision.md probe: measured cost front.

Loads the committed completion evidence
(checkpoints/echo_completion_evidence_2026-11.json) through the same
loud schema validation the tests use and prints the measured
latency-vs-cost Pareto front as JSON. Prints measured numbers only;
asserts nothing about the values.
"""
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helioslm_v5.eval.cost_per_task import (  # noqa: E402
    cost_front, load_completion_records)

EVIDENCE = _REPO_ROOT / "checkpoints" / "echo_completion_evidence_2026-11.json"


def main():
    records = load_completion_records(EVIDENCE)
    out = cost_front(records)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
