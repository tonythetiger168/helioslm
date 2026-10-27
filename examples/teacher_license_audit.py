"""teacher_license_audit.py — licences of the five open-weight top models
considered as distillation teachers (helioslm_v5/eval/teacher_license_audit.py).

Needs network access to huggingface.co.

Usage:
  python examples/teacher_license_audit.py --output benchmarks/teacher_license_audit_2026-10-20.json
  python examples/teacher_license_audit.py --pending   # OS4 rows, pin their hashes
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.teacher_license_audit import (  # noqa: E402
    PENDING_TEACHERS, SCOPE_NOTES, TEACHERS, audit, pin_pending)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--pending", action="store_true",
                    help="audit the rows whose hash is not pinned yet and "
                         "print the REVIEWS entries to paste once the "
                         "fetched texts match the recorded readings")
    args = ap.parse_args()
    rows = audit(PENDING_TEACHERS if args.pending else TEACHERS)
    if args.pending:
        print(json.dumps(pin_pending(rows), indent=4))
    for r in rows:
        print(f"{r['verdict']:24s} {r['license']:45s} {r['id']}")
    out = {"experiment": "distillation teacher licence audit",
           "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scope_notes": list(SCOPE_NOTES),
           "rows": rows,
           "rows_sha256": hashlib.sha256(
               json.dumps(rows, sort_keys=True).encode()).hexdigest(),
           "note": "a reviewed reading of the licence text, not legal advice"}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
