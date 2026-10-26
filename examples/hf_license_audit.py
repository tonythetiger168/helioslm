"""hf_license_audit.py — roadmap TW1: licence screen of the Traditional and
Simplified Chinese, Spanish, Japanese and Korean datasets the training plan names
(helioslm_v5/eval/hf_license_audit.py).

Needs network access to huggingface.co (environment settings: Network
access -> Allowed domains) and, for gated cards, HF_TOKEN in the
environment. Never pass a token on the command line.

Usage:
  python examples/hf_license_audit.py --output benchmarks/hf_license_audit_<date>.json
  python examples/hf_license_audit.py --ids some/dataset other/dataset
"""
import argparse
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.hf_license_audit import (  # noqa: E402
    ALL, audit, hub_fetch, summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+",
                    default=list(ALL))
    ap.add_argument("--output")
    args = ap.parse_args()
    fetch = partial(hub_fetch, token=os.environ.get("HF_TOKEN"))
    rows = audit(args.ids, fetch)
    for r in rows:
        print(f"{r['class']:15s} {str(r['license']):22s} "
              f"gated={r['gated']!s:5s} {r['id']}"
              + (f"  ERROR {r['error']}" if r["error"] else ""))
    out = {"experiment": "TW1 Hugging Face dataset licence audit",
           "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rows": rows, "summary": summary(rows),
           "rows_sha256": hashlib.sha256(
               json.dumps(rows, sort_keys=True).encode()).hexdigest(),
           "note": "card metadata only; a screening list, not legal advice"}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)
    if summary(rows)["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
