"""fetch_std_bench.py — pinned download of the standard-benchmark data.

Downloads the full GSM8K test split and HumanEval from their upstream
GitHub repos AT PINNED COMMITS, verifies sha256, and writes:

- data/std_bench/full/            full files (gitignored, re-fetchable)
- data/std_bench/gsm8k_test_first50.jsonl     committed CI subset
- data/std_bench/humaneval_first20.jsonl      committed CI subset
- data/std_bench/MANIFEST.json    upstream commits + every sha256

Subsets are the FIRST n records in upstream order — no selection, so
nobody can cherry-pick an easy slice. Both datasets are MIT-licensed;
their notices are in data/std_bench/LICENSES.md.

Run from repo root:
    python scripts/fetch_std_bench.py            # fetch + verify + write
    python scripts/fetch_std_bench.py --verify   # re-check committed files
"""
import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "data" / "std_bench"

SOURCES = {
    "gsm8k": {
        "repo": "openai/grade-school-math",
        "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
        "path": "grade_school_math/data/test.jsonl",
        "gzip": False,
        "full_name": "gsm8k_test.jsonl",
        "subset_name": "gsm8k_test_first50.jsonl",
        "subset_n": 50,
    },
    "humaneval": {
        "repo": "openai/human-eval",
        "commit": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "path": "data/HumanEval.jsonl.gz",
        "gzip": True,
        "full_name": "HumanEval.jsonl",
        "subset_name": "humaneval_first20.jsonl",
        "subset_n": 20,
    },
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(src) -> bytes:
    url = (f"https://raw.githubusercontent.com/{src['repo']}/"
           f"{src['commit']}/{src['path']}")
    raw = urllib.request.urlopen(url, timeout=120).read()
    return gzip.decompress(raw) if src["gzip"] else raw


def first_n_lines(data: bytes, n: int) -> bytes:
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    if len(lines) < n:
        sys.exit(f"upstream file has only {len(lines)} records (< {n})")
    for ln in lines[:n]:
        json.loads(ln)   # every committed record must be valid JSON
    return ("\n".join(lines[:n]) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check committed subsets against MANIFEST.json")
    args = ap.parse_args()
    manifest_path = OUT / "MANIFEST.json"

    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest["datasets"].items():
            got = sha256_bytes((OUT / entry["subset_file"]).read_bytes())
            if got != entry["subset_sha256"]:
                sys.exit(f"{name}: subset sha256 mismatch")
        print("std_bench subsets match MANIFEST.json")
        return

    (OUT / "full").mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "std-bench-manifest/1", "datasets": {}}
    for name, src in SOURCES.items():
        data = fetch(src)
        (OUT / "full" / src["full_name"]).write_bytes(data)
        subset = first_n_lines(data, src["subset_n"])
        (OUT / src["subset_name"]).write_bytes(subset)
        manifest["datasets"][name] = {
            "upstream": f"https://github.com/{src['repo']}",
            "commit": src["commit"],
            "path": src["path"],
            "license": "MIT",
            "full_sha256": sha256_bytes(data),
            "full_records": sum(1 for ln in data.splitlines() if ln.strip()),
            "subset_file": src["subset_name"],
            "subset_rule": f"first {src['subset_n']} records in upstream order",
            "subset_sha256": sha256_bytes(subset),
        }
        print(f"{name}: {manifest['datasets'][name]['full_records']} records, "
              f"subset {src['subset_n']} -> {src['subset_name']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()
