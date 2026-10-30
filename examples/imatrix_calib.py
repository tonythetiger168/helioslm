"""imatrix_calib.py — calibration text for llama-imatrix (v5.208).

Interleaves documents from several sources (plain text, one document per
blank-line block, or JSONL) so every part of the file covers each source,
drops exact duplicates and stops at --max-chars. JSONL rows give their
text through --field (default ``text``); chat rows with ``messages`` use
the message contents.

  python examples/imatrix_calib.py --src kept_v2.jsonl wiki_tw.txt code.jsonl \\
      --out calib.txt --max-chars 600000

Then: examples/package_hf.py ... --imatrix-data calib.txt
"""
import argparse
import json
from itertools import zip_longest
from pathlib import Path


def read_docs(path, field="text", min_chars=200):
    """Documents of at least ``min_chars`` characters from ``path``."""
    path = Path(path)
    if path.suffix == ".jsonl":
        docs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get(field), str):
                docs.append(row[field])
            elif isinstance(row.get("messages"), list):
                docs.append("\n".join(str(m.get("content", "")) for m in row["messages"]
                                      if isinstance(m, dict)))
    else:
        docs = path.read_text(encoding="utf-8").split("\n\n")
    return [d.strip() for d in docs if len(d.strip()) >= min_chars]


def build(sources, max_chars, field="text", min_chars=200):
    """Round-robin over the sources' documents, deduplicated, up to ``max_chars``."""
    seen, out, n = set(), [], 0
    for group in zip_longest(*(read_docs(s, field, min_chars) for s in sources)):
        for doc in group:
            if doc is None or doc in seen:
                continue
            seen.add(doc)
            if n + len(doc) > max_chars:
                return "\n\n".join(out) + "\n"
            out.append(doc)
            n += len(doc) + 2
    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help=".txt or .jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=600_000)
    ap.add_argument("--field", default="text")
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()
    text = build(args.src, args.max_chars, args.field, args.min_chars)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"{args.out}: {len(text)} chars")


if __name__ == "__main__":
    main()
