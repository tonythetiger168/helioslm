"""Knowledge rephrasing of a zh-TW corpus (v5.206,
helioslm_v5/src/training/rephrase.py). Standard library only.

    # 1. prompts: every (document, style, chunk) of a tw2_prepare_corpus.py output
    python examples/rephrase_corpus.py prompts --docs /workspace/tw2 \\
        --styles textbook qa notes encyclopedia spoken --max-docs 20000 \\
        --out runs/rephrase/prompts.jsonl
    # 2. the teacher rewrites them (thinking off); safe to re-run after a crash
    python examples/distill_generate.py generate --prompts runs/rephrase/prompts.jsonl \\
        --out runs/rephrase/answers.jsonl --no-thinking --temperature 0.8 \\
        --max-tokens 4096 --per-hour 15 --max-cost 40
    # 3. fidelity check + join into documents (tw2 record layout)
    python examples/rephrase_corpus.py assemble --docs /workspace/tw2 \\
        --answers runs/rephrase/answers.jsonl --out runs/rephrase/tw2_rephrased.jsonl

Then decontaminate and tokenise the output like any TW2 shard
(``pilot_prepare_data.py --text``) and give it its own mixture source.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training import rephrase as rp  # noqa: E402
from helioslm_v5.src.training.distill_gen import read_jsonl  # noqa: E402


def iter_docs(path):
    """Records of a .jsonl file or of every .jsonl under a directory
    (sorted, so the order is reproducible)."""
    p = Path(path)
    files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    for f in files:
        for rec in read_jsonl(str(f)):
            if rec.get("id") and rec.get("text"):
                yield rec


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="rewrite prompts for a corpus")
    p.add_argument("--docs", required=True)
    p.add_argument("--styles", nargs="+", default=list(rp.DEFAULT_STYLES),
                   choices=sorted(rp.STYLES))
    p.add_argument("--max-chars", type=int, default=1500)
    p.add_argument("--context-chars", type=int, default=300)
    p.add_argument("--min-chars", type=int, default=200,
                   help="skip shorter documents")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--out", required=True)
    s = sub.add_parser("assemble", help="check rewrites and join them")
    s.add_argument("--docs", required=True)
    s.add_argument("--answers", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-overlap", type=float, default=0.15)
    s.add_argument("--max-overlap", type=float, default=0.85)
    a = ap.parse_args(argv)

    if a.cmd == "prompts":
        def rows():
            n = 0
            for d in iter_docs(a.docs):
                if len(d["text"]) < a.min_chars:
                    continue
                if a.max_docs is not None and n >= a.max_docs:
                    return
                n += 1
                yield from rp.rephrase_prompts(d, a.styles, a.max_chars,
                                               a.context_chars)
        print(json.dumps({"prompts": write_jsonl(a.out, rows())}))
    else:
        recs = read_jsonl(a.answers)
        wanted = {(r.get("meta") or {}).get("doc_id") for r in recs}
        docs = {d["id"]: d for d in iter_docs(a.docs) if d["id"] in wanted}
        out, report = rp.assemble(recs, docs, min_overlap=a.min_overlap,
                                  max_overlap=a.max_overlap)
        write_jsonl(a.out, out)
        report["chars"] = sum(len(d["text"]) for d in out)
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
