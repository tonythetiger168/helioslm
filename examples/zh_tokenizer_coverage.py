"""zh_tokenizer_coverage.py — roadmap TW4: Traditional-Chinese tokenizer
coverage report (tokens per character, single-token coverage).

Usage:
  python examples/zh_tokenizer_coverage.py --output benchmarks/zh_tokenizer_coverage_2026-10-20.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("RAYON_NUM_THREADS", "1")   # deterministic BPE training

from helioslm_v5.eval.zh_tokenizer_coverage import (  # noqa: E402
    HELD_OUT, Bpe, CharToy, Utf8Bytes, eval_split, eval_stats, measure,
    train_bpe, training_corpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[8000, 32000])
    ap.add_argument("--output")
    args = ap.parse_args()
    t0 = time.time()
    zh, en = eval_split(_REPO)
    corpora = {"bpe_code_only": training_corpus(_REPO, strip_cjk=True),
               "bpe_repo": training_corpus(_REPO)}
    rows = [{"tokenizer": "char_toy", "vocab": 1024,
             **measure(CharToy(1024), zh, en)},
            {"tokenizer": "utf8_bytes", "vocab": 256,
             **measure(Utf8Bytes(), zh, en)}]
    for name, text in corpora.items():
        for v in args.vocab_sizes:
            tok = train_bpe(text, v)
            rows.append({"tokenizer": name, "vocab": tok.get_vocab_size(),
                         "vocab_requested": v, **measure(Bpe(tok), zh, en)})
            print(json.dumps(rows[-1]), flush=True)
    out = {"experiment": "TW4 zh-TW tokenizer coverage on held-out repo "
                         "Traditional-Chinese documents",
           "held_out": list(HELD_OUT),
           "eval": eval_stats(zh, en),
           "train_chars": {k: len(v) for k, v in corpora.items()},
           "results": rows,
           "results_sha256": hashlib.sha256(
               json.dumps(rows, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    for r in rows[:2]:
        print(json.dumps(r))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
