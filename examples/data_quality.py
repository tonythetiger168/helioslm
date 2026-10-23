"""data_quality.py — LB3 corpus tools (v5.116).

  train    fit the hashed n-gram quality classifier on positives (curated
           text, teacher answers) vs negatives (random web text):
             --pos PATH ... --neg PATH ... --out MODEL
  score    score documents and report the kept share per threshold:
             --model MODEL --text PATH ... [--thresholds 0.3 0.5 0.7]
  profile  heuristic drops and zh variant counts for documents:
             --text PATH ...
  budget   validate a mixture manifest and print its token budget:
             --mixture data/mixtures/lb3_0p5b_v1.json --tokens 100e9

Documents are read like pilot_prepare_data.py: .txt files (one document
each), .jsonl (field --field, default "text"), or directories of those.
Filtering while building shards is done by pilot_prepare_data.py with
--heuristics / --zh-variants / --quality-model.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, mixture_sha256, token_budget, usable_sources)
from helioslm_v5.src.training.data_quality import (  # noqa: E402
    HashedNgramClassifier, heuristic_check, zh_variant)


def read_docs(paths, field="text"):
    files = []
    for p in map(Path, paths):
        files += (sorted(q for q in p.rglob("*")
                         if q.suffix in (".txt", ".jsonl"))
                  if p.is_dir() else [p])
    for f in files:
        if f.suffix == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = json.loads(line).get(field)
                    if t:
                        yield t
        else:
            t = f.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                yield t


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--pos", nargs="+", required=True)
    t.add_argument("--neg", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--dim", type=int, default=1 << 18)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--text", nargs="+", required=True)
    s.add_argument("--thresholds", nargs="*", type=float,
                   default=[0.3, 0.5, 0.7])
    p = sub.add_parser("profile")
    p.add_argument("--text", nargs="+", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--mixture", required=True)
    b.add_argument("--tokens", type=float, required=True)
    b.add_argument("--decay-frac", type=float, default=0.2)
    for x in (t, s, p):
        x.add_argument("--field", default="text")
    a = ap.parse_args(argv)

    if a.cmd == "train":
        pos = list(read_docs(a.pos, a.field))
        neg = list(read_docs(a.neg, a.field))
        clf = HashedNgramClassifier(a.dim).fit(pos, neg, epochs=a.epochs,
                                               seed=a.seed)
        path = clf.save(a.out)
        acc = (sum(clf.score(d) >= 0.5 for d in pos)
               + sum(clf.score(d) < 0.5 for d in neg)) / (len(pos) + len(neg))
        out = {"model": str(path), "pos": len(pos), "neg": len(neg),
               "train_accuracy": round(acc, 4)}
    elif a.cmd == "score":
        clf = HashedNgramClassifier.load(a.model)
        sc = [clf.score(d) for d in read_docs(a.text, a.field)]
        out = {"docs": len(sc), "kept_share": {
            str(th): round(sum(x >= th for x in sc) / max(len(sc), 1), 4)
            for th in a.thresholds}}
    elif a.cmd == "profile":
        drops, variants, n = Counter(), Counter(), 0
        for d in read_docs(a.text, a.field):
            n += 1
            ok, reasons = heuristic_check(d)
            drops.update(reasons or ["kept"])
            variants[zh_variant(d)] += 1
        out = {"docs": n, "heuristics": dict(drops),
               "zh_variant": dict(variants)}
    else:
        mix = load_mixture(a.mixture)
        out = {"name": mix["name"], "sha256": mixture_sha256(mix),
               "tokens": token_budget(mix, int(a.tokens), a.decay_frac),
               "usable": [x["name"] for x in usable_sources(mix)]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
