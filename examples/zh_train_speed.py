"""zh_train_speed.py — Traditional-Chinese training speed: Big5-seeded BPE
vs plain BPE (helioslm_v5/eval/zh_train_speed.py).

Five byte-level BPEs (32000 requested), trained as in TW4:

  code_only        repo corpus with CJK stripped (never saw Chinese);
  code_only_big5   the same + the 5401 Big5 common characters as seed;
  code_only_both   the same + Big5 and the 3755 GB2312 Simplified common
                   characters (6722 together);
  repo             repo corpus (minus the TW4 held-out documents);
  repo_big5        the same + the Big5 seed.

Each gets the same tiny model, steps, batch and sequence length on the
repo's zh-TW training lines; held-out text = the TW4 held-out documents.

Usage:
  python examples/zh_train_speed.py --output benchmarks/zh_train_speed_2026-10-20.json
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

import torch  # noqa: E402

from helioslm_v5.eval.zh_tokenizer_coverage import (  # noqa: E402
    Bpe, eval_split, measure, train_bpe, training_corpus, zh_seed_text)
from helioslm_v5.eval.zh_train_speed import run, zh_training_text  # noqa: E402

# arm -> (BPE training corpus, seed scripts or None)
ARM_SPEC = {"code_only": ("code_only", None),
            "code_only_big5": ("code_only", "tw"),
            "code_only_both": ("code_only", "both"),
            "repo": ("repo", None),
            "repo_big5": ("repo", "tw")}
ARMS = tuple(ARM_SPEC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    corpora = {"code_only": training_corpus(_REPO, strip_cjk=True),
               "repo": training_corpus(_REPO)}
    train_text = zh_training_text(corpora["repo"])
    zh, en = eval_split(_REPO)
    results, timings = [], {}
    for arm in args.arms:
        corpus, seed = ARM_SPEC[arm]
        text = corpora[corpus]
        if seed:
            text = text + "\n" + zh_seed_text(seed)
        tok = train_bpe(text, args.vocab)
        cov = measure(Bpe(tok), zh, en)
        row = run(tok, train_text, _REPO, args.steps, args.seq, args.batch,
                  args.lr, args.seed, args.eval_every)
        timings[arm] = row.pop("timing")
        results.append({"arm": arm, "coverage": cov, **row})
        print(json.dumps(results[-1]), json.dumps(timings[arm]), flush=True)
    out = {"experiment": "zh-TW training speed: Big5-seeded vs plain "
                         "byte-level BPE, same model / steps / batch / seq",
           "train_text_chars": len(train_text),
           "train": {k: v for k, v in vars(args).items()
                     if k not in ("output", "arms", "threads")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "timing_not_hashed": {"threads": args.threads, **timings},
           "wall_seconds": round(time.time() - t0, 1)}
    if args.output:
        Path(args.output).write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
