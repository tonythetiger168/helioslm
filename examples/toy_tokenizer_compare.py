"""Issue #5 acceptance run: BPE toy model vs the char baseline, paired.

For every seed both arms get the same file-level split (the seed picks it),
the same text (characters below the char vocabulary, --max-ord), the same
model, steps, batch, sequence length and learning rate, and the same
evaluation: held-out NLL per UTF-8 byte. Writes one JSON with every run,
the per-seed differences and their mean.

    python examples/toy_tokenizer_compare.py --seeds 13 14 15 --steps 1000 \\
        --out benchmarks/toy_bpe_vs_char_2026-10-21.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import train_toy_bpe  # noqa: E402

KEEP = ("tokenizer", "seed", "vocab_size", "train_tokens", "train_utf8_bytes",
        "val_tokens", "val_utf8_bytes", "val_loss_per_token",
        "val_nll_per_utf8_byte", "val_text_sha256", "train_seconds", "sample",
        "sample_real_word_rate")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 14, 15])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-ord", type=int, default=1024)
    ap.add_argument("--keep-checkpoints", default=None,
                    help="directory for the trained checkpoints (default: "
                         "discarded); a run whose metrics are already there "
                         "with the same settings is reused, so an interrupted "
                         "comparison resumes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds must be distinct")
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(args.keep_checkpoints or tmp)
        ck.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            for kind in ("char", "bpe"):
                ns = argparse.Namespace(
                    steps=args.steps, batch=args.batch, seq=args.seq, lr=args.lr,
                    seed=seed, vocab_size=None, tokenizer=kind,
                    max_ord=args.max_ord, zh_seed=None,
                    out=str(ck / f"toy_{kind}_s{seed}.pt"))
                done = Path(ns.out).with_suffix(".metrics.json")
                m = json.loads(done.read_text()) if done.exists() else None
                # resume: reuse a finished run only if it had the same settings
                if m is None or any(m.get(k) != getattr(ns, k) for k in
                                    ("steps", "batch", "seq", "lr", "max_ord")):
                    m = train_toy_bpe.run(ns)
                runs.append({k: m[k] for k in KEEP})
                print(json.dumps({k: runs[-1][k] for k in
                                  ("tokenizer", "seed", "val_nll_per_utf8_byte",
                                   "sample_real_word_rate")}), flush=True)
    paired = []
    for seed in args.seeds:
        c, b = (next(r for r in runs if r["seed"] == seed and r["tokenizer"] == k)
                for k in ("char", "bytelevel-bpe"))
        assert c["val_text_sha256"] == b["val_text_sha256"]   # identical text
        paired.append({"seed": seed,
                       "char_nll_per_byte": c["val_nll_per_utf8_byte"],
                       "bpe_nll_per_byte": b["val_nll_per_utf8_byte"],
                       "bpe_minus_char": b["val_nll_per_utf8_byte"] - c["val_nll_per_utf8_byte"],
                       "char_real_word_rate": c["sample_real_word_rate"],
                       "bpe_real_word_rate": b["sample_real_word_rate"]})
    diffs = [p["bpe_minus_char"] for p in paired]
    summary = {"mean_bpe_minus_char_nll_per_byte": statistics.mean(diffs),
               "bpe_better_in_seeds": sum(d < 0 for d in diffs),
               "seeds": len(diffs),
               "mean_char_real_word_rate": statistics.mean(p["char_real_word_rate"] for p in paired),
               "mean_bpe_real_word_rate": statistics.mean(p["bpe_real_word_rate"] for p in paired)}
    payload = {"experiment": "issue5_toy_bpe_vs_char",
               "config": {"model": "lite", "steps": args.steps, "batch": args.batch,
                          "seq": args.seq, "lr": args.lr, "max_ord": args.max_ord,
                          "seeds": args.seeds},
               "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "platform": platform.platform()},
               "runs": runs, "paired": paired, "summary": summary}
    payload["runs_sha256"] = hashlib.sha256(
        json.dumps(runs, sort_keys=True).encode()).hexdigest()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
