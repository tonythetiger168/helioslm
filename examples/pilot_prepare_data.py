"""pilot_prepare_data.py — token shards for the 100M pilot run (v5.83).

Reads text documents, trains (or loads) a byte-level BPE tokenizer, and
writes ``train.bin`` / ``val.bin`` (uint16), ``tokenizer.json`` and
``meta.json`` (tokenizer sha256, document / token / UTF-8 byte / character
counts per split, the sources) into ``--out``.

Sources (any mix):
  --text PATH ...      .txt files (one document each), .jsonl files (one
                       document per line, field --field, default "text"),
                       or directories of those;
  --hf NAME[:CONFIG]   a Hugging Face dataset, streamed (needs the
                       ``datasets`` package and network access; run it on
                       your own machine or the cluster), --split, --max-docs.

The validation split is chosen per document by a hash of its index, so it
is fixed for a given source order (--val-fraction, default 0.005).

Tokenizer: --tokenizer PATH loads an existing tokenizer.json; otherwise a
byte-level BPE (--vocab, default 32768) is trained on the TRAIN documents
only, seeded with the common Chinese characters (--zh-seed both|tw|cn|none,
default both; v5.80) so each is a single token.

Example (RTX 4060 pilot, Traditional + Simplified + English web text):
  python examples/pilot_prepare_data.py --hf HuggingFaceFW/fineweb-2:cmn_Hani \
      --max-docs 200000 --out data/pilot
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.pretrain import write_shards  # noqa: E402


def iter_text_sources(paths, field="text"):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(q for q in p.rglob("*")
                            if q.suffix in (".txt", ".jsonl"))
        else:
            files.append(p)
    for f in files:
        if f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        t = json.loads(line).get(field)
                        if t:
                            yield t
        else:
            t = f.read_text(encoding="utf-8", errors="replace")
            if t.strip():
                yield t


def iter_hf(name, split, max_docs, field="text"):
    from datasets import load_dataset
    repo, _, config = name.partition(":")
    ds = load_dataset(repo, config or None, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_docs and i >= max_docs:
            break
        t = row.get(field)
        if t:
            yield t


def is_val(index, fraction):
    h = int(hashlib.sha256(str(index).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


def train_tokenizer(texts, vocab, zh_seed):
    from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                            trainers)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|endoftext|>"])
    it = iter(texts)
    if zh_seed != "none":
        from helioslm_v5.eval.zh_tokenizer_coverage import zh_seed_text
        seed = zh_seed_text(zh_seed)
        it = (t for src in (texts, [seed]) for t in src)
    tok.train_from_iterator(it, trainer=trainer)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--field", default="text")
    ap.add_argument("--hf", action="append", default=[])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--tokenizer")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--zh-seed", choices=["both", "tw", "cn", "none"],
                    default="both")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.vocab > 65536:
        sys.exit("--vocab must fit uint16 shards (<= 65536)")
    docs = list(iter_text_sources(args.text, args.field))
    for name in args.hf:
        docs += list(iter_hf(name, args.split, args.max_docs, args.field))
    if not docs:
        sys.exit("no documents read")
    split = {"train": [], "val": []}
    for i, d in enumerate(docs):
        split["val" if is_val(i, args.val_fraction) else "train"].append(d)
    if not split["val"]:                    # tiny corpora: keep one doc
        split["val"].append(split["train"].pop())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    else:
        tok = train_tokenizer(split["train"], args.vocab, args.zh_seed)
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = tok.get_vocab_size() - 1
    meta = {"tokenizer_sha256": hashlib.sha256(
                (out / "tokenizer.json").read_bytes()).hexdigest(),
            "vocab_size": tok.get_vocab_size(), "eot_id": eot,
            "zh_seed": None if args.tokenizer else args.zh_seed,
            "sources": {"text": args.text, "hf": args.hf,
                        "split": args.split, "max_docs": args.max_docs},
            "splits": {}}
    for name, ds in split.items():
        ids = (tok.encode(d).ids + [eot] for d in ds)
        n_tok = write_shards(ids, out, name)
        meta["splits"][name] = {
            "docs": len(ds), "tokens": n_tok,
            "chars": sum(len(d) for d in ds),
            "utf8_bytes": sum(len(d.encode("utf-8")) for d in ds)}
    s = meta["splits"]["val"]
    meta["val_bytes_per_token"] = round(s["utf8_bytes"] / max(s["tokens"], 1), 6)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["splits"]), "vocab", meta["vocab_size"])


if __name__ == "__main__":
    main()
