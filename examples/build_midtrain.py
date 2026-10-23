"""build_midtrain.py — LB5 decay-phase (mid-training) shard (v5.121).

Builds the shard that ``pilot_100m.py --schedule wsd --decay-data``
switches to at the first decay step, from one phase of a mixture
manifest (``data/mixtures/*.json``, LB3):

  python3 examples/build_midtrain.py \\
      --mixture data/mixtures/lb3_0p5b_v1.json --phase decay \\
      --source distill-d1=runs/distill/kept.jsonl \\
      --source fineweb-edu-top=data/raw/fineweb_edu_top \\
      --source finemath-top=data/raw/finemath4 \\
      --tokenizer data/pilot/tokenizer.json --tokens 2e9 --out data/mid

Every source of the phase needs a local copy (``--source name=PATH``;
.txt / .jsonl files or directories, as pilot_prepare_data.py reads).
``text`` sources pass the LB3 heuristics plus their manifest
``zh_variants`` and, with --quality-model, ``quality_threshold``;
``trace`` sources are D1 records rendered by ``mid_train.render_trace``.
The phase's weights split --tokens; ``mid_train.mix_docs`` interleaves
the documents. Sources whose licence is not verified are refused unless
--allow-unverified (pilot runs on local test data only).

Writes ``train.bin`` and ``meta.json`` (tokenizer hash, mixture hash,
per-source tokens / docs / epochs / short) into --out.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from examples.data_quality import read_docs  # noqa: E402
from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, mixture_sha256, usable_sources)
from helioslm_v5.src.training.data_quality import (  # noqa: E402
    HashedNgramClassifier, filter_docs)
from helioslm_v5.src.training.mid_train import (  # noqa: E402
    mix_docs, render_trace)


def read_records(path):
    files = (sorted(Path(path).rglob("*.jsonl"))
             if Path(path).is_dir() else [Path(path)])
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--phase", choices=["main", "decay"], default="decay")
    ap.add_argument("--source", action="append", default=[],
                    metavar="NAME=PATH")
    ap.add_argument("--field", default="text")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--quality-model")
    ap.add_argument("--max-epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    mix = load_mixture(a.mixture)
    srcs = [s for s in mix["sources"] if s.get("phase", "main") == a.phase]
    if not srcs:
        sys.exit(f"mixture has no {a.phase}-phase sources")
    usable = {s["name"] for s in usable_sources(mix)}
    blocked = [s["name"] for s in srcs if s["name"] not in usable]
    if blocked and not a.allow_unverified:
        sys.exit(f"licence not verified for {blocked}; verify them in the "
                 "manifest or pass --allow-unverified for a local pilot")
    paths = dict(x.split("=", 1) for x in a.source)
    missing = [s["name"] for s in srcs if s["name"] not in paths]
    if missing:
        sys.exit(f"no --source for {missing}")
    clf = (HashedNgramClassifier.load(a.quality_model)
           if a.quality_model else None)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = tok.get_vocab_size() - 1

    total = sum(s["weight"] for s in srcs)
    budgets, docs, filters = {}, {}, {}
    for s in srcs:
        name = s["name"]
        budgets[name] = max(1, int(round(a.tokens * s["weight"] / total)))
        if s.get("format", "text") == "trace":
            texts = [render_trace(r, s.get("think_ratio", 1.0))
                     for r in read_records(paths[name])]
        else:
            texts, rep = filter_docs(
                read_docs([paths[name]], a.field),
                variants=set(s["zh_variants"]) if s.get("zh_variants")
                else None,
                classifier=clf if s.get("quality_threshold") is not None
                else None,
                threshold=s.get("quality_threshold") or 0.5)
            filters[name] = rep
        docs[name] = [tok.encode(t).ids + [eot] for t in texts]
        if not docs[name]:
            sys.exit(f"{name}: no documents left")
    order, report = mix_docs(docs, budgets, seed=a.seed,
                             max_epochs=a.max_epochs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "train.bin", "wb") as f:
        for _, ids in order:
            if max(ids) > 65535:
                sys.exit("token ids must fit in uint16")
            f.write(np.asarray(ids, dtype=np.uint16).tobytes())
            n += len(ids)
    for name, r in report.items():
        r["budget"] = budgets[name]
        if name in filters:
            r["filter"] = filters[name]
    meta = {"tokenizer_sha256": hashlib.sha256(
                Path(a.tokenizer).read_bytes()).hexdigest(),
            "mixture": mix["name"], "mixture_sha256": mixture_sha256(mix),
            "phase": a.phase, "tokens": n, "unverified_licences": blocked,
            "sources": report}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"tokens": n, "sources": {
        k: {x: v[x] for x in ("tokens", "epochs", "short")}
        for k, v in report.items()}}))
    return meta


if __name__ == "__main__":
    main()
