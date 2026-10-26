"""tw2_prepare_corpus.py — roadmap TW2: a ~100M-token zh-TW pretraining
corpus as attributed JSONL shards (v5.154).

  python examples/tw2_prepare_corpus.py --out /workspace/tw2 \\
      --tokens 1e8 --tokenizer data/pilot/tokenizer.json \\
      --datagov /workspace/datagov_text/

Sources (``helioslm_v5/src/training/zhtw_corpus.py`` SOURCES), read in this
order so the first copy dedup keeps is the best-attributed one:

1. ``datagov``: local .jsonl files or directories (--datagov), one record
   per line with ``text`` and ``provider`` (the agency OGDL requires
   naming; records without it are dropped), optional ``title`` / ``url``.
   data.gov.tw is not streamed: download its text resources yourself and
   record the agency per document;
2. ``zhwiki``: wikimedia/wikipedia 20231101.zh, converted with OpenCC
   s2twp (needs ``pip install opencc``);
3. ``fineweb2``: HuggingFaceFW/fineweb-2 cmn_Hani, Traditional-only
   documents kept as is.

Each source gets ``--tokens`` x its --shares weight (renormalised over the
sources that are present), counted with --tokenizer when given, else in
units (CJK characters + Latin words; the manifest says which). A source
stops at its budget or after --max-scan records. Every kept document
passes the LB3 heuristics, the source's zh-variant rule and a global
exact + MinHash near-duplicate filter.

Output: ``<out>/<source>/<source>-NNNNN.jsonl`` (fields id, source,
text, license, attribution, [title, url]), ``<out>/manifest.json``
(settings, per-source licence, dataset revision, budget, drop counts,
Traditional share, shard sha256, and ``corpus_sha256`` over the shard
hashes) and ``<out>/ATTRIBUTION.md``. The same inputs and revisions give
the same ``corpus_sha256``. Tokenise with
``pilot_prepare_data.py --text <out> --tokenizer ...``.

--local SRC=PATH replaces an HF source with a local .jsonl (an offline
mirror, or the tests). HF sources need ``datasets`` and network access
to the HF data CDN (the pod); the revision is pinned to the dataset's
current commit unless --revision SRC=SHA is given.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.zhtw_corpus import (  # noqa: E402
    SOURCES, Deduper, new_stats, opencc_converter, process, trad_share)

ORDER = ("datagov", "zhwiki", "fineweb2")
DEFAULT_SHARES = {"datagov": 0.1, "zhwiki": 0.3, "fineweb2": 0.6}


def _kv(items, cast=str):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in SOURCES or not v:
            raise SystemExit(f"bad SRC=VALUE {it!r} (SRC in {list(SOURCES)})")
        out[k] = cast(v)
    return out


def iter_jsonl(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def iter_hf(spec, revision):
    from datasets import load_dataset
    return iter(load_dataset(spec["hf"], spec["config"], split="train",
                             streaming=True, revision=revision))


def hf_revision(repo):
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def budgets(total, shares, present):
    w = {s: shares[s] for s in present if shares.get(s, 0) > 0}
    z = sum(w.values())
    return {s: int(total * v / z) for s, v in w.items()}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=1e8)
    ap.add_argument("--shares", nargs="*", default=[],
                    help="SRC=WEIGHT, default datagov=0.1 zhwiki=0.3 "
                         "fineweb2=0.6")
    ap.add_argument("--datagov", nargs="*", default=[])
    ap.add_argument("--local", nargs="*", default=[])
    ap.add_argument("--revision", nargs="*", default=[])
    ap.add_argument("--sources", nargs="*", default=list(ORDER),
                    choices=list(ORDER))
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="records read per source at most (0 = no cap)")
    ap.add_argument("--shard-docs", type=int, default=50_000)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--dup-threshold", type=float, default=0.8)
    args = ap.parse_args(argv)

    shares = dict(DEFAULT_SHARES, **_kv(args.shares, float))
    local, revisions = _kv(args.local), _kv(args.revision)
    if args.datagov:
        local["datagov"] = args.datagov
    elif "datagov" in local:
        local["datagov"] = [local["datagov"]]
    present = [s for s in ORDER if s in args.sources
               and (s in local or SOURCES[s]["hf"])]
    skipped = {s: "no --datagov input" for s in args.sources
               if s not in present}
    budget = budgets(args.tokens, shares, present)

    count = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        count = lambda t: len(tok.encode(t).ids)  # noqa: E731
    unit = "tokens" if count else "units"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dedup = Deduper(args.num_perm, args.bands, args.dup_threshold)
    report = {}
    for src in present:
        spec = SOURCES[src]
        if src == "datagov":
            rows, rev = iter_jsonl(local[src]), None
        elif src in local:
            rows, rev = iter_jsonl([local[src]]), "local:" + local[src]
        else:
            rev = revisions.get(src) or hf_revision(spec["hf"])
            rows = iter_hf(spec, rev)
        conv = opencc_converter(spec["convert"]) if spec["convert"] else None
        st, shards, fh, n_in_shard = new_stats(), [], None, 0
        (out / src).mkdir(exist_ok=True)
        for rec in rows:
            if st[unit] >= budget[src]:
                break
            if args.max_scan and st["in"] >= args.max_scan:
                break
            doc = process(rec, src, spec, dedup, st, converter=conv,
                          count_tokens=count)
            if doc is None:
                continue
            if fh is None or n_in_shard >= args.shard_docs:
                if fh:
                    fh.close()
                path = out / src / f"{src}-{len(shards):05d}.jsonl"
                shards.append({"path": str(path.relative_to(out))})
                fh, n_in_shard = open(path, "w", encoding="utf-8"), 0
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_in_shard += 1
            shards[-1]["docs"] = n_in_shard
        if fh:
            fh.close()
        for sh in shards:
            sh["sha256"] = file_sha256(out / sh["path"])
        report[src] = {
            "license": spec["license"], "attribution": spec["attribution"],
            "hf": spec["hf"], "config": spec["config"], "revision": rev,
            "convert": spec["convert"], "zh_variants": list(spec["variants"]),
            "budget": budget[src], "budget_reached": st[unit] >= budget[src],
            "stats": st, "trad_share": trad_share(st), "shards": shards}
        print(f"{src}: kept {st['kept']}/{st['in']} docs, {st[unit]} "
              f"{unit} of {budget[src]}, trad share {trad_share(st)}, "
              f"dropped {st['dropped']}", flush=True)

    hashes = [sh["sha256"] for r in report.values() for sh in r["shards"]]
    manifest = {
        "name": "tw2-zhtw-v1", "budget_unit": unit,
        "tokenizer": args.tokenizer and file_sha256(args.tokenizer),
        "tokens_requested": int(args.tokens), "shares": shares,
        "dedup": {"num_perm": args.num_perm, "bands": args.bands,
                  "threshold": args.dup_threshold, "ngram": 5, "seed": 0},
        "sources": report, "skipped": skipped,
        "total": {"docs": sum(r["stats"]["kept"] for r in report.values()),
                  unit: sum(r["stats"][unit] for r in report.values())},
        "corpus_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    lines = ["# TW2 corpus attribution", "",
             "Each record carries its own `license` and `attribution`. "
             "Redistributing the zhwiki shards requires CC BY-SA 4.0; "
             "datagov records must keep their providing agency (OGDL).", ""]
    lines += [f"- **{s}** ({r['license']}): {r['attribution']}"
              for s, r in report.items()]
    (out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"total {manifest['total']}, corpus_sha256 "
          f"{manifest['corpus_sha256']}")
    return manifest


if __name__ == "__main__":
    main()
