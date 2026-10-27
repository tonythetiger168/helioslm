"""fetch_mixture.py — route B: download and tokenise a mixture's sources
into per-source token shards (v5.137).

  python examples/fetch_mixture.py \\
      --mixture data/mixtures/b1_gemma4_e4b_cpt_v1.json \\
      --tokenizer google/gemma-4-E2B --tokens 3e9 --out /workspace/cpt_data

For every ``hf:`` source of the mixture (``hf:repo[:config]``):

1. its dataset card licence is read (``huggingface_hub``) and compared
   with the manifest's; a mismatch, a ``review`` licence or a ``path:``
   source is skipped and reported, never used;
2. its token budget is ``data_mixture.token_budget`` over the usable
   sources (weights renormalised within each phase, so a skipped source's
   share goes to the others);
3. rows are streamed (``datasets``, ``streaming=True``; the source's
   ``data_dir`` / ``data_files``), its ``field`` (default ``--field``) is
   passed through the LB3 heuristics (unless ``"heuristics": false``) and
   the source's ``zh_variants``, tokenised in
   batches (``tokenizers`` ``encode_batch``, multithreaded), wrapped as
   BOS + ids + EOS and appended to ``<out>/<phase>/<name>.bin`` (uint32)
   until the budget is reached.

``quality_threshold`` is not applied (no trained quality classifier yet).
``<out>/manifest.json`` records per source: phase, weight, licence check,
dataset revision, tokens, documents kept / seen, and the filter report;
``cpt_hf.py`` reads it. A source that fails is recorded and the run goes
on. Needs ``datasets``, ``huggingface_hub`` and ``tokenizers`` (the pod);
HF_TOKEN for gated datasets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, mixture_sha256, token_budget)
from helioslm_v5.src.training.data_quality import filter_docs  # noqa: E402

_LICENSE_ALIASES = {"odc-by-1.0": "odc-by", "cc0-1.0": "cc0"}


def parse_source(src):
    """'hf:repo[:config]' -> (repo, config or None); other schemes -> None."""
    if not src.startswith("hf:"):
        return None
    parts = src[3:].split(":", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def _family(lic):
    lic = _LICENSE_ALIASES.get(lic.lower(), lic.lower())
    for fam in ("cc-by-sa", "cc-by-nc", "cc-by-nd", "cc-by"):
        if lic.startswith(fam):
            return fam
    return lic


def licence_matches(manifest, card):
    """Manifest licence vs the card's (a string or a list): same licence
    family (CC versions may differ, e.g. Wikipedia's cc-by-sa-3.0 card
    against CC-BY-SA-4.0)."""
    cards = card if isinstance(card, (list, tuple)) else [card]
    want = _family(manifest)
    return any(c and _family(str(c)) == want for c in cards)


NO_ACCESS = "no download access"


def plan(mix, tokens, decay_frac, card_licences, no_access=None):
    """{name: {"budget", "phase", "weight", "usable", "reason"}}.

    ``card_licences``: {name: card licence or None if unreadable}.
    ``no_access``: {name: why} for sources this HF account cannot
    download (gated, not yet approved). Usable sources share each phase's
    tokens by renormalised weight."""
    no_access = no_access or {}
    rows, usable = {}, []
    for s in mix["sources"]:
        r = {"phase": s.get("phase", "main"), "weight": s["weight"],
             "usable": False, "reason": None, "budget": 0}
        rows[s["name"]] = r
        if parse_source(s["source"]) is None:
            r["reason"] = "not a hf: source"
        elif s.get("license") == "review":
            r["reason"] = "licence under review"
        elif s["name"] in no_access:
            r["reason"] = f"{NO_ACCESS} ({no_access[s['name']]})"
        elif card_licences.get(s["name"]) is None:
            r["reason"] = "card licence unreadable"
        elif not licence_matches(s["license"], card_licences[s["name"]]):
            r["reason"] = (f"card says {card_licences[s['name']]!r}, "
                           f"manifest {s['license']!r}")
        else:
            r["usable"] = True
            usable.append(s)
    if not any(s.get("phase", "main") == "main" for s in usable):
        raise SystemExit("no usable main-phase source")
    sub = {"name": mix["name"], "sources": []}
    for ph in ("main", "decay"):
        ss = [s for s in usable if s.get("phase", "main") == ph]
        tot = sum(s["weight"] for s in ss)
        sub["sources"] += [dict(s, weight=s["weight"] / tot) for s in ss]
    has_decay = any(s.get("phase") == "decay" for s in sub["sources"])
    for name, b in token_budget(sub, tokens,
                                decay_frac if has_decay else 0.2).items():
        rows[name]["budget"] = b
        rows[name]["weight"] = next(s["weight"] for s in sub["sources"]
                                    if s["name"] == name)
    return rows


def blocked(rows):
    """Sources dropped only because this account cannot download them."""
    return sorted(n for n, r in rows.items()
                  if (r["reason"] or "").startswith(NO_ACCESS))


def write_source(texts, encode_batch, path, budget, bos, eos, *,
                 variants=None, heuristics=True, batch=1000):
    """Filter, tokenise and append documents to ``path`` (uint32) until
    ``budget`` tokens. ``heuristics`` False keeps documents the prose
    heuristics would drop (source code). Returns {"tokens", "docs",
    "seen", "filter"}."""
    n = docs = seen = 0
    report = {"in": 0, "kept": 0, "dropped": {}}
    buf = []
    with open(path, "wb") as f:
        def flush(chunk):
            nonlocal n, docs
            kept, rep = filter_docs(chunk, heuristics=heuristics,
                                    variants=variants)
            report["in"] += rep["in"]
            report["kept"] += rep["kept"]
            for k, v in rep["dropped"].items():
                report["dropped"][k] = report["dropped"].get(k, 0) + v
            for ids in encode_batch(kept):
                if n >= budget:
                    return True
                row = ([bos] if bos is not None else []) + list(ids) + [eos]
                f.write(np.asarray(row, dtype=np.uint32).tobytes())
                n += len(row)
                docs += 1
            return n >= budget
        for t in texts:
            seen += 1
            buf.append(t)
            if len(buf) == batch:
                if flush(buf):
                    buf = []
                    break
                buf = []
        if buf and n < budget:
            flush(buf)
    return {"tokens": n, "docs": docs, "seen": seen, "filter": report}


def load_kwargs(s):
    """A source's ``data_dir`` / ``data_files`` for ``load_dataset``."""
    return {k: s[k] for k in ("data_dir", "data_files") if k in s}


def _card_licence(api, repo):
    info = api.dataset_info(repo)
    card = getattr(info, "card_data", None) or {}
    lic = card.get("license") if hasattr(card, "get") else None
    return lic, info.sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="Hub model id or tokenizer.json path")
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="licence + access plan only, no download")
    ap.add_argument("--allow-blocked", action="store_true",
                    help="go on when a source has no download access "
                         "(its share goes to the rest of its phase)")
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from tokenizers import Tokenizer

    mix = load_mixture(a.mixture)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    api, cards, revs, no_access = HfApi(), {}, {}, {}
    for s in mix["sources"]:
        ps = parse_source(s["source"])
        if ps is None or s.get("license") == "review":
            continue
        try:
            cards[s["name"]], revs[s["name"]] = _card_licence(api, ps[0])
        except Exception as e:
            cards[s["name"]] = None
            revs[s["name"]] = f"{type(e).__name__}: {e}"[:200]
        try:
            api.auth_check(ps[0], repo_type="dataset")
        except Exception as e:
            no_access[s["name"]] = type(e).__name__
    rows = plan(mix, a.tokens, a.decay_frac, cards, no_access)
    for name, r in rows.items():
        print(f"{name:28} {r['phase']:5} {r['budget'] / 1e6:9.1f}M  "
              f"{r['reason'] or 'ok'}")
    stuck = blocked(rows)
    manifest = {"schema": "cpt-data/1", "mixture": mix["name"],
                "mixture_sha256": mixture_sha256(mix),
                "tokenizer": a.tokenizer, "dtype": "uint32",
                "bos": None, "eos": None, "sources": {}}

    def save():
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    if stuck and not a.allow_blocked:
        manifest["sources"] = rows
        save()
        raise SystemExit(
            f"no download access to {', '.join(stuck)}: request access on "
            "the Hub, or pass --allow-blocked to train without them")
    if a.check_only:
        manifest["sources"] = rows
        save()
        return manifest

    tok = (Tokenizer.from_file(a.tokenizer) if Path(a.tokenizer).exists()
           else Tokenizer.from_pretrained(a.tokenizer))
    bos = next((tok.token_to_id(t) for t in ("<bos>", "<s>")
                if tok.token_to_id(t) is not None), None)
    eos = next((tok.token_to_id(t) for t in ("<eos>", "</s>", "<|endoftext|>")
                if tok.token_to_id(t) is not None), None)
    if eos is None:
        raise SystemExit("tokenizer has no EOS token")
    manifest.update(bos=bos, eos=eos)

    for s in mix["sources"]:
        r = dict(rows[s["name"]], card_licence=cards.get(s["name"]),
                 revision=revs.get(s["name"]))
        manifest["sources"][s["name"]] = r
        if not r["usable"]:
            save()
            continue
        repo, cfg = parse_source(s["source"])
        (out / r["phase"]).mkdir(exist_ok=True)
        path = out / r["phase"] / f"{s['name']}.bin"
        t0 = time.time()
        try:
            ds = load_dataset(repo, cfg, split="train", streaming=True,
                              revision=r["revision"], **load_kwargs(s))
            field = s.get("field", a.field)
            texts = (row.get(field) or "" for row in ds)
            res = write_source(
                texts, lambda b: [e.ids for e in tok.encode_batch(
                    b, add_special_tokens=False)],
                path, r["budget"], bos, eos,
                variants=set(s["zh_variants"]) if s.get("zh_variants")
                else None, heuristics=s.get("heuristics", True))
            r.update(res, path=str(path.relative_to(out)),
                     status="ok" if res["tokens"] >= r["budget"] else "short")
        except Exception as e:
            r.update(status="FAILED", error=f"{type(e).__name__}: {e}"[:500])
        r["seconds"] = round(time.time() - t0)
        save()
        print(json.dumps({s["name"]: {k: r.get(k) for k in
                                      ("status", "tokens", "budget")}}))
    save()
    return manifest


if __name__ == "__main__":
    main()
