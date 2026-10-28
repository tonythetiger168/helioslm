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

A ``git:<url>@<commit>`` source (v5.183) is checked out at that commit
(shallow) under ``<out>/_git/<name>``; the licence in its LICENSE file
stands in for the card licence (Solderpad 2.x counts as Apache-2.0, which
it lets the user choose), and the files matching ``include`` and not
``exclude`` (sorted, exact duplicates and files over 1 MB dropped) are the
documents, each prefixed with a ``// file: <path>`` line.

A source with ``projects`` (v5.186, HDL track H5) is read as projects
that must synthesise (``hdl_projects``): git sources by ``units`` globs
plus shared ``keep`` libraries, hf GitHub code grouped by repository
after the permissive-licence filter (``per-repository`` licence), or one
row per project (``per_row``). Each passing project is one document in
dependency order; near-duplicate projects and files overlapping the
``--decontam`` benchmarks are dropped. ``requires`` skips a source until
the named sources have a passing project (its share is not given to
the others).

``quality_threshold`` is not applied (no trained quality classifier yet).
``<out>/manifest.json`` records per source: phase, weight, licence check,
dataset revision, tokens, documents kept / seen, and the filter report;
``cpt_hf.py`` reads it. A source that fails is recorded and the run goes
on. Needs ``datasets``, ``huggingface_hub`` and ``tokenizers`` (the pod);
HF_TOKEN for gated datasets.
"""
import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, mixture_sha256, token_budget)
from helioslm_v5.src.training.data_quality import filter_docs  # noqa: E402
from helioslm_v5.src.training import hdl_projects  # noqa: E402

_LICENSE_ALIASES = {"odc-by-1.0": "odc-by", "cc0-1.0": "cc0"}


def parse_source(src):
    """'hf:repo[:config]' -> (repo, config or None); 'git:url@commit' ->
    ("git", url, commit); other schemes -> None."""
    if src.startswith("git:"):
        url, _, commit = src[4:].rpartition("@")
        if not url.startswith("https://") or len(commit) != 40:
            return None
        return "git", url, commit
    if not src.startswith("hf:"):
        return None
    parts = src[3:].split(":", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


MAX_FILE_BYTES = 1 << 20
_LICENCE_TEXTS = (("solderpad hardware license version 0.51", "SHL-0.51"),
                  ("Solderpad Hardware Licen", "Apache-2.0"),
                  ("Apache License", "Apache-2.0"),
                  ("MIT License", "MIT"),
                  ("Permission is hereby granted, free of charge", "MIT"))


def git_checkout(url, commit, dest):
    """Shallow checkout of ``commit`` into ``dest``; returns ``dest``."""
    dest = Path(dest)
    if not (dest / ".git").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", url]):
            subprocess.run(cmd, cwd=dest, check=True)
    for cmd in (["git", "fetch", "-q", "--depth", "1", "origin", commit],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, cwd=dest, check=True)
    return dest


def git_licence(root):
    """The licence named by the repo's top-level LICENSE file, or None."""
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        p = Path(root) / name
        if p.is_file():
            head = p.read_text(encoding="utf-8", errors="replace")[:4000]
            low = head.lower()
            return next((lic for key, lic in _LICENCE_TEXTS
                         if key.lower() in low), None)
    return None


def git_texts(root, include, exclude=()):
    """Documents of a checkout: files matching ``include`` and not
    ``exclude`` (repo-relative globs), sorted, exact duplicates and files
    over ``MAX_FILE_BYTES`` dropped, each prefixed ``// file: <path>``."""
    root, seen = Path(root), set()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
    for p in paths:
        rel = p.relative_to(root).as_posix()
        if not any(fnmatch.fnmatch(rel, g) for g in include):
            continue
        if any(fnmatch.fnmatch(rel, g) for g in exclude):
            continue
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) > MAX_FILE_BYTES or digest in seen:
            continue
        seen.add(digest)
        yield f"// file: {rel}\n" + data.decode("utf-8", errors="replace")


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
            r["reason"] = "not a hf: or git: source"
        elif s.get("license") == "review":
            r["reason"] = "licence under review"
        elif s["name"] in no_access:
            r["reason"] = f"{NO_ACCESS} ({no_access[s['name']]})"
        elif s.get("license") == "per-repository":
            r["usable"] = True
            usable.append(s)
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


PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
              "0bsd", "cc0-1.0", "unlicense", "zlib", "shl-0.51", "shl-2.0",
              "shl-2.1", "bsl-1.0", "mit-0", "x11"}


def licence_ok(lics, allow=PERMISSIVE):
    """True when every licence of a row is in ``allow`` (and there is at
    least one): The Stack's ``max_stars_repo_licenses``."""
    if isinstance(lics, str):
        lics = [lics]
    lics = [str(x).lower() for x in (lics or [])]
    return bool(lics) and all(x in allow for x in lics)


def stack_rows(ds, s, field):
    """(repo, path, text) of rows whose repository licences are all
    permissive (source keys ``repo_field``, ``path_field``,
    ``licence_field``, ``allow_licences``)."""
    allow = {x.lower() for x in s.get("allow_licences", PERMISSIVE)}
    lf = s.get("licence_field", "max_stars_repo_licenses")
    rf = s.get("repo_field", "max_stars_repo_name")
    pf = s.get("path_field", "max_stars_repo_path")
    for row in ds:
        if licence_ok(row.get(lf), allow):
            yield row.get(rf), row.get(pf), row.get(field) or ""


class ProgressReport(list):
    """The gate report; every row is also printed (flushed) as it comes,
    so a long pod run shows its progress."""

    def __init__(self, name):
        super().__init__()
        self.name, self.t0 = name, time.time()

    def append(self, row):
        super().append(row)
        passed = sum(1 for r in self if r.get("pass"))
        print(json.dumps({"source": self.name, "n": len(self), "passed": passed,
                          "s": round(time.time() - self.t0),
                          "project": row.get("project"), "pass": row.get("pass"),
                          "error": (row.get("error") or "")[:120]},
                         ensure_ascii=False), flush=True)


def hdl_texts(s, ps, out, ds=None, field="text", *, tool=None, dedup=None,
              decontam=None, workers=1, gate=True):
    """Project documents of an HDL source with ``projects`` (v5.186), and
    the per-project gate report. git: every HDL file of the checkout is
    read (vendored ones for dependencies only), the ``include`` /
    ``exclude`` files are grouped into ``projects.units``. hf: rows are
    grouped by repository after the licence filter."""
    cfg, report = s["projects"], ProgressReport(s["name"])
    kw = dict(gate=gate and cfg.get("gate", True), tool=tool, dedup=dedup,
              decontam=decontam, report=report, workers=workers,
              timeout=cfg.get("timeout", 600))
    if cfg.get("tops"):
        kw["tops"] = cfg["tops"]
    if ps[0] == "git":
        root = out / "_git" / s["name"]
        all_files = hdl_projects.read_tree(root, MAX_FILE_BYTES)
        files = {p: t for p, t in all_files.items()
                 if any(fnmatch.fnmatch(p, g) for g in s["include"])
                 and not any(fnmatch.fnmatch(p, g)
                             for g in s.get("exclude", ()))}
        docs = hdl_projects.project_docs(
            s["name"], files, all_files, root, units=cfg.get("units"),
            keep=cfg.get("keep", ()), **kw)
        return iter(docs), report
    if cfg.get("per_row"):
        return per_row_docs(ds, field, out, kw), report
    return hdl_projects.repo_docs(stack_rows(ds, s, field),
                                  workdir=out, **kw), report


def per_row_docs(ds, field, out, kw, batch=256):
    """Rows without repository names (one file each): every row is its
    own project, gated alone, ``batch`` rows at a time (streamed)."""
    import itertools
    rows = ((f"row/{i:09d}", "file.sv", row.get(field) or "")
            for i, row in enumerate(ds))
    while True:
        chunk = list(itertools.islice(rows, batch))
        if not chunk:
            return
        yield from hdl_projects.repo_docs(chunk, batch=batch, workdir=out,
                                          **kw)


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
    ap.add_argument("--decontam", nargs="*", default=[],
                    help="benchmark files / directories: HDL documents "
                         "sharing a 13-gram with them are dropped")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the lint + synth gate of HDL projects")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel gate jobs (0: the CPUs this container may use)")
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
        if ps[0] == "git":
            try:
                root = git_checkout(ps[1], ps[2], out / "_git" / s["name"])
                cards[s["name"]], revs[s["name"]] = git_licence(root), ps[2]
            except Exception as e:
                no_access[s["name"]] = f"git {type(e).__name__}"
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

    dedup = hdl_projects.MinHashDedup()
    decontam = hdl_projects.Decontam(hdl_projects.bench_texts(a.decontam))
    tool = hdl_projects.tools()
    if a.workers <= 0:
        from helioslm_v5.eval.uvm_bench import cpu_limit
        a.workers = cpu_limit()
    manifest["hdl_gate"] = {k: bool(v) for k, v in tool.items()}
    manifest["decontam_grams"] = len(decontam.grams)

    for s in mix["sources"]:
        r = dict(rows[s["name"]], card_licence=cards.get(s["name"]),
                 revision=revs.get(s["name"]))
        manifest["sources"][s["name"]] = r
        if not r["usable"]:
            save()
            continue
        need = [n for n in s.get("requires", ())
                if not manifest["sources"].get(n, {}).get("projects_passed")]
        if need:
            r.update(usable=False, status="skipped",
                     reason=f"needs a passing project in {', '.join(need)}")
            save()
            continue
        ps = parse_source(s["source"])
        (out / r["phase"]).mkdir(exist_ok=True)
        path = out / r["phase"] / f"{s['name']}.bin"
        t0 = time.time()
        try:
            report = None
            ds = None
            field = s.get("field", a.field)
            if ps[0] != "git":
                ds = load_dataset(ps[0], ps[1], split="train", streaming=True,
                                  revision=r["revision"], **load_kwargs(s))
            if s.get("projects"):
                texts, report = hdl_texts(
                    s, ps, out, ds, field, tool=tool, dedup=dedup,
                    decontam=decontam, workers=a.workers,
                    gate=not a.no_gate)
            elif ps[0] == "git":
                texts = git_texts(out / "_git" / s["name"], s["include"],
                                  s.get("exclude", ()))
            else:
                texts = (row.get(field) or "" for row in ds)
            res = write_source(
                texts, lambda b: [e.ids for e in tok.encode_batch(
                    b, add_special_tokens=False)],
                path, r["budget"], bos, eos,
                variants=set(s["zh_variants"]) if s.get("zh_variants")
                else None, heuristics=s.get("heuristics", True))
            r.update(res, path=str(path.relative_to(out)),
                     status="ok" if res["tokens"] >= r["budget"] else "short")
            if report is not None:
                r["projects"] = report
                r["projects_passed"] = sum(1 for x in report if x.get("pass"))
                r["projects_gated"] = sum(1 for x in report
                                          if x.get("pass") is not None)
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
