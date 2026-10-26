"""zh-TW pretraining corpus: s2twp conversion, dedup, attribution
(v5.154, roadmap TW2).

TW2 asks for ~100M tokens of Taiwan Traditional Chinese from three
licence-clean sources, with reproducible dedup and Traditional-share
statistics and every document keeping its licence and attribution:

- zh Wikipedia (CC BY-SA 4.0): articles are stored in whichever variant
  their editors wrote, so they are converted with OpenCC ``s2twp``
  (Simplified -> Taiwan Traditional with Taiwan phrasing: 软件 -> 軟體);
- FineWeb-2 ``cmn_Hani`` (ODC-By 1.0): web text is kept only when it is
  already Traditional (``zh_variant == "tw"``). It is not converted:
  converted mainland web text is not Taiwan text;
- data.gov.tw (OGDL-Taiwan-1.0): local JSONL with a ``provider`` field.
  OGDL requires naming the providing agency, so a record without one is
  dropped, never kept unattributed.

This module is CPU-only and needs only numpy; OpenCC (``pip install
opencc``) is imported only when a source converts. ``Deduper`` drops
exact duplicates (sha256 of NFKC-normalised, whitespace-free text) and
near duplicates (MinHash over character 5-grams with LSH banding, then
an estimated-Jaccard check), so the same input order gives the same
output on every machine.
"""
import hashlib
import re
import unicodedata
import zlib

import numpy as np

from .data_quality import heuristic_check, zh_profile, zh_variant

_WS = re.compile(r"\s+")
_PRIME = (1 << 31) - 1    # Mersenne prime for MinHash

SOURCES = {
    "zhwiki": {
        "hf": "wikimedia/wikipedia", "config": "20231101.zh",
        "license": "CC-BY-SA-4.0", "convert": "s2twp",
        "variants": ("tw", "mixed"),
        "attribution": ("Wikipedia contributors, zh.wikipedia.org "
                        "(CC BY-SA 4.0); converted to Taiwan Traditional "
                        "with OpenCC s2twp"),
    },
    "fineweb2": {
        "hf": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani",
        "license": "ODC-BY-1.0", "convert": None, "variants": ("tw",),
        "attribution": ("FineWeb-2 (Hugging Face, ODC-By 1.0); original "
                        "page copyright remains with its authors"),
    },
    "datagov": {
        "hf": None, "config": None,
        "license": "OGDL-Taiwan-1.0", "convert": None,
        "variants": ("tw", "mixed"),
        "attribution": "提供機關：{provider}（政府資料開放授權條款第1版）",
    },
}


def normalize(text):
    """NFKC with all whitespace removed: the form both dedup keys use."""
    return _WS.sub("", unicodedata.normalize("NFKC", text))


def exact_key(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text, n=5):
    """crc32 ids of the character n-grams of ``normalize(text)``."""
    t = normalize(text)
    if len(t) <= n:
        return {zlib.crc32(t.encode("utf-8"))} if t else set()
    return {zlib.crc32(t[i:i + n].encode("utf-8"))
            for i in range(len(t) - n + 1)}


class MinHash:
    """``num_perm`` hash functions h(x) = (a x + b) mod (2^31 - 1), with
    a, b drawn from a seeded generator, so signatures are identical
    across machines. x is a crc32 id reduced mod p, so a * x < 2^62
    never overflows uint64."""

    def __init__(self, num_perm=128, seed=0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _PRIME, num_perm, dtype=np.uint64)[:, None]
        self.b = rng.integers(0, _PRIME, num_perm, dtype=np.uint64)[:, None]

    def signature(self, ids):
        if not ids:
            return np.full(len(self.a), _PRIME, dtype=np.uint32)
        x = np.fromiter(sorted(ids), dtype=np.uint64, count=len(ids))
        x = x % np.uint64(_PRIME)
        sig = np.full(len(self.a), _PRIME, dtype=np.uint64)
        for i in range(0, len(x), 4096):    # bound memory on long pages
            h = (self.a * x[None, i:i + 4096] + self.b) % np.uint64(_PRIME)
            sig = np.minimum(sig, h.min(axis=1))
        return sig.astype(np.uint32)


class Deduper:
    """Exact + MinHash-LSH near-duplicate filter over a document stream.

    ``bands`` x ``rows`` must equal ``num_perm``. A document is a near
    duplicate when it shares a band bucket with an earlier kept document
    and their estimated Jaccard similarity is at least ``threshold``.
    The first occurrence is always the one kept."""

    def __init__(self, num_perm=128, bands=16, threshold=0.8, ngram=5,
                 seed=0):
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        self.mh = MinHash(num_perm, seed)
        self.bands, self.rows = bands, num_perm // bands
        self.threshold, self.ngram = threshold, ngram
        self.seen = set()
        self.buckets = [dict() for _ in range(bands)]
        self.sigs = []

    def check(self, text):
        """'exact_dup', 'near_dup' or None (kept and remembered)."""
        key = exact_key(text)
        if key in self.seen:
            return "exact_dup"
        sig = self.mh.signature(shingles(text, self.ngram))
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes()
                for i in range(self.bands)]
        cands = {self.buckets[i][k] for i, k in enumerate(keys)
                 if k in self.buckets[i]}
        for c in sorted(cands):
            if float(np.mean(self.sigs[c] == sig)) >= self.threshold:
                return "near_dup"
        self.seen.add(key)
        idx = len(self.sigs)
        self.sigs.append(sig)
        for i, k in enumerate(keys):
            self.buckets[i].setdefault(k, idx)
        return None


def opencc_converter(config="s2twp"):
    try:
        import opencc
    except ImportError as e:
        raise ImportError("zh Wikipedia needs OpenCC: pip install opencc"
                          ) from e
    return opencc.OpenCC(config).convert


def new_stats():
    return {"in": 0, "kept": 0, "dropped": {}, "chars": 0, "units": 0,
            "tokens": 0, "cjk": 0, "trad_only": 0, "simp_only": 0}


def trad_share(stats):
    """Traditional-only over variant-specific CJK characters (None if the
    kept text has none)."""
    m = stats["trad_only"] + stats["simp_only"]
    return stats["trad_only"] / m if m else None


def process(rec, source, spec, deduper, stats, *, converter=None,
            count_tokens=None, rules=None):
    """One raw record -> an output record, or None (the reason is counted
    in ``stats['dropped']``). Order: attribution, heuristics, conversion,
    zh variant, dedup."""
    stats["in"] += 1

    def drop(reason):
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

    text = rec.get("text") or ""
    attribution = spec["attribution"]
    if "{provider}" in attribution:
        provider = (rec.get("provider") or "").strip()
        if not provider:
            drop("no_provider")
            return None
        attribution = attribution.format(provider=provider)
    ok, reasons = heuristic_check(text, rules)
    if not ok:
        drop("heuristic:" + reasons[0])
        return None
    if spec.get("convert"):
        if converter is None:
            raise ValueError(f"{source}: needs a {spec['convert']} converter")
        text = converter(text)
    if zh_variant(text) not in spec["variants"]:
        drop("zh_variant")
        return None
    dup = deduper.check(text)
    if dup:
        drop(dup)
        return None
    p = zh_profile(text)
    stats["kept"] += 1
    stats["chars"] += len(text)
    stats["units"] += len(re.findall(r"[A-Za-z0-9]+", text)) + p["cjk"]
    for k in ("cjk", "trad_only", "simp_only"):
        stats[k] += p[k]
    if count_tokens is not None:
        stats["tokens"] += count_tokens(text)
    out = {"id": f"{source}-{stats['kept'] - 1:08d}", "source": source,
           "text": text, "license": spec["license"],
           "attribution": attribution}
    for k in ("url", "title"):
        if rec.get(k):
            out[k] = rec[k]
    return out
