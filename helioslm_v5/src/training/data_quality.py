"""Pretraining data quality: heuristics, zh variant, a hashed classifier
(v5.115, LB3 of the 2026-10-23 leaderboard review).

The 2026 small-model leaders win on data rather than architecture:
MiniCPM5 trains on Ultra-FineWeb (a classifier-filtered FineWeb), K2
Horizon on ~20T curated tokens. ``pilot_prepare_data.py`` only streamed
and decontaminated. This module adds the three filters a #40 corpus
needs, all CPU-only and dependency-free (numpy):

- ``heuristic_check``: Gopher / C4 style document rules (length,
  duplicate lines, symbol and ellipsis ratios, letter/CJK share), with
  CJK-aware length so Chinese text is not judged by word counts;
- ``zh_profile``: Traditional vs Simplified without opencc. A character
  that encodes in Big5 but not GB2312 is Traditional-only (龜, 這, 國);
  one that encodes in GB2312 but not Big5 is Simplified-only (龟, 这,
  国); shared characters (山, 的) count for neither. ``zh_variant``
  labels a document tw / cn / mixed / none;
- ``HashedNgramClassifier``: the fastText-style quality classifier of
  FineWeb-Edu / Ultra-FineWeb, as logistic regression over hashed
  features (CJK character 1-2-grams, lower-cased word 1-2-grams; crc32
  hashing, so scores are identical across machines and Python hash
  seeds). Train it on positives (curated text, teacher answers) against
  random web text, then keep documents scoring above a threshold.
"""
import json
import re
import zlib
from pathlib import Path

import numpy as np

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"[A-Za-z0-9]+")
_TOKEN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+|[A-Za-z0-9]+")

DEFAULT_RULES = {
    "min_units": 100,          # words + CJK characters
    "max_units": 200_000,
    "max_dup_line_frac": 0.3,
    "max_symbol_ratio": 0.1,   # '#' and ellipses per unit
    "max_ellipsis_line_frac": 0.3,
    "min_text_char_frac": 0.5,  # letters/digits/CJK over non-space chars
}


def _encodes(ch, codec):
    try:
        ch.encode(codec)
        return True
    except UnicodeEncodeError:
        return False


def zh_profile(text):
    """CJK character counts: total, Traditional-only, Simplified-only."""
    cjk = trad = simp = 0
    for ch in _CJK.findall(text):
        cjk += 1
        b5, gb = _encodes(ch, "big5"), _encodes(ch, "gb2312")
        if b5 and not gb:
            trad += 1
        elif gb and not b5:
            simp += 1
    return {"cjk": cjk, "trad_only": trad, "simp_only": simp}


def zh_variant(text, min_cjk=20, min_share=0.8):
    """'tw', 'cn', 'mixed' or 'none' (fewer than ``min_cjk`` CJK chars).

    Decided by the variant-specific characters only: 'tw' when at least
    ``min_share`` of them are Traditional-only, 'cn' likewise, else
    'mixed'. A CJK text with no variant-specific character at all is
    'mixed' (it reads the same in both)."""
    p = zh_profile(text)
    if p["cjk"] < min_cjk:
        return "none"
    marked = p["trad_only"] + p["simp_only"]
    if marked == 0:
        return "mixed"
    if p["trad_only"] / marked >= min_share:
        return "tw"
    if p["simp_only"] / marked >= min_share:
        return "cn"
    return "mixed"


def heuristic_check(text, rules=None):
    """(keep, reasons): reasons lists every rule the document fails."""
    r = dict(DEFAULT_RULES, **(rules or {}))
    units = len(_WORD.findall(text)) + len(_CJK.findall(text))
    reasons = []
    if units < r["min_units"]:
        reasons.append("too_short")
    if units > r["max_units"]:
        reasons.append("too_long")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        dup = len(lines) - len(set(lines))
        if dup / len(lines) > r["max_dup_line_frac"]:
            reasons.append("duplicate_lines")
        ell = sum(ln.endswith(("...", "…")) for ln in lines)
        if ell / len(lines) > r["max_ellipsis_line_frac"]:
            reasons.append("ellipsis_lines")
    sym = text.count("#") + text.count("...") + text.count("…")
    if units and sym / units > r["max_symbol_ratio"]:
        reasons.append("symbols")
    nonspace = sum(not c.isspace() for c in text)
    textual = sum(map(len, _WORD.findall(text))) + len(_CJK.findall(text))
    if nonspace and textual / nonspace < r["min_text_char_frac"]:
        reasons.append("non_text")
    return not reasons, reasons


def features(text, dim):
    """Hashed feature ids (with repeats) for one document."""
    ids = []
    for run in _TOKEN.findall(text):
        if _CJK.match(run):
            grams = list(run) + [run[i:i + 2] for i in range(len(run) - 1)]
        else:
            grams = [run.lower()]
        ids += [zlib.crc32(g.encode("utf-8")) % dim for g in grams]
    words = [w.lower() for w in _WORD.findall(text)]
    ids += [zlib.crc32(("%s %s" % p).encode("utf-8")) % dim
            for p in zip(words, words[1:])]
    return ids


class HashedNgramClassifier:
    """Binary logistic regression over hashed n-gram counts (log-scaled,
    L2-normalised per document), trained by per-example SGD."""

    def __init__(self, dim=1 << 18):
        self.dim = dim
        self.w = np.zeros(dim, dtype=np.float64)
        self.b = 0.0

    def _vec(self, text):
        ids = features(text, self.dim)
        if not ids:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        u, c = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        v = np.log1p(c.astype(np.float64))
        return u, v / np.linalg.norm(v)

    def fit(self, positives, negatives, epochs=5, lr=0.5, l2=1e-6, seed=0):
        if not positives or not negatives:
            raise ValueError("need both positive and negative examples")
        data = [(self._vec(t), 1.0) for t in positives] + \
               [(self._vec(t), 0.0) for t in negatives]
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(data)):
                (u, v), y = data[i]
                z = float(self.w[u] @ v) + self.b
                p = 1.0 / (1.0 + np.exp(-z))
                g = p - y
                self.w[u] -= lr * (g * v + l2 * self.w[u])
                self.b -= lr * g
        return self

    def score(self, text):
        """P(high quality) for one document."""
        u, v = self._vec(text)
        z = float(self.w[u] @ v) + self.b if len(u) else self.b
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), w=self.w,
                            b=np.array([self.b]))
        path.with_suffix(".json").write_text(json.dumps(
            {"kind": "hashed_ngram_logreg", "dim": self.dim,
             "features": "cjk char 1-2gram + word 1-2gram, crc32"}) + "\n")
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        clf = cls(meta["dim"])
        z = np.load(path.with_suffix(".npz"))
        clf.w, clf.b = z["w"], float(z["b"][0])
        return clf


def filter_docs(docs, *, heuristics=True, rules=None, variants=None,
                classifier=None, threshold=0.5):
    """Apply the three filters in order; returns (kept, report).

    ``heuristics``: run ``heuristic_check`` with ``rules`` (overrides of
    DEFAULT_RULES); False skips it.
    ``variants``: a set like {"tw", "mixed", "none"} of zh_variant labels
    to keep (None keeps all). ``classifier`` + ``threshold``: drop
    documents scoring below it. The report counts drops per reason."""
    kept, report = [], {"in": 0, "kept": 0, "dropped": {}}

    def drop(reason):
        report["dropped"][reason] = report["dropped"].get(reason, 0) + 1

    for d in docs:
        report["in"] += 1
        ok, reasons = heuristic_check(d, rules) if heuristics else (True, [])
        if not ok:
            drop("heuristic:" + reasons[0])
            continue
        if variants is not None and zh_variant(d) not in variants:
            drop("zh_variant")
            continue
        if classifier is not None and classifier.score(d) < threshold:
            drop("quality")
            continue
        kept.append(d)
    report["kept"] = len(kept)
    return kept, report
