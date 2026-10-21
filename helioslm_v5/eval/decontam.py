"""Training / evaluation overlap check (plan E2, docs/GPU_AND_EVAL_PLAN_2026-10.md).

A benchmark score means little if the benchmark's text was in the
training data. This module finds that overlap with the n-gram test the
GPT-3 and Llama reports use: an evaluation item is contaminated when any
of its n-grams (default n = 13) also occurs in a training document.

- Words are lowercased runs of letters / digits; every CJK character (and
  kana / hangul syllable) is a word of its own, since Chinese and Japanese
  are written without spaces. Punctuation and whitespace are ignored, so
  reformatting does not hide a copy.
- n-grams are stored as 8-byte BLAKE2b digests: stable across processes
  (Python's ``hash`` of a string is not), small enough to index a whole
  benchmark.
- Items shorter than n words are compared on all their words at once (one
  n-gram of their own length), so short questions are still checked; items
  under ``min_words`` (default 8) would match common phrases, so they are
  listed as unchecked instead.

``EvalIndex(items)`` indexes the evaluation side once;
``index.scan(doc)`` returns the ids of the items a training document hits;
``index.report(docs)`` scans a corpus and summarises it;
``decontaminate(docs, index)`` drops the training documents that hit.
"""
import hashlib
import re
from collections import defaultdict

_CJK = ("぀-ヿ"            # hiragana, katakana
        "㐀-䶿一-鿿豈-﫿"   # CJK ideographs
        "가-힯")           # hangul syllables
_WORD = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+")

DEFAULT_N = 13
DEFAULT_MIN_WORDS = 8


def words(text):
    return [w.lower() for w in _WORD.findall(text)]


def _digest(gram):
    return hashlib.blake2b("\x1f".join(gram).encode("utf-8"),
                           digest_size=8).digest()


def ngram_digests(text, n=DEFAULT_N, short_ok=False):
    """Digests of the text's word n-grams. With ``short_ok`` a text shorter
    than n words yields one gram of all its words (none if it is empty)."""
    w = words(text)
    if len(w) < n:
        return {_digest(w)} if (short_ok and w) else set()
    return {_digest(w[i:i + n]) for i in range(len(w) - n + 1)}


class EvalIndex:
    """n-gram index of evaluation items ({item_id: text})."""

    def __init__(self, items, n=DEFAULT_N, min_words=DEFAULT_MIN_WORDS):
        if n < 1 or min_words < 1:
            raise ValueError(f"n and min_words must be >= 1, got {n}, {min_words}")
        self.n = n
        self.items = dict(items)
        self._owners = defaultdict(set)
        self._short = {}                   # items shorter than n words
        self.unchecked = []                # items under min_words
        for item_id, text in self.items.items():
            w = words(text)
            if len(w) < n:
                if len(w) >= min_words:
                    self._short[item_id] = w
                else:
                    self.unchecked.append(item_id)
                continue
            for d in ngram_digests(text, n):
                self._owners[d].add(item_id)
        # short items: look for their whole word sequence as a gram of its
        # own length inside training documents
        self._short_by_len = defaultdict(dict)
        for item_id, w in self._short.items():
            self._short_by_len[len(w)].setdefault(_digest(w), set()).add(item_id)

    def scan(self, doc):
        """Ids of the evaluation items this training document overlaps."""
        hit = set()
        w = words(doc)
        for i in range(len(w) - self.n + 1):
            owners = self._owners.get(_digest(w[i:i + self.n]))
            if owners:
                hit |= owners
        for k, table in self._short_by_len.items():
            for i in range(len(w) - k + 1):
                owners = table.get(_digest(w[i:i + k]))
                if owners:
                    hit |= owners
        return hit

    def report(self, docs):
        """Scan a training corpus: which items are contaminated, by how
        many documents, and which documents overlap anything."""
        by_item = defaultdict(int)
        bad_docs = []
        n_docs = 0
        for j, doc in enumerate(docs):
            n_docs += 1
            hit = self.scan(doc)
            if hit:
                bad_docs.append(j)
                for item_id in hit:
                    by_item[item_id] += 1
        total = len(self.items)
        return {"n": self.n, "eval_items": total,
                "unchecked_items": sorted(self.unchecked),
                "contaminated_items": len(by_item),
                "contaminated_fraction": len(by_item) / total if total else 0.0,
                "training_docs": n_docs,
                "overlapping_docs": len(bad_docs),
                "overlapping_doc_indices": bad_docs,
                "items": {k: by_item[k] for k in sorted(by_item)}}


def decontaminate(docs, index):
    """Training documents with no evaluation overlap, plus the report."""
    docs = list(docs)
    rep = index.report(docs)
    drop = set(rep["overlapping_doc_indices"])
    return [d for j, d in enumerate(docs) if j not in drop], rep
