"""Traditional-Chinese (zh-TW) tokenizer coverage report (v5.75, roadmap TW4).

How many tokens does each tokenizer this repo can use spend per character
of Traditional Chinese, and how many characters can it represent at all?

Text: the repo's own Traditional-Chinese documents (no download needed;
Hugging Face is not reachable from the CI / dev containers). Two documents
are HELD OUT for evaluation and removed from every training corpus; the
rest of the repo is the training side, so the BPE numbers measure
generalisation to unseen zh-TW text, not memorisation.

Tokenizers:
  char_toy       the toy pipeline's char vocabulary (token id = ord(ch),
                 kept only when ord < vocab_size): CJK falls outside it;
  utf8_bytes     raw UTF-8 bytes (every byte-level BPE's fallback floor);
  bpe_code_only  byte-level BPE trained on the repo corpus with every CJK
                 character stripped (an English / code tokenizer);
  bpe_repo       the same BPE trained on the repo corpus as is (minus the
                 held-out documents), i.e. with the zh-TW the repo has.

The BPEs use the ``tokenizers`` library exactly as examples/train_toy_bpe.py
(Issue #5) does: byte-level pre-tokenizer, BpeTrainer, min_frequency 2.
"""
import re
from pathlib import Path

HELD_OUT = ("docs/TW_DATASETS_2026Q4.md", "docs/V5.4_FIXES_REPORT.md")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")


def is_cjk(ch):
    return bool(_CJK.fullmatch(ch))


def repo_root():
    return Path(__file__).resolve().parents[2]


def held_out_texts(root=None):
    root = Path(root or repo_root())
    return {p: (root / p).read_text(encoding="utf-8") for p in HELD_OUT}


def eval_split(root=None):
    """(zh_lines, en_lines) from the held-out documents: lines with at
    least one CJK character, and lines with letters but no CJK."""
    zh, en = [], []
    for text in held_out_texts(root).values():
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if _CJK.search(line):
                zh.append(line)
            elif re.search(r"[A-Za-z]{3}", line):
                en.append(line)
    return zh, en


def training_corpus(root=None, strip_cjk=False):
    """The repo BPE corpus (tokenizer.bpe.build_corpus) with the held-out
    documents removed; optionally with every CJK character stripped."""
    from evastar.src.tokenizer.bpe import build_corpus
    corpus = build_corpus(root)
    for text in held_out_texts(root).values():
        corpus = corpus.replace(text[:20000], "")   # build_corpus keeps 20k
    assert_no_leak(corpus, held_out_texts(root).values())
    if strip_cjk:
        corpus = _CJK.sub("", corpus)
    return corpus


def assert_no_leak(corpus, held_texts):
    """Raise if any substantial zh line of a held-out document is still in
    the training corpus."""
    for text in held_texts:
        for line in text.splitlines():
            if len(line) > 20 and _CJK.search(line) and line in corpus:
                raise RuntimeError(
                    f"held-out text leaked into the corpus: {line[:40]!r}")


def big5_common_chars():
    """The 5401 frequently used Traditional characters (常用字): Big5 level 1,
    0xA440..0xC67E, decoded with Python's built-in big5 codec (no download).
    Stroke order, as Big5 stores them."""
    chars = []
    for hi in range(0xA4, 0xC7):
        for lo in list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF)):
            if (hi, lo) > (0xC6, 0x7E):
                break
            try:
                chars.append(bytes([hi, lo]).decode("big5"))
            except UnicodeDecodeError:
                pass
    return chars


def gb2312_common_chars():
    """The 3755 level-1 common Simplified characters (一级汉字): GB2312
    0xB0A1..0xD7F9, decoded with Python's built-in gb2312 codec (no
    download). Pinyin order, as GB2312 stores them. 2434 of them are also
    Big5 common characters (shared forms such as 一, 人, 大)."""
    chars = []
    for hi in range(0xB0, 0xD8):
        for lo in range(0xA1, 0xFF):
            if (hi, lo) > (0xD7, 0xF9):
                break
            try:
                chars.append(bytes([hi, lo]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return chars


SEED_SCRIPTS = ("tw", "cn", "both")


def zh_seed_chars(scripts="tw"):
    """Common characters to seed: "tw" = Big5 (Traditional, 5401), "cn" =
    GB2312 (Simplified, 3755), "both" = their union (6722), in a fixed
    order (Big5 first, then the Simplified-only ones)."""
    if scripts not in SEED_SCRIPTS:
        raise ValueError(f"scripts must be one of {SEED_SCRIPTS}, got {scripts!r}")
    tw = big5_common_chars() if scripts in ("tw", "both") else []
    cn = gb2312_common_chars() if scripts in ("cn", "both") else []
    seen = set(tw)
    return tw + [c for c in cn if c not in seen]


def zh_seed_text(scripts="tw", repeats=2):
    """Seed lines for BPE training (v5.80): every common character on its
    own line ``repeats`` times. With repeats >= min_frequency (2) each
    character's byte merges are learned, so it becomes ONE token when the
    vocabulary has room (~32k); multi-character merges from the corpus are
    still learned on top."""
    return ("\n".join(zh_seed_chars(scripts)) + "\n") * repeats


def big5_seed_text(repeats=2):
    """The Traditional-only seed (``zh_seed_text("tw")``)."""
    return zh_seed_text("tw", repeats)


def train_bpe(text, vocab_size):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]"], show_progress=False)
    tok.train_from_iterator(text.splitlines(), trainer=trainer)
    return tok


class CharToy:
    """token id = ord(ch) for ord < vocab_size; anything else is lost."""

    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size

    def count(self, text):
        return sum(1 for ch in text if ord(ch) < self.vocab_size)

    def covers(self, ch):
        return ord(ch) < self.vocab_size

    def loses(self, ch):
        return ord(ch) >= self.vocab_size


class Utf8Bytes:
    def count(self, text):
        return len(text.encode("utf-8"))

    def covers(self, ch):
        return len(ch.encode("utf-8")) == 1


class Bpe:
    def __init__(self, tok):
        self.tok = tok
        self.vocab = tok.get_vocab()
        from tokenizers import pre_tokenizers
        self._pre = pre_tokenizers.ByteLevel(add_prefix_space=False)

    def count(self, text):
        return len(self.tok.encode(text).ids)

    def covers(self, ch):
        """True when the character is ONE vocabulary token (not split
        into its UTF-8 bytes)."""
        mapped = self._pre.pre_tokenize_str(ch)[0][0]
        return mapped in self.vocab


def measure(tokenizer, zh_lines, en_lines):
    zh_text = "\n".join(zh_lines)
    cjk_runs = _CJK_RUN.findall(zh_text)
    cjk_chars = sum(len(r) for r in cjk_runs)
    distinct = sorted({ch for r in cjk_runs for ch in r})
    en_text = "\n".join(en_lines)
    return {
        "tokens_per_char_zh_lines": round(tokenizer.count(zh_text) / len(zh_text), 4),
        "tokens_per_cjk_char": round(
            sum(tokenizer.count(r) for r in cjk_runs) / cjk_chars, 4),
        "cjk_single_token_coverage": round(
            sum(tokenizer.covers(ch) for ch in distinct) / len(distinct), 4),
        # characters the tokenizer cannot represent at all (dropped)
        "cjk_lost_fraction": round(
            sum(getattr(tokenizer, "loses", lambda c: False)(ch)
                for r in cjk_runs for ch in r) / cjk_chars, 4),
        "tokens_per_char_en_lines": round(tokenizer.count(en_text) / len(en_text), 4),
    }


def eval_stats(zh_lines, en_lines):
    zh_text = "\n".join(zh_lines)
    cjk = _CJK_RUN.findall(zh_text)
    return {"zh_lines": len(zh_lines), "zh_chars": len(zh_text),
            "cjk_chars": sum(len(r) for r in cjk),
            "distinct_cjk": len({ch for r in cjk for ch in r}),
            "en_lines": len(en_lines), "en_chars": len("\n".join(en_lines))}
