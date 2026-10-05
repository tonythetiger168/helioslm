"""HeliosLM v5.33 P0 — byte-level BPE tokenizer (mid-size foundation).

Why this exists (recorded, not hidden): the char-level vocab (ord<1024)
is the structural ceiling behind the tool-channel mode-choice 0/20 and
parse-rate limits — every JSON character is one autoregressive decision.
The v5.33 "mid" preset (~150-300M) upgrades to BPE; this module is its
foundation and stays additive: the char pipeline is untouched.

Design:
- Byte-level (GPT-2 style byte map): any byte sequence encodes/decodes,
  OOB bytes fall back to per-byte tokens — robustness by construction.
- Trained on the repo's OWN corpus (source + tool/chat protocol strings +
  env task texts). Deliberately domain-matched, no external text. Recorded
  deviation: a production tokenizer would train on broad corpora; ours is
  a research vehicle and the corpus is the operating domain.
- Pure-python trainer: word-frequency dict + greedy merge by pair count
  with per-word pair indexing. Vocab = 3 specials + 256 byte tokens +
  merges. Deterministic (stable iteration order).
- Interface: HeliosBPE.encode/decode/save/load; ids are stable across
  save/load (roundtrip-tested, T31).
"""
import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = (PAD, BOS, EOS)


def _byte_maps():
    bs = list(range(ord("!"), ord("~") + 1)) \
        + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    return b2u, {v: k for k, v in b2u.items()}


def _pretokenize(text):
    """Split into raw byte-words: spaces attach to the FOLLOWING word
    (byte-level convention, same as the Qwen3 reference runner). Shared
    by train and encode so vocab and encoding can never drift. Pure
    separators ('  ') become standalone space bytes and roundtrip."""
    out = []
    for idx, word in enumerate(text.split(" ")):
        if idx == 0:
            raw = word.encode("utf-8")
        else:
            raw = b" " + word.encode("utf-8")
        if raw:
            out.append(raw)
    return out


def build_corpus(repo_root=None):
    """The training corpus: helioslm_v5 python sources + protocol strings
    + env task samples + chat transcript shapes. Domain-matched by
    design (see module docstring)."""
    import random
    import sys
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    parts = []
    # SORTED file lists: rglob order is filesystem-dependent and NOT
    # stable across processes on some systems (found 2026-09-28 -- a
    # tokenizer retrained in an eval process diverged from the training
    # process one, producing confident garbage on a healthy checkpoint)
    for p in sorted(Path(repo_root).rglob("*.py")):
        try:
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    for p in sorted(Path(repo_root).rglob("*.md")):
        try:
            parts.append(p.read_text(encoding="utf-8", errors="ignore")[:20000])
        except OSError:
            pass
    sys.path.insert(0, str(Path(repo_root) / "helioslm_v5" / "agent"))
    try:
        from envs import make_envs, make_long_envs
        rng = random.Random(7)
        for env in make_envs() + make_long_envs():
            for _ in range(300):
                parts.append(env.sample(rng).text)
    except Exception as e:  # corpus still usable without envs
        parts.append(f"corpus-env-skip {e!r}")
    protocol = [
        '@@tool@@{"calls":[{"name":"calc","args":{"expr":"1 + 1"}}]}@@end@@',
        '@@tool@@{"calls":[{"name":"finish","args":{"answer":"42"}}]}@@end@@',
        "##user## Compute the value of: 3 * 4 + 5",
        "##assistant## @@tool@@{\"calls\":[{\"name\":\"calc\",\"args\":{\"expr\":\"3 * 4 + 5\"}}]}@@end@@",
        "##tool## 17", "The magic word is kiwi. What is the magic word?",
        "PARSE_ERROR: missing or misplaced tool markers",
        "TOOL_ERROR: calc: division by zero",
    ] * 50
    parts.extend(protocol)
    return "\n".join(parts)


def train_bpe(texts, vocab_size=16000, seed=0):
    """Greedy byte-pair merge training. Returns (vocab, merges).

    vocab: {token_str: id} with ids 0..2 = specials, 3..258 = byte
    tokens, then merged tokens. merges: [(a_str, b_str), ...] in rank
    order. Deterministic given the same input (sorted iteration)."""
    b2u, _ = _byte_maps()
    word_freq = Counter()
    for text in texts:
        for raw in _pretokenize(text):
            syms = tuple(b2u[b] for b in raw)
            word_freq[syms] += 1

    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    merges = []

    def pair_counts():
        pc = Counter()
        for word, f in word_freq.items():
            for i in range(len(word) - 1):
                pc[(word[i], word[i + 1])] += f
        return pc

    while len(vocab) < vocab_size:
        pc = pair_counts()
        if not pc:
            break
        (a, b), _ = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
        merged = a + b
        if merged in vocab:
            break
        merges.append((a, b))
        vocab[merged] = len(vocab)
        new_freq = Counter()
        for word, f in word_freq.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new_freq[tuple(out)] += f
        word_freq = new_freq
    return vocab, merges


class HeliosBPE:
    """Trained byte-level BPE tokenizer. Deterministic encode; decode
    via inverse byte map; save/load roundtrips exactly (T31)."""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.id2tok = {i: s for s, i in vocab.items()}
        _, self.u2b = _byte_maps()
        self.rank = {p: i for i, p in enumerate(merges)}

    @classmethod
    def train(cls, texts, vocab_size=16000):
        vocab, merges = train_bpe(texts, vocab_size)
        return cls(vocab, merges)

    # -- io ---------------------------------------------------------------
    def save(self, path):
        d = {"vocab": self.vocab, "merges": list(self.rank.keys())}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])

    # -- encode/decode ------------------------------------------------------
    def encode(self, text, add_bos=False):
        b2u, _ = _byte_maps()
        ids = [self.vocab[BOS]] if add_bos else []
        for raw in _pretokenize(text):
            symbols = [b2u[b] for b in raw]
            while len(symbols) > 1:
                best_rank, best_pair = None, None
                for i in range(len(symbols) - 1):
                    r = self.rank.get((symbols[i], symbols[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_pair = r, (symbols[i], symbols[i + 1])
                if best_pair is None:
                    break
                a, b = best_pair
                out, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a \
                            and symbols[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                symbols = out
            ids.extend(self.vocab[s] for s in symbols)
        return ids

    def decode(self, ids, skip_specials=True):
        buf = bytearray()
        for i in ids:
            tok = self.id2tok.get(int(i))
            if tok is None:
                continue
            if tok in SPECIALS:
                if not skip_specials:
                    buf.extend(tok.encode("utf-8"))
                continue
            for ch in tok:
                b = self.u2b.get(ch)
                buf.append(b) if b is not None else buf.extend(
                    ch.encode("utf-8"))
        return buf.decode("utf-8", errors="replace")
