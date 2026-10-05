"""T31 - v5.33 P0: byte-level BPE tokenizer oracles.

Covers: roundtrip on domain strings (code, protocol blocks, chat
transcripts), space/separator fidelity, OOV byte robustness, special
token stability, save/load exactness, determinism, and corpus builder
sanity. Small vocab keeps the suite fast; the shipped artifact trains
larger (see module docstring).
Run from repo root: python3 helioslm_v5/tests/test_bpe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.tokenizer.bpe import (BOS, EOS, PAD, HeliosBPE,
                                           build_corpus, train_bpe)

DOMAIN_STRINGS = [
    '@@tool@@{"calls":[{"name":"calc","args":{"expr":"3 * 4 + 5"}}]}@@end@@',
    "##user## Compute the value of: 71 * 14 - 22",
    "##assistant## The magic word is kiwi.",
    "def verify_replay(traj, registry, impls):",
    "PARSE_ERROR: missing or misplaced tool markers",
    "step 0: 17",
    "Apply upper to the string: \"hello world\"",
]


def _mk(vocab_size=1200):
    corpus = build_corpus()[:60000]  # trimmed for test speed
    return HeliosBPE.train([corpus], vocab_size=vocab_size)


def test_roundtrip_domain_strings():
    tok = _mk()
    for s in DOMAIN_STRINGS:
        ids = tok.encode(s)
        assert all(0 <= i < len(tok.vocab) for i in ids)
        back = tok.decode(ids)
        assert back == s, f"roundtrip failed: {s!r} -> {back!r}"
    print("PASS test_roundtrip_domain_strings")


def test_separator_and_space_fidelity():
    tok = _mk()
    for s in ["a  b", " leading", "trailing ", "x   y",
              "line1\nline2", "tab\there"]:
        assert tok.decode(tok.encode(s)) == s, repr(s)
    print("PASS test_separator_and_space_fidelity")


def test_oov_bytes_robust():
    tok = _mk()
    # characters unseen in any merge still encode via byte tokens and
    # decode exactly (byte-level by construction)
    s = "日本語 emoji \U0001f600 snowman ☃"
    assert tok.decode(tok.encode(s)) == s
    print("PASS test_oov_bytes_robust")


def test_specials_and_bos():
    tok = _mk()
    assert tok.vocab[PAD] == 0 and tok.vocab[BOS] == 1 and tok.vocab[EOS] == 2
    ids = tok.encode("hello", add_bos=True)
    assert ids[0] == tok.vocab[BOS]
    assert tok.decode(ids) == "hello", "specials skipped in decode"
    assert "<bos>hello" in tok.decode(ids, skip_specials=False)
    print("PASS test_specials_and_bos")


def test_save_load_exact():
    import tempfile
    tok = _mk()
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "tok.json")
        tok.save(p)
        tok2 = HeliosBPE.load(p)
        for s in DOMAIN_STRINGS:
            assert tok2.encode(s) == tok.encode(s)
            assert tok2.decode(tok.encode(s)) == s
        assert tok2.vocab == tok.vocab
    print("PASS test_save_load_exact")


def test_determinism_and_vocab_cap():
    corpus = build_corpus()[:40000]
    a = HeliosBPE.train([corpus], vocab_size=900)
    b = HeliosBPE.train([corpus], vocab_size=900)
    assert a.vocab == b.vocab, "training must be deterministic"
    assert len(a.vocab) <= 900
    s = DOMAIN_STRINGS[0]
    assert a.encode(s) == b.encode(s)
    print("PASS test_determinism_and_vocab_cap")


def test_corpus_builder_covers_protocol():
    c = build_corpus()
    assert "@@tool@@" in c and "##user##" in c
    assert "Compute the value of:" in c
    print(f"PASS test_corpus_builder_covers_protocol ({len(c)} chars)")


if __name__ == "__main__":
    test_roundtrip_domain_strings()
    test_separator_and_space_fidelity()
    test_oov_bytes_robust()
    test_specials_and_bos()
    test_save_load_exact()
    test_determinism_and_vocab_cap()
    test_corpus_builder_covers_protocol()
    print("\n7/7 BPE tokenizer tests passed")
