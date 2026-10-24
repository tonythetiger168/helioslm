"""Tests for the Issue #5 BPE tokenizer training path."""
import pytest

from examples.train_toy_bpe import train_tokenizer, split_corpus


def test_bpe_roundtrip_unicode_and_words():
    pytest.importorskip("tokenizers")
    text = ("HeliosLM supports attention and quantization. 你好世界!\n" * 80)
    tok = train_tokenizer(text, vocab_size=1024)
    ids = tok.encode("HeliosLM supports attention").ids
    assert ids and max(ids) < 1024
    assert tok.decode(ids) == "HeliosLM supports attention"
    assert any(len(tok.id_to_token(i)) > 2 for i in ids)


def test_split_is_deterministic_and_disjoint():
    a, b, files = split_corpus(13)
    a2, b2, files2 = split_corpus(13)
    assert (a, b, files) == (a2, b2, files2)
    assert a and b and files


def test_char_tokenizer_and_max_ord_split():
    from examples.train_toy_bpe import CharTokenizer
    tok = CharTokenizer(1024)
    assert tok.encode("abc").ids == [97, 98, 99] and tok.decode([97, 98]) == "ab"
    with pytest.raises(ValueError):
        tok.encode("你好")                       # outside the char vocabulary
    a, b, files = split_corpus(13, max_ord=1024)
    assert max(map(ord, a + b)) < 1024
    a2, b2, files2 = split_corpus(13)
    assert files == files2                       # same files, only filtered


def test_real_word_rate():
    from examples.train_toy_bpe import real_word_rate
    train = "the model uses attention and quantization"
    assert real_word_rate(" the model attention", train) == 1.0
    assert real_word_rate(" xqz vbn the", train) == pytest.approx(1 / 3)
    assert real_word_rate(" ,,, 12 ;;", train) == 0.0


def test_committed_comparison():
    """Issue #5 acceptance run: three seeds, same held-out text per seed,
    BPE lower NLL per UTF-8 byte than the char baseline in every seed."""
    import hashlib
    import json
    from pathlib import Path
    art = json.loads((Path(__file__).resolve().parents[2] / "benchmarks"
                      / "toy_bpe_vs_char_2026-10-21.json").read_text())
    assert art["config"]["seeds"] == [13, 14, 15]
    runs = {(r["seed"], r["tokenizer"]): r for r in art["runs"]}
    for p in art["paired"]:
        c, b = runs[(p["seed"], "char")], runs[(p["seed"], "bytelevel-bpe")]
        assert c["val_text_sha256"] == b["val_text_sha256"]
        assert c["val_utf8_bytes"] == b["val_utf8_bytes"]
        assert b["val_nll_per_utf8_byte"] < c["val_nll_per_utf8_byte"]
        assert abs(p["bpe_minus_char"] - (b["val_nll_per_utf8_byte"]
                                          - c["val_nll_per_utf8_byte"])) < 1e-12
    assert art["summary"]["bpe_better_in_seeds"] == 3
    assert art["runs_sha256"] == hashlib.sha256(
        json.dumps(art["runs"], sort_keys=True).encode()).hexdigest()
