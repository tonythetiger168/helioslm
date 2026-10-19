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
