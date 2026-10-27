"""T133 — byte-level BPE vocabulary in the llama.cpp GGUF export, v5.135.

``export_llamacpp_gguf(..., tokenizer=...)`` writes the pretraining
``tokenizer.json`` as llama.cpp's ``gpt2`` vocabulary. CI has no
llama.cpp, so ``_bpe_from_gguf`` re-encodes text from the GGUF arrays alone
(tokens + merge ranks, applied to the same GPT-2 regex pieces llama.cpp's
``gpt-2`` pre-tokenizer produces) and must give the tokenizer's own ids:
that checks the id order, the merge format and the merge ranks. llama.cpp
itself was run on the same files during development (see the CHANGELOG).
"""
import json

import pytest
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.export.gguf import read_gguf
from helioslm_v5.src.export.llamacpp import (
    TOKEN_CONTROL, TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_UNUSED,
    export_llamacpp_gguf, gpt2_tokenizer_metadata)
from helioslm_v5.src.model_v5 import HeliosLMv5

tokenizers = pytest.importorskip("tokenizers")

TEXTS = [
    "台灣的天氣今天很好，我們去淡水看夕陽吧。數字 12345 和標點「引號」！",
    "def f(x):\n    return x ** 2  # square\n\n\tclass A:  pass\n",
    "I'm sure you'll see they've done it. Naïve café, emoji 😀🎉.",
    "a  b   c    d     trailing   ",
] * 20


def _train(vocab=600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(TEXTS, trainer=trainers.BpeTrainer(
        vocab_size=vocab, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|endoftext|>"]))
    return tok


def _bpe_from_gguf(md, pieces):
    ids = {t: i for i, t in enumerate(md["tokenizer.ggml.tokens"])}
    rank = {tuple(m.split(" ")): r for r, m in enumerate(md["tokenizer.ggml.merges"])}
    out = []
    for piece in pieces:
        syms = list(piece)
        while len(syms) > 1:
            pairs = [(rank.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(syms, syms[1:]))]
            r, k = min(pairs)
            if r == 1 << 30:
                break
            syms[k:k + 2] = [syms[k] + syms[k + 1]]
        out.extend(ids[s] for s in syms)
    return out


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tok = _train()
    torch.manual_seed(0)
    config = HeliosLMv5Config(size="lite", num_hidden_layers=1)
    config.vocab_size = tok.get_vocab_size() + 5          # padded embedding rows
    path = tmp_path_factory.mktemp("tok") / "m.gguf"
    export_llamacpp_gguf(HeliosLMv5(config).eval(), path, config, dtype="f32", tokenizer=tok)
    return tok, config, read_gguf(path).metadata


def test_gguf_vocabulary_reencodes_like_the_tokenizer(exported):
    tok, _, md = exported
    probe = TEXTS[:4] + ["unseen wordz 未見過的字 ✓", "  ", "\n\n", "x" * 40]
    for text in probe:
        pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
        assert _bpe_from_gguf(md, pieces) == tok.encode(text).ids, text


def test_vocabulary_keys(exported):
    tok, config, md = exported
    assert md["tokenizer.ggml.model"] == "gpt2"
    assert md["tokenizer.ggml.pre"] == "gpt-2"
    names, kinds = md["tokenizer.ggml.tokens"], md["tokenizer.ggml.token_type"]
    n = tok.get_vocab_size()
    assert len(names) == len(kinds) == config.vocab_size
    assert names[:n] == [tok.id_to_token(i) for i in range(n)]
    assert names[n:] == [f"[PAD{i}]" for i in range(n, config.vocab_size)]
    assert kinds[n:] == [TOKEN_UNUSED] * 5
    eot, unk = tok.token_to_id("<|endoftext|>"), tok.token_to_id("[UNK]")
    assert kinds[eot] == TOKEN_CONTROL and kinds[unk] == TOKEN_UNKNOWN
    assert kinds[tok.token_to_id("Ġ")] == TOKEN_NORMAL
    assert md["tokenizer.ggml.eos_token_id"] == md["tokenizer.ggml.bos_token_id"] == eot
    assert md["tokenizer.ggml.unknown_token_id"] == unk
    assert md["tokenizer.ggml.add_bos_token"] is False
    merges = json.loads(tok.to_str())["model"]["merges"]
    assert md["tokenizer.ggml.merges"] == [" ".join(m) if not isinstance(m, str) else m
                                           for m in merges]


def test_id_only_file_without_tokenizer(tmp_path):
    torch.manual_seed(0)
    config = HeliosLMv5Config(size="lite", num_hidden_layers=1)
    path = export_llamacpp_gguf(HeliosLMv5(config).eval(), tmp_path / "m.gguf", config, dtype="f32")
    md = read_gguf(path).metadata
    assert md["tokenizer.ggml.model"] == "none" and "tokenizer.ggml.tokens" not in md


@pytest.mark.parametrize("edit,match", [
    (lambda j: j.update(normalizer={"type": "NFC"}), "normalizer"),
    (lambda j: j["pre_tokenizer"].update(add_prefix_space=True), "pre_tokenizer"),
    (lambda j: j.update(pre_tokenizer={"type": "Metaspace"}), "pre_tokenizer"),
    (lambda j: j["model"].update(byte_fallback=True), "byte_fallback"),
    (lambda j: j["model"].update(type="Unigram"), "model type"),
    (lambda j: j.update(post_processor={"type": "TemplateProcessing"}), "post_processor"),
])
def test_tokenizers_llamacpp_would_split_differently_are_refused(edit, match):
    tj = json.loads(_train(400).to_str())
    edit(tj)
    with pytest.raises(ValueError, match=match):
        gpt2_tokenizer_metadata(tj, 1000)


def test_more_ids_than_embedding_rows_is_refused():
    tok = _train(400)
    with pytest.raises(ValueError, match="embedding rows"):
        gpt2_tokenizer_metadata(tok, tok.get_vocab_size() - 1)
