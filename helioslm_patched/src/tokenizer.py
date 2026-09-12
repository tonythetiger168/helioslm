"""Kimi K3+ Tokenizer - SentencePiece/Tiktoken wrapper"""
import os
import json
from typing import List, Optional, Dict
import torch


class KimiTokenizer:
    """
    Unified tokenizer supporting SentencePiece and Tiktoken backends.
    Features BPE tokenization, special tokens, chat template, batch encoding.
    """

    SPECIAL_TOKENS = {
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|pad|>": 0,
        "<|unk|>": 3,
        "<|image|>": 4,
        "<|audio|>": 5,
        "<|system|>": 6,
        "<|user|>": 7,
        "<|assistant|>": 8,
        "<|tool|>": 9,
        "<|endofchunk|>": 10,
        "<|refusal|>": 11,
    }

    def __init__(self, vocab_size=160000, model_path=None, backend="sentencepiece"):
        self.vocab_size = vocab_size
        self.backend = backend
        self.model_path = model_path
        self.special_tokens = self.SPECIAL_TOKENS.copy()
        self._inverse_special = {v: k for k, v in self.special_tokens.items()}
        self._init_backend()

    def _init_backend(self):
        self.sp = None
        self.enc = None
        if self.backend == "sentencepiece":
            try:
                import sentencepiece as spm
                if self.model_path and os.path.exists(self.model_path):
                    self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
            except ImportError:
                pass
        elif self.backend == "tiktoken":
            try:
                import tiktoken
                self.enc = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass

    def encode(self, text, add_special_tokens=True, max_length=None):
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens["<|bos|>"])
        if self.sp is not None:
            tokens.extend(self.sp.encode(text, out_type=int))
        elif self.enc is not None:
            tokens.extend(self.enc.encode(text))
        else:
            tokens.extend([ord(c) % self.vocab_size for c in text])
        if add_special_tokens:
            tokens.append(self.special_tokens["<|eos|>"])
        if max_length is not None:
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                tokens.extend([self.special_tokens["<|pad|>"]] * (max_length - len(tokens)))
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in self.special_tokens.values()]
        if self.sp is not None:
            return self.sp.decode(token_ids)
        elif self.enc is not None:
            return self.enc.decode(token_ids)
        return "".join(chr(t % 128) for t in token_ids if t < 128)

    def batch_encode(self, texts, max_length=None, padding=True):
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        if padding and max_length is None:
            max_length = max(len(t) for t in encoded)
        if padding:
            for i in range(len(encoded)):
                pad_len = max_length - len(encoded[i])
                if pad_len > 0:
                    encoded[i] = encoded[i] + [self.special_tokens["<|pad|>"]] * pad_len
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != self.special_tokens["<|pad|>"]).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(self, messages, add_generation_prompt=True):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append("<|system|>\n" + content)
            elif role == "user":
                parts.append("<|user|>\n" + content)
            elif role == "assistant":
                parts.append("<|assistant|>\n" + content)
            elif role == "tool":
                parts.append("<|tool|>\n" + content)
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def encode_chat(self, messages, max_length=None):
        prompt = self.apply_chat_template(messages)
        return self.encode(prompt, max_length=max_length)

    def convert_tokens_to_ids(self, tokens):
        unk = self.special_tokens["<|unk|>"]
        return [self.special_tokens.get(t, unk) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._inverse_special.get(i, "<token_{}>".format(i)) for i in ids]

    def save(self, path):
        config = {
            "vocab_size": self.vocab_size,
            "backend": self.backend,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        tok = cls(vocab_size=config["vocab_size"], backend=config["backend"])
        tok.special_tokens = config["special_tokens"]
        tok._inverse_special = {v: k for k, v in tok.special_tokens.items()}
        return tok
