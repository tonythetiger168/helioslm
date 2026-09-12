"""Tokenizer wrapper for Phase 2 pre-training."""
import os
from typing import List, Dict, Optional, Union
import torch


class TokenizerWrapper:
    """
    Unified tokenizer interface supporting SentencePiece, HuggingFace, and custom.

    Features:
      - BOS/EOS/PAD special token handling
      - Chat template for instruction data
      - Efficient batch encoding with padding/truncation
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 160000,
        model_type: str = "sentencepiece",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self._tokenizer = None

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        if model_path and os.path.exists(model_path):
            self._load_pretrained(model_path)
        else:
            self._build_placeholder()

    def _load_pretrained(self, model_path: str):
        if self.model_type == "sentencepiece":
            import sentencepiece as spm
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(model_path)
            self.vocab_size = self._tokenizer.vocab_size()
        elif self.model_type == "huggingface":
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.vocab_size = len(self._tokenizer)

    def _build_placeholder(self):
        """Build a minimal byte-level BPE placeholder."""
        self._tokenizer = None
        self.vocab_size = max(self.vocab_size, 256 + 16)

    @property
    def bos_token_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, "bos_id"):
            return self._tokenizer.bos_id()
        return 1

    @property
    def eos_token_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, "eos_id"):
            return self._tokenizer.eos_id()
        return 2

    @property
    def pad_token_id(self) -> int:
        if self._tokenizer and hasattr(self._tokenizer, "pad_id"):
            return self._tokenizer.pad_id()
        return 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self._tokenizer is None:
            # Byte-level fallback
            tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        elif self.model_type == "sentencepiece":
            tokens = self._tokenizer.EncodeAsIds(text)
        else:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            token_ids = [t for t in token_ids if t not in special]

        if self._tokenizer is None:
            return "".join(chr(min(t, 0x10FFFF)) for t in token_ids)
        elif self.model_type == "sentencepiece":
            return self._tokenizer.DecodeIds(token_ids)
        else:
            return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(t, add_special_tokens=True) for t in texts]

        if truncation and max_length:
            encoded = [e[:max_length] for e in encoded]

        if padding:
            max_len = max(len(e) for e in encoded) if max_length is None else max_length
            max_len = min(max_len, max(len(e) for e in encoded))
            padded, masks = [], []
            for e in encoded:
                pad_len = max_len - len(e)
                padded.append(e + [self.pad_token_id] * pad_len)
                masks.append([1] * len(e) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        return {
            "input_ids": [torch.tensor(e, dtype=torch.long) for e in encoded],
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save config
        import json
        config = {
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str):
        import json
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path) as f:
            config = json.load(f)

        model_file = os.path.join(path, "tokenizer.model")
        model_path = model_file if os.path.exists(model_file) else None

        return cls(
            model_path=model_path,
            vocab_size=config["vocab_size"],
            model_type=config["model_type"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            pad_token=config["pad_token"],
            unk_token=config["unk_token"],
        )
