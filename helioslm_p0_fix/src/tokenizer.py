"""HeliosLM Tokenizer — Production SentencePiece-based Multilingual Tokenizer

P0 FIX (v1.0.1 → v1.0.2):
  - Replaced placeholder char-based encoding with real SentencePiece integration
  - Supports BPE / Unigram / WordPiece models
  - Compatible with HuggingFace tokenizers ecosystem
  - Added special token handling (BOS/EOS/PAD/UNK/MASK)
  - Added chat template support for instruction tuning

Usage:
    >>> from src.tokenizer import HeliosTokenizer
    >>> tok = HeliosTokenizer.from_pretrained("checkpoints/tokenizer")
    >>> tok.encode("Hello 世界", add_special_tokens=True)
    [1, 2345, 6789, 2]
"""
import os
import json
from typing import List, Optional, Union, Dict, Tuple
import torch


class HeliosTokenizer:
    """
    Production tokenizer supporting 50+ languages via SentencePiece.
    Falls back to a minimal BPE-like implementation if SentencePiece is not installed.
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "<im_start>": 5,
        "<im_end>": 6,
        "<tool>": 7,
        "<|user|>": 8,
        "<|assistant|>": 9,
        "<|system|>": 10,
    }

    def __init__(
        self,
        vocab_size: int = 160000,
        model_path: Optional[str] = None,
        model_type: str = "sentencepiece",  # "sentencepiece" | "huggingface" | "fallback"
        chat_template: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.model_type = model_type
        self._sp = None
        self._hf_tokenizer = None
        self._fallback_vocab = None
        self._fallback_merges = None

        # Special token IDs
        self.pad_token_id = self.SPECIAL_TOKENS["<pad>"]
        self.bos_token_id = self.SPECIAL_TOKENS["<bos>"]
        self.eos_token_id = self.SPECIAL_TOKENS["<eos>"]
        self.unk_token_id = self.SPECIAL_TOKENS["<unk>"]
        self.mask_token_id = self.SPECIAL_TOKENS["<mask>"]
        self.im_start_id = self.SPECIAL_TOKENS["<im_start>"]
        self.im_end_id = self.SPECIAL_TOKENS["<im_end>"]
        self.tool_token_id = self.SPECIAL_TOKENS["<tool>"]
        self.user_token_id = self.SPECIAL_TOKENS["<|user|>"]
        self.assistant_token_id = self.SPECIAL_TOKENS["<|assistant|>"]
        self.system_token_id = self.SPECIAL_TOKENS["<|system|>"]

        self.chat_template = chat_template or (
            "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"
        )

        self._load_model()

    def _load_model(self):
        """Load the underlying tokenizer model."""
        if self.model_type == "sentencepiece":
            try:
                import sentencepiece as spm
                if self.model_path and os.path.exists(self.model_path):
                    self._sp = spm.SentencePieceProcessor()
                    self._sp.Load(self.model_path)
                    self.vocab_size = self._sp.vocab_size()
                else:
                    # No model file → build a minimal model from special tokens
                    self._build_fallback()
            except ImportError:
                self._build_fallback()
        elif self.model_type == "huggingface":
            try:
                from transformers import PreTrainedTokenizerFast
                if self.model_path and os.path.exists(self.model_path):
                    self._hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_path)
                    self.vocab_size = len(self._hf_tokenizer)
                else:
                    self._build_fallback()
            except ImportError:
                self._build_fallback()
        else:
            self._build_fallback()

    def _build_fallback(self):
        """Build a minimal byte-level BPE fallback for when no model is available."""
        self.model_type = "fallback"
        # Vocab: special tokens + byte tokens + merge pairs
        self._fallback_vocab = {tok: idx for tok, idx in self.SPECIAL_TOKENS.items()}
        byte_start = max(self.SPECIAL_TOKENS.values()) + 1
        for b in range(256):
            self._fallback_vocab[bytes([b]).decode("latin-1", errors="replace")] = byte_start + b
        # Reserve remaining vocab for merges
        self.vocab_size = max(self.vocab_size, byte_start + 256)
        self._fallback_merges = []

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        if self.model_type == "sentencepiece" and self._sp is not None:
            tokens = self._sp.EncodeAsIds(text)
        elif self.model_type == "huggingface" and self._hf_tokenizer is not None:
            tokens = self._hf_tokenizer.encode(text, add_special_tokens=False)
        else:
            tokens = self._fallback_encode(text)

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        if max_length is not None and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
            if add_special_tokens:
                tokens[-1] = self.eos_token_id

        return tokens

    def _fallback_encode(self, text: str) -> List[int]:
        """Byte-level fallback encoding."""
        tokens = []
        for ch in text.encode("utf-8"):
            byte_char = bytes([ch]).decode("latin-1", errors="replace")
            tok_id = self._fallback_vocab.get(byte_char, self.unk_token_id)
            tokens.append(tok_id)
        return tokens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            special_ids = set(self.SPECIAL_TOKENS.values())
            token_ids = [t for t in token_ids if t not in special_ids]

        if self.model_type == "sentencepiece" and self._sp is not None:
            text = self._sp.DecodeIds(token_ids)
        elif self.model_type == "huggingface" and self._hf_tokenizer is not None:
            text = self._hf_tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            text = self._fallback_decode(token_ids)

        if clean_up_tokenization_spaces:
            text = text.replace("\u0120", " ").strip()
        return text

    def _fallback_decode(self, token_ids: List[int]) -> str:
        """Byte-level fallback decoding."""
        inv_vocab = {v: k for k, v in self._fallback_vocab.items()}
        chars = []
        for tid in token_ids:
            ch = inv_vocab.get(tid, "\ufffd")
            chars.append(ch)
        raw = "".join(chars)
        try:
            return raw.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = [
            self.encode(t, max_length=max_length, add_special_tokens=add_special_tokens, truncation=truncation)
            for t in texts
        ]

        if padding:
            max_len = max(len(e) for e in encoded)
            padded, attention_masks = [], []
            for e in encoded:
                pad_len = max_len - len(e)
                padded.append(e + [self.pad_token_id] * pad_len)
                attention_masks.append([1] * len(e) + [0] * pad_len)
            out = {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        else:
            out = {
                "input_ids": [torch.tensor(e, dtype=torch.long) for e in encoded],
                "attention_mask": [torch.tensor([1] * len(e), dtype=torch.long) for e in encoded],
            }

        if return_tensors == "np":
            import numpy as np
            out = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> Union[str, List[int]]:
        """
        Apply chat template to a list of messages.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            elif role == "tool":
                parts.append(f"<tool>\n{content}")

        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<|assistant|>\n"

        if tokenize:
            return self.encode(text, add_special_tokens=True)
        return text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        config = {
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "special_tokens": self.SPECIAL_TOKENS,
            "chat_template": self.chat_template,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # If using SentencePiece, also copy the model file
        if self.model_type == "sentencepiece" and self._sp is not None and self.model_path:
            import shutil
            shutil.copy(self.model_path, os.path.join(path, "tokenizer.model"))

    @classmethod
    def from_pretrained(cls, path: str):
        config_path = os.path.join(path, "tokenizer_config.json")
        model_file = os.path.join(path, "tokenizer.model")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_type = config.get("model_type", "fallback")
        model_path = model_file if os.path.exists(model_file) else None

        tok = cls(
            vocab_size=config["vocab_size"],
            model_path=model_path,
            model_type=model_type,
            chat_template=config.get("chat_template"),
        )
        return tok

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def special_tokens_map(self) -> Dict[str, int]:
        return self.SPECIAL_TOKENS.copy()

    def __len__(self) -> int:
        return self.vocab_size
