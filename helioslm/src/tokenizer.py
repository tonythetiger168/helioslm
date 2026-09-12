"""HeliosLM Tokenizer - SentencePiece-based Multilingual Tokenizer"""

import os
import json
from typing import List, Optional, Union
import torch


class HeliosTokenizer:
    """
    Production tokenizer supporting 50+ languages.
    Placeholder implementation - integrate SentencePiece or HuggingFace Tokenizers in production.
    """
    def __init__(self, vocab_size: int = 160000, model_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 3
        
        # In production, load SentencePiece model:
        # import sentencepiece as spm
        # self.sp = spm.SentencePieceProcessor()
        # self.sp.Load(model_path)
    
    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        # Placeholder: simple char-based encoding for demo
        tokens = [ord(c) % self.vocab_size for c in text[:max_length or len(text)]]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [
                self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id
            ]]
        
        # Placeholder: simple char decoding
        chars = []
        for t in token_ids:
            c = chr(t % 128)
            if 32 <= ord(c) <= 126:
                chars.append(c)
            else:
                chars.append(" ")
        return "".join(chars)
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, padding: bool = True) -> dict:
        """Batch encode texts. Returns dict with input_ids and attention_mask."""
        encoded = [self.encode(t, max_length, add_special_tokens=True) for t in texts]
        
        if padding:
            max_len = max(len(e) for e in encoded)
            padded = []
            attention_masks = []
            for e in encoded:
                pad_len = max_len - len(e)
                padded.append(e + [self.pad_token_id] * pad_len)
                attention_masks.append([1] * len(e) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        
        return {"input_ids": [torch.tensor(e, dtype=torch.long) for e in encoded]}
    
    def save(self, path: str):
        """Save tokenizer config."""
        os.makedirs(path, exist_ok=True)
        config = {
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from config."""
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(vocab_size=config["vocab_size"], model_path=path)
