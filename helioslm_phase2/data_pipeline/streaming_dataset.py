"""Streaming datasets for large-scale pre-training."""
import json
import random
from pathlib import Path
from typing import Iterator, Dict, Optional, List, Callable
import torch
from torch.utils.data import IterableDataset


class StreamingPretrainDataset(IterableDataset):
    """
    Streaming dataset for pre-training on web-scale data.

    Features:
      - Multi-worker sharding
      - Sequence packing for efficiency
      - Dynamic sequence length support
      - Integration with data mixer
    """

    def __init__(
        self,
        data_iterator: Iterator[Dict],
        tokenizer,
        seq_len: int = 4096,
        pack_sequences: bool = True,
        buffer_size: int = 100000,
    ):
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pack_sequences = pack_sequences
        self.buffer_size = buffer_size

        self._token_buffer: List[int] = []

    def _tokenize_document(self, doc: Dict) -> List[int]:
        """Tokenize a single document."""
        text = doc.get("text", "")
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return tokens

    def _pack_sequences(self, token_stream: Iterator[int]) -> Iterator[Dict]:
        """Pack tokens into fixed-length sequences with attention mask."""
        buffer = []

        for token in token_stream:
            buffer.append(token)

            if len(buffer) >= self.seq_len + 1:
                # Create input/label pair
                input_ids = buffer[:self.seq_len]
                labels = buffer[1:self.seq_len + 1]

                # Create attention mask (1 for real tokens, 0 for padding)
                # For packed sequences, we might want to mask across document boundaries
                attention_mask = [1] * self.seq_len

                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }

                buffer = buffer[self.seq_len:]

    def _unpacked_sequences(self, token_stream: Iterator[int]) -> Iterator[Dict]:
        """Create sequences without packing (pad to seq_len)."""
        buffer = []

        for token in token_stream:
            buffer.append(token)

            if len(buffer) >= self.seq_len + 1:
                input_ids = buffer[:self.seq_len]
                labels = buffer[1:self.seq_len + 1]

                # Pad if needed
                if len(input_ids) < self.seq_len:
                    pad_len = self.seq_len - len(input_ids)
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len

                attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]

                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }

                buffer = []

    def __iter__(self) -> Iterator[Dict]:
        worker_info = torch.utils.data.get_worker_info()

        # Shard across workers
        if worker_info is not None and worker_info.num_workers > 1:
            # Simple round-robin sharding
            shard_id = worker_info.id
            num_shards = worker_info.num_workers
        else:
            shard_id = 0
            num_shards = 1

        # Create token stream
        def token_stream():
            count = 0
            for doc in self.data_iterator:
                if count % num_shards == shard_id:
                    tokens = self._tokenize_document(doc)
                    for t in tokens:
                        yield t
                count += 1

        # Return sequences
        if self.pack_sequences:
            yield from self._pack_sequences(token_stream())
        else:
            yield from self._unpacked_sequences(token_stream())


class StreamingSFTDataset(IterableDataset):
    """Streaming dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_iterator: Iterator[Dict],
        tokenizer,
        seq_len: int = 4096,
        mask_prompt: bool = True,
    ):
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prompt = mask_prompt

    def __iter__(self) -> Iterator[Dict]:
        worker_info = torch.utils.data.get_worker_info()
        shard_id = worker_info.id if worker_info else 0
        num_shards = worker_info.num_workers if worker_info else 1

        count = 0
        for doc in self.data_iterator:
            if count % num_shards != shard_id:
                count += 1
                continue

            # Format: instruction + response
            instruction = doc.get("instruction", doc.get("prompt", ""))
            response = doc.get("response", doc.get("output", ""))

            # Apply chat template
            text = f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            if len(tokens) < 2:
                count += 1
                continue

            input_ids = tokens[:-1][:self.seq_len]
            labels = tokens[1:][:self.seq_len]

            # Mask prompt tokens (only compute loss on response)
            if self.mask_prompt:
                prompt_tokens = self.tokenizer.encode(
                    f"<|user|>\n{instruction}\n<|assistant|>\n",
                    add_special_tokens=True,
                )
                for i in range(min(len(prompt_tokens) - 1, len(labels))):
                    labels[i] = -100

            # Pad
            if len(input_ids) < self.seq_len:
                pad_len = self.seq_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
            count += 1
