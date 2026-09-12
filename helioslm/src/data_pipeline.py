"""HeliosLM Data Pipeline - WebDataset/MosaicML-compatible Streaming"""

import json
import random
from pathlib import Path
from typing import Iterator, Dict, Optional, List
import torch
from torch.utils.data import IterableDataset


class PretrainDataset(IterableDataset):
    """Streaming dataset for pre-training on web-scale data."""
    
    def __init__(self, data_dir: str, seq_len: int = 4096, tokenizer=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.files = sorted(self.data_dir.glob("*.jsonl"))
        if not self.files:
            raise ValueError(f"No .jsonl files found in {data_dir}")
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        files = self.files.copy()
        
        if worker_info is not None and worker_info.num_workers > 1:
            # Shard files across workers
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]
        
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        
                        if self.tokenizer:
                            tokens = self.tokenizer.encode(text, max_length=self.seq_len + 1)
                        else:
                            tokens = [ord(c) % 160000 for c in text[:self.seq_len + 1]]
                        
                        if len(tokens) < 2:
                            continue
                        
                        # Create input/label pairs (next token prediction)
                        input_ids = tokens[:-1]
                        labels = tokens[1:]
                        
                        # Pad or truncate to seq_len
                        if len(input_ids) < self.seq_len:
                            pad_len = self.seq_len - len(input_ids)
                            input_ids = input_ids + [0] * pad_len
                            labels = labels + [-100] * pad_len
                        else:
                            input_ids = input_ids[:self.seq_len]
                            labels = labels[:self.seq_len]
                        
                        yield {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        }
                    except (json.JSONDecodeError, KeyError):
                        continue


class SFTDataset(IterableDataset):
    """Supervised Fine-Tuning dataset for instruction following."""
    
    def __init__(self, data_path: str, seq_len: int = 4096, tokenizer=None):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        if not self.data_path.exists():
            raise ValueError(f"Data file not found: {data_path}")
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt", "")
                    response = data.get("response", "")
                    
                    # Format: User/Assistant conversation
                    full_text = f"User: {prompt}\nAssistant: {response}"
                    
                    if self.tokenizer:
                        tokens = self.tokenizer.encode(full_text, max_length=self.seq_len + 1)
                    else:
                        tokens = [ord(c) % 160000 for c in full_text[:self.seq_len + 1]]
                    
                    if len(tokens) < 2:
                        continue
                    
                    input_ids = tokens[:-1]
                    labels = tokens[1:]
                    
                    if len(input_ids) < self.seq_len:
                        pad_len = self.seq_len - len(input_ids)
                        input_ids = input_ids + [0] * pad_len
                        labels = labels + [-100] * pad_len
                    else:
                        input_ids = input_ids[:self.seq_len]
                        labels = labels[:self.seq_len]
                    
                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                except (json.JSONDecodeError, KeyError):
                    continue


class RLHFDataset(IterableDataset):
    """RLHF dataset with prompt, chosen, rejected triplets."""
    
    def __init__(self, data_path: str, seq_len: int = 4096, tokenizer=None):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt", "")
                    chosen = data.get("chosen", "")
                    rejected = data.get("rejected", "")
                    
                    chosen_text = f"User: {prompt}\nAssistant: {chosen}"
                    rejected_text = f"User: {prompt}\nAssistant: {rejected}"
                    
                    if self.tokenizer:
                        chosen_tokens = self.tokenizer.encode(chosen_text, max_length=self.seq_len)
                        rejected_tokens = self.tokenizer.encode(rejected_text, max_length=self.seq_len)
                    else:
                        chosen_tokens = [ord(c) % 160000 for c in chosen_text[:self.seq_len]]
                        rejected_tokens = [ord(c) % 160000 for c in rejected_text[:self.seq_len]]
                    
                    yield {
                        "chosen_input_ids": torch.tensor(chosen_tokens, dtype=torch.long),
                        "rejected_input_ids": torch.tensor(rejected_tokens, dtype=torch.long),
                    }
                except (json.JSONDecodeError, KeyError):
                    continue


def create_dataloader(dataset: IterableDataset, batch_size: int, num_workers: int = 4):
    """Create DataLoader for streaming dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
