#!/usr/bin/env python3
"""HeliosLM Training Script - Production Distributed Training with DeepSpeed Support

Usage:
    # Pre-training (requires 256x H100)
    python scripts/train.py --config configs/ultra_config.py --stage pretrain --data data/pretrain

    # Supervised Fine-Tuning
    python scripts/train.py --config configs/pro_config.py --stage sft --data data/sft.jsonl

    # RLHF
    python scripts/train.py --config configs/pro_config.py --stage rlhf --data data/rlhf.jsonl

    # With DeepSpeed
    deepspeed scripts/train.py --deepspeed ds_config.json --config configs/pro_config.py --stage pretrain
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import HeliosLMConfig, ultra_config, pro_config, lite_config, nano_config
from src import HeliosLM


class DummyDataset(Dataset):
    """Placeholder dataset for demonstration. Replace with real data pipeline."""
    def __init__(self, num_samples=10000, seq_len=4096, vocab_size=160000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In production, load from WebDataset / MosaicML Streaming
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": labels}


def get_config(size: str) -> HeliosLMConfig:
    configs = {
        "ultra": ultra_config,
        "pro": pro_config,
        "lite": lite_config,
        "nano": nano_config,
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    return configs[size]


def train_pretrain(model: HeliosLM, dataloader: DataLoader, config: HeliosLMConfig,
                   epochs: int = 1, device: str = "cuda"):
    """Pre-training loop with DeepSpeed / FSDP support."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
                    logits = outputs["logits"]
                    aux_loss = outputs["aux_loss"]
                    # Simple next-token prediction loss
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    ) + aux_loss
            else:
                outputs = model(input_ids)
                logits = outputs["logits"]
                aux_loss = outputs["aux_loss"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                ) + aux_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")


def train_sft(model: HeliosLM, dataloader: DataLoader, config: HeliosLMConfig,
              epochs: int = 3, device: str = "cuda"):
    """Supervised Fine-Tuning loop."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = outputs["logits"]
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"SFT Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"SFT Epoch {epoch} completed. Avg loss: {total_loss / len(dataloader):.4f}")


def train_rlhf(model: HeliosLM, dataloader: DataLoader, config: HeliosLMConfig,
               epochs: int = 1, device: str = "cuda"):
    """RLHF training with PPO/DPO. Placeholder for full implementation."""
    print("RLHF training requires preference data and reward model.")
    print("This is a placeholder. Implement DPO/PPO here.")
    # TODO: Implement DPO or PPO training loop


def main():
    parser = argparse.ArgumentParser(description="HeliosLM Training")
    parser.add_argument("--config", type=str, default="configs/nano_config.py",
                        help="Path to config or size name (ultra/pro/lite/nano)")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "sft", "rlhf"],
                        help="Training stage")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--size", type=str, default="nano",
                        choices=["ultra", "pro", "lite", "nano"],
                        help="Model size")

    args = parser.parse_args()

    # Load config
    config = get_config(args.size)
    print(f"Loaded config: {config.model_name}")
    print(f"Hidden size: {config.hidden_size}, Layers: {config.num_hidden_layers}")
    print(f"MoE experts: {config.moe.num_experts}, Activated: {config.moe.num_activated_experts}")

    # Create model
    model = HeliosLM(config, size=args.size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Create dummy dataset (replace with real data in production)
    dataset = DummyDataset(num_samples=1000, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training
    os.makedirs(args.output, exist_ok=True)

    if args.stage == "pretrain":
        train_pretrain(model, dataloader, config, epochs=args.epochs, device=args.device)
    elif args.stage == "sft":
        train_sft(model, dataloader, config, epochs=args.epochs, device=args.device)
    elif args.stage == "rlhf":
        train_rlhf(model, dataloader, config, epochs=args.epochs, device=args.device)

    # Save checkpoint
    checkpoint_path = os.path.join(args.output, f"helioslm_{args.size}_{args.stage}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "stage": args.stage,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
