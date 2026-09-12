#!/usr/bin/env python3
"""Train the HeliosLM Data Quality Classifier.

Usage:
    # 1. Generate synthetic labels from raw data
    python scripts/train_quality_classifier.py --generate-labels \
        --input data/raw.jsonl --output data/labeled.jsonl --num-samples 100000

    # 2. Train classifier
    python scripts/train_quality_classifier.py --train \
        --data data/labeled.jsonl --output checkpoints/quality_classifier.pt \
        --epochs 3 --batch-size 32

    # 3. Full pipeline (generate + train)
    python scripts/train_quality_classifier.py --full-pipeline \
        --input data/raw.jsonl --output checkpoints/quality_classifier.pt
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data_pipeline.tokenizer_wrapper import TokenizerWrapper
from data_pipeline.quality_classifier import (
    QualityClassifier,
    QualityClassifierTrainer,
    create_synthetic_labels,
)
from data_pipeline.quality_classifier.trainer import QualityDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train HeliosLM Quality Classifier")

    # Mode
    parser.add_argument("--generate-labels", action="store_true",
                        help="Generate synthetic labels from raw data")
    parser.add_argument("--train", action="store_true",
                        help="Train classifier on labeled data")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run full pipeline: label generation + training")

    # Data paths
    parser.add_argument("--input", default="data/raw.jsonl",
                        help="Raw data input path")
    parser.add_argument("--data", default="data/labeled.jsonl",
                        help="Labeled data path for training")
    parser.add_argument("--output", default="checkpoints/quality_classifier.pt",
                        help="Output checkpoint path")

    # Label generation
    parser.add_argument("--num-samples", type=int, default=100000,
                        help="Number of samples to label")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)

    # Model
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)

    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("HeliosLM Quality Classifier Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.hidden_size}d, {args.num_layers}L, {args.num_heads}H")
    print()

    # Step 1: Generate labels if needed
    if args.generate_labels or args.full_pipeline:
        print("📋 Step 1: Generating synthetic labels...")
        create_synthetic_labels(
            input_path=args.input,
            output_path=args.data,
            num_samples=args.num_samples,
        )
        print()

    # Step 2: Train classifier
    if args.train or args.full_pipeline:
        print("🎓 Step 2: Training quality classifier...")

        # Tokenizer
        tokenizer = TokenizerWrapper(vocab_size=160000, model_type="fallback")
        print(f"Tokenizer: {len(tokenizer)} vocab")

        # Model
        model = QualityClassifier(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
        )
        print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        # Dataset
        dataset = QualityDataset(args.data, tokenizer, max_length=args.max_seq_len)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        print(f"Dataset: {len(dataset)} samples")

        # Trainer
        trainer = QualityClassifierTrainer(
            model=model,
            learning_rate=args.learning_rate,
            device=args.device,
        )

        # Train
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            avg_loss = trainer.train_epoch(dataloader, log_interval=50)
            print(f"Average loss: {avg_loss:.4f}")

        # Save
        trainer.save(args.output)
        print(f"\n✅ Training complete! Model saved: {args.output}")

    print("=" * 60)


if __name__ == "__main__":
    main()
