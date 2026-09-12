#!/usr/bin/env python3
"""Score and filter pre-training data using the quality classifier.

Usage:
    python scripts/score_data.py \
        --input data/raw.jsonl \
        --output data/filtered.jsonl \
        --classifier checkpoints/quality_classifier.pt \
        --top-percent 30 \
        --batch-size 64

Output format:
    Each line is a JSON with added fields:
      - quality_scores: {grammar, knowledge, coherence, toxicity, diversity}
      - quality_composite: float (0-1)
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data_pipeline.tokenizer_wrapper import TokenizerWrapper
from data_pipeline.quality_classifier import (
    QualityClassifier,
    batch_score_documents,
    filter_by_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Score and filter pre-training data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--classifier", required=True, help="Classifier checkpoint path")
    parser.add_argument("--top-percent", type=float, default=30.0,
                        help="Keep top K%% by quality score")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum composite score (alternative to top-percent)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to process")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("HeliosLM Data Scoring & Filtering")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Classifier: {args.classifier}")
    print(f"Device: {args.device}")
    print()

    # Load tokenizer
    tokenizer = TokenizerWrapper(vocab_size=160000, model_type="fallback")

    # Load classifier
    print("📂 Loading classifier...")
    model = QualityClassifier(vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(args.classifier, map_location=args.device))
    model.to(args.device)
    model.eval()
    print("✅ Classifier loaded")

    # Stream documents
    def document_stream():
        count = 0
        with open(args.input) as f:
            for line in f:
                if args.max_docs and count >= args.max_docs:
                    break
                try:
                    doc = json.loads(line)
                    if "text" in doc and len(doc["text"]) > 50:
                        yield doc
                        count += 1
                except json.JSONDecodeError:
                    continue

    # Score
    print(f"\n🔍 Scoring documents...")
    scored_docs = batch_score_documents(
        document_stream(),
        model,
        tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Filter
    print(f"\n🔧 Filtering (top {args.top_percent}%%)...")
    if args.threshold > 0:
        filtered = filter_by_score(scored_docs, threshold=args.threshold)
    else:
        filtered = filter_by_score(scored_docs, top_k_percent=args.top_percent / 100)

    # Write output
    kept = 0
    total = 0
    with open(args.output, "w") as f:
        for doc in filtered:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
            total += 1

    print(f"\n📊 Results:")
    print(f"  Processed: {total} documents")
    print(f"  Kept:      {kept} documents ({kept/max(total,1)*100:.1f}%)")
    print(f"  Output:    {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
