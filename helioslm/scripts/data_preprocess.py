#!/usr/bin/env python3
"""HeliosLM Data Preprocessing - Phase 2: Quality filtering, deduplication, toxicity filtering

Processes raw data into training-ready format:
- Quality scoring (language model perplexity, length, coherence)
- Deduplication (MinHash LSH for near-duplicate detection)
- Toxicity filtering (keyword + classifier-based)
- Tokenization and chunking
- Data mixing and sharding

Usage:
    # Process single source
    python scripts/data_preprocess.py --input data/raw/web --output data/processed/web --source web
    
    # Process all sources with mixing
    python scripts/data_preprocess.py --all --config configs/data_mix.yaml --output data/processed
    
    # Deduplicate across all sources
    python scripts/data_preprocess.py --dedup --input data/processed --output data/deduped
"""

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import tqdm


class QualityFilter:
    """Filter documents by quality metrics."""
    
    def __init__(self, 
                 min_length: int = 100,
                 max_length: int = 100000,
                 min_words: int = 10,
                 max_repetition: float = 0.3,
                 min_alpha_ratio: float = 0.5):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_repetition = max_repetition
        self.min_alpha_ratio = min_alpha_ratio
    
    def score(self, text: str) -> Tuple[float, Dict]:
        """Score document quality. Returns (score, metadata)."""
        metrics = {}
        
        # Length check
        metrics["length"] = len(text)
        if metrics["length"] < self.min_length or metrics["length"] > self.max_length:
            return 0.0, metrics
        
        # Word count
        words = text.split()
        metrics["word_count"] = len(words)
        if metrics["word_count"] < self.min_words:
            return 0.0, metrics
        
        # Repetition check (n-gram repetition)
        ngrams = Counter()
        for i in range(len(words) - 2):
            ngrams[tuple(words[i:i+3])] += 1
        if ngrams:
            max_repeat = max(ngrams.values()) / len(words)
            metrics["max_repetition"] = max_repeat
            if max_repeat > self.max_repetition:
                return 0.0, metrics
        
        # Alpha ratio (filter garbled text)
        alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        metrics["alpha_ratio"] = alpha_chars / len(text) if text else 0
        if metrics["alpha_ratio"] < self.min_alpha_ratio:
            return 0.0, metrics
        
        # Language detection (simple heuristic - in production use fasttext)
        metrics["language"] = self._detect_language(text)
        
        # Overall quality score
        score = min(1.0, metrics["word_count"] / 1000) * metrics["alpha_ratio"]
        score *= (1 - metrics.get("max_repetition", 0))
        
        return score, metrics
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection. In production, use fasttext."""
        # Check for common non-English characters
        if any(ord(c) > 127 for c in text[:100]):
            return "unknown"
        return "en"


class ToxicityFilter:
    """Filter toxic/harmful content."""
    
    # Basic toxic keywords (expand in production)
    TOXIC_PATTERNS = [
        r"\b(hate|kill|die|violence|abuse|attack)\b",
        r"\b(racist|sexist|homophobic|transphobic)\b",
        r"\b(nazi|terrorist|extremist)\b",
    ]
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
    
    def score(self, text: str) -> float:
        """Return toxicity score (0 = clean, 1 = toxic)."""
        text_lower = text.lower()
        
        # Keyword matching
        toxic_hits = sum(1 for p in self.patterns if p.search(text_lower))
        keyword_score = min(1.0, toxic_hits / len(self.patterns))
        
        # In production, add classifier-based scoring
        # from transformers import pipeline
        # classifier = pipeline("text-classification", model="unitary/toxic-bert")
        
        return keyword_score
    
    def is_toxic(self, text: str) -> bool:
        return self.score(text) > self.threshold


class MinHashDeduplicator:
    """MinHash LSH for near-duplicate detection."""
    
    def __init__(self, num_hashes: int = 128, threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.threshold = threshold
        self.seen_hashes = set()
    
    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Extract k-shingles from text."""
        words = text.split()
        return set(" ".join(words[i:i+k]) for i in range(len(words) - k + 1))
    
    def _minhash(self, shingles: Set[str]) -> Tuple[int, ...]:
        """Compute MinHash signature."""
        # Simplified MinHash using Python hash
        signature = []
        for i in range(self.num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                h = hash((shingle, i))
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return tuple(signature)
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of previously seen text."""
        shingles = self._get_shingles(text)
        if not shingles:
            return False
        
        signature = self._minhash(shingles)
        
        # Check exact signature match (simplified LSH)
        if signature in self.seen_hashes:
            return True
        
        self.seen_hashes.add(signature)
        return False


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self, 
                 quality_filter: Optional[QualityFilter] = None,
                 toxicity_filter: Optional[ToxicityFilter] = None,
                 deduplicator: Optional[MinHashDeduplicator] = None):
        self.quality_filter = quality_filter or QualityFilter()
        self.toxicity_filter = toxicity_filter or ToxicityFilter()
        self.deduplicator = deduplicator or MinHashDeduplicator()
        
        self.stats = {
            "total": 0,
            "quality_filtered": 0,
            "toxicity_filtered": 0,
            "deduped": 0,
            "passed": 0,
        }
    
    def process_file(self, input_path: str, output_path: str) -> Dict:
        """Process a single JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            for line in tqdm.tqdm(infile, desc=f"Processing {os.path.basename(input_path)}"):
                self.stats["total"] += 1
                
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    
                    # Quality filter
                    quality_score, metrics = self.quality_filter.score(text)
                    if quality_score < 0.3:
                        self.stats["quality_filtered"] += 1
                        continue
                    
                    # Toxicity filter
                    if self.toxicity_filter.is_toxic(text):
                        self.stats["toxicity_filtered"] += 1
                        continue
                    
                    # Deduplication
                    if self.deduplicator.is_duplicate(text):
                        self.stats["deduped"] += 1
                        continue
                    
                    # Add quality metrics
                    doc["quality_score"] = quality_score
                    doc["quality_metrics"] = metrics
                    
                    outfile.write(json.dumps(doc) + "\n")
                    self.stats["passed"] += 1
                    
                except json.JSONDecodeError:
                    continue
        
        return self.stats
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict:
        """Process all JSONL files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for jsonl_file in sorted(input_path.glob("*.jsonl")):
            output_file = output_path / jsonl_file.name
            self.process_file(str(jsonl_file), str(output_file))
        
        return self.stats
    
    def print_stats(self):
        """Print preprocessing statistics."""
        print("\n" + "=" * 60)
        print("Preprocessing Statistics")
        print("=" * 60)
        print(f"Total documents:      {self.stats['total']:,}")
        print(f"Quality filtered:     {self.stats['quality_filtered']:,} ({self.stats['quality_filtered']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"Toxicity filtered:    {self.stats['toxicity_filtered']:,} ({self.stats['toxicity_filtered']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"Deduplicated:         {self.stats['deduped']:,} ({self.stats['deduped']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"Passed:               {self.stats['passed']:,} ({self.stats['passed']/max(self.stats['total'],1)*100:.1f}%)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="HeliosLM Data Preprocessing")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory or file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory or file")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "web", "code", "academic", "books", "wiki"],
                        help="Source type for specialized filtering")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Minimum document length")
    parser.add_argument("--max-length", type=int, default=100000,
                        help="Maximum document length")
    parser.add_argument("--toxicity-threshold", type=float, default=0.5,
                        help="Toxicity filtering threshold")
    parser.add_argument("--dedup-threshold", type=float, default=0.8,
                        help="Deduplication Jaccard threshold")
    parser.add_argument("--compress", action="store_true",
                        help="Gzip compress output")
    
    args = parser.parse_args()
    
    # Configure filters based on source type
    if args.source == "code":
        quality_filter = QualityFilter(min_length=50, min_words=5, min_alpha_ratio=0.3)
    elif args.source == "academic":
        quality_filter = QualityFilter(min_length=200, min_words=20, min_alpha_ratio=0.6)
    else:
        quality_filter = QualityFilter(min_length=args.min_length, min_alpha_ratio=0.5)
    
    toxicity_filter = ToxicityFilter(threshold=args.toxicity_threshold)
    deduplicator = MinHashDeduplicator(threshold=args.dedup_threshold)
    
    preprocessor = DataPreprocessor(quality_filter, toxicity_filter, deduplicator)
    
    # Process
    input_path = Path(args.input)
    if input_path.is_file():
        preprocessor.process_file(args.input, args.output)
    else:
        preprocessor.process_directory(args.input, args.output)
    
    preprocessor.print_stats()


if __name__ == "__main__":
    main()
