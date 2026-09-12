"""Data Quality Classifier for 15-20T token filtering

Trains a small model (~0.5B params) to score data quality across dimensions:
  - grammar: syntactic correctness
  - knowledge: information density
  - coherence: logical flow
  - toxicity: harmful content
  - diversity: lexical diversity

Usage:
  1. Train: python scripts/train_quality_classifier.py
  2. Score:  python scripts/score_data.py --input data/raw --output data/scored
  3. Filter: Keep top 30% by composite score
"""
from .classifier_model import QualityClassifier, QualityDimensions
from .trainer import QualityClassifierTrainer
from .inference import batch_score_documents, filter_by_score
from .data_labeling import auto_label_documents, create_synthetic_labels

__all__ = ["QualityClassifier","QualityDimensions","QualityClassifierTrainer","batch_score_documents","filter_by_score","auto_label_documents","create_synthetic_labels"]
