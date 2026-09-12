"""Automatic data labeling for quality classifier training."""
import re
import math
from typing import Dict, List
from collections import Counter


def auto_label_documents(text: str) -> Dict[str, float]:
    """
    Automatically label a document with quality scores using heuristics.

    These are weak labels used for initial training. Fine-tune with
    human annotations for production quality.
    """
    labels = {}

    # Grammar: check for proper sentence structure
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    labels["grammar"] = min(1.0, max(0.0, 1.0 - abs(avg_sentence_length - 15) / 30))

    # Knowledge: check for entity density and numbers
    numbers = len(re.findall(r'\d+', text))
    entities = len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text))
    labels["knowledge"] = min(1.0, (numbers + entities) / (len(text.split()) / 50 + 1))

    # Coherence: check for repeated phrases and transitions
    words = text.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 2)
    labels["coherence"] = min(1.0, max(0.0, 1.0 - repeated_bigrams / max(len(bigrams), 1) * 10))

    # Toxicity: check for toxic keywords (inverted: higher = safer)
    toxic_words = ['hate', 'kill', 'die', 'stupid', 'idiot', 'moron']
    toxic_count = sum(text.lower().count(w) for w in toxic_words)
    labels["toxicity"] = min(1.0, max(0.0, 1.0 - toxic_count / max(len(words), 1) * 100))

    # Diversity: unique word ratio
    unique_words = len(set(words))
    labels["diversity"] = min(1.0, unique_words / max(len(words), 1) * 2)

    # Composite
    labels["composite"] = (
        0.25 * labels["grammar"] +
        0.25 * labels["knowledge"] +
        0.20 * labels["coherence"] +
        0.15 * labels["toxicity"] +
        0.15 * labels["diversity"]
    )

    return labels


def create_synthetic_labels(input_path: str, output_path: str, num_samples: int = 100000):
    """
    Create synthetic training data by auto-labeling raw documents.

    Usage:
      python -c "from data_pipeline.quality_classifier import create_synthetic_labels; create_synthetic_labels('raw.jsonl', 'labeled.jsonl')"
    """
    import json
    count = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if count >= num_samples:
                break

            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if len(text) < 100:
                    continue

                labels = auto_label_documents(text)
                doc["labels"] = labels

                fout.write(json.dumps(doc) + "\n")
                count += 1

                if count % 10000 == 0:
                    print(f"Labeled {count} documents...")
            except Exception:
                continue

    print(f"✅ Created {count} synthetic labels: {output_path}")
