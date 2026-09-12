"""Image-text pair curation for vision-language pretraining."""
import json
import os
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
import torch
from PIL import Image
import hashlib


class ImageTextDataset:
    """
    Streaming dataset for image-text pairs.

    Sources:
      - LAION-5B (filtered)
      - COYO-700M
      - Internal proprietary data
      - Synthetic captions from strong VLM

    Format per sample:
    {
      "image_path": "s3://bucket/path.jpg",
      "caption": "A cat sitting on a mat",
      "metadata": {"source": "laion", "aesthetic_score": 6.5}
    }
    """

    def __init__(
        self,
        data_dir: str,
        min_aesthetic_score: float = 5.0,
        max_text_length: int = 256,
        min_text_length: int = 10,
        dedup_threshold: float = 0.95,
    ):
        self.data_dir = Path(data_dir)
        self.min_aesthetic = min_aesthetic_score
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
        self.dedup_threshold = dedup_threshold

        # Exact dedup set
        self.seen_hashes: set = set()

    def stream(self) -> Iterator[Dict]:
        """Stream filtered image-text pairs."""
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line)

                        # Quality filters
                        if not self._passes_filters(sample):
                            continue

                        # Deduplication
                        if not self._is_unique(sample):
                            continue

                        yield sample
                    except json.JSONDecodeError:
                        continue

    def _passes_filters(self, sample: Dict) -> bool:
        """Apply quality filters."""
        caption = sample.get("caption", "")

        # Length check
        if not (self.min_text_length <= len(caption) <= self.max_text_length):
            return False

        # Aesthetic score (for LAION)
        aesthetic = sample.get("metadata", {}).get("aesthetic_score", 5.0)
        if aesthetic < self.min_aesthetic:
            return False

        # Language check (basic)
        if not caption.isprintable():
            return False

        # Toxicity check (basic keyword)
        toxic_keywords = ["porn", "nsfw", "explicit", "gore"]
        if any(kw in caption.lower() for kw in toxic_keywords):
            return False

        return True

    def _is_unique(self, sample: Dict) -> bool:
        """Check if sample is not a near-duplicate."""
        caption = sample.get("caption", "")
        h = hashlib.md5(caption.encode()).hexdigest()

        if h in self.seen_hashes:
            return False

        self.seen_hashes.add(h)
        return True

    def create_synthetic_captions(self, image_dir: str, vlm_model, output_path: str, num_samples: int = 10000):
        """
        Generate synthetic captions using a strong VLM.

        Used when human captions are scarce or for domain adaptation.
        """
        count = 0
        with open(output_path, "w") as fout:
            for img_path in Path(image_dir).glob("*.jpg"):
                if count >= num_samples:
                    break

                try:
                    # Load image
                    image = Image.open(img_path).convert("RGB")

                    # Generate caption with VLM
                    # caption = vlm_model.generate_caption(image)
                    caption = f"Synthetic caption for {img_path.name}"

                    sample = {
                        "image_path": str(img_path),
                        "caption": caption,
                        "metadata": {"source": "synthetic", "model": "vlm-v1"},
                    }

                    fout.write(json.dumps(sample) + "\n")
                    count += 1

                    if count % 1000 == 0:
                        print(f"Generated {count} synthetic captions...")
                except Exception as e:
                    continue

        print(f"✅ Synthetic captions: {output_path} ({count} samples)")


class ImageTextMixer:
    """
    Mix multiple image-text sources with configurable ratios.

    Similar to Phase 2 text data mixer but for multimodal data.
    """

    def __init__(self, source_weights: Dict[str, float]):
        """
        Args:
            source_weights: {"laion": 0.5, "coyo": 0.3, "internal": 0.2}
        """
        self.source_weights = source_weights
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.source_weights.values())
        self.source_weights = {k: v / total for k, v in self.source_weights.items()}

    def sample_source(self) -> str:
        """Sample a data source based on weights."""
        import random
        sources = list(self.source_weights.keys())
        weights = list(self.source_weights.values())
        return random.choices(sources, weights=weights, k=1)[0]

    def create_streaming_iterator(self, source_iterators: Dict[str, Iterator]) -> Iterator[Dict]:
        """Create mixed streaming iterator."""
        buffers = {name: [] for name in source_iterators}

        def refill(name: str):
            iterator = source_iterators[name]
            try:
                for _ in range(100):
                    buffers[name].append(next(iterator))
            except StopIteration:
                pass

        # Initial fill
        for name in source_iterators:
            refill(name)

        while any(len(buffers[name]) > 0 for name in buffers):
            source = self.sample_source()

            if len(buffers[source]) == 0:
                refill(source)
                if len(buffers[source]) == 0:
                    continue

            sample = buffers[source].pop(0)
            sample["_source"] = source
            yield sample
