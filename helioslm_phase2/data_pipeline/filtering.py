"""Quality filtering for pre-training data."""
import re
import math
from typing import Dict, Optional, List, Callable
import torch
import torch.nn as nn


class QualityFilter:
    """Base quality filter."""

    def __call__(self, doc: Dict) -> bool:
        raise NotImplementedError


class LanguageFilter(QualityFilter):
    """Filter by language detection and character ratio."""

    def __init__(
        self,
        allowed_langs: Optional[List[str]] = None,
        min_char_ratio: float = 0.8,
        max_nonprintable_ratio: float = 0.1,
    ):
        self.allowed_langs = allowed_langs or ["en", "zh", "ja", "ko", "de", "fr", "es", "ru"]
        self.min_char_ratio = min_char_ratio
        self.max_nonprintable_ratio = max_nonprintable_ratio

    def __call__(self, doc: Dict) -> bool:
        text = doc.get("text", "")
        if not text:
            return False

        # Character ratio check
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        if printable / max(len(text), 1) < self.min_char_ratio:
            return False

        # Non-printable ratio
        nonprint = sum(1 for c in text if not c.isprintable() and not c.isspace())
        if nonprint / max(len(text), 1) > self.max_nonprintable_ratio:
            return False

        # Length check
        if len(text) < 100:
            return False

        return True


class ToxicityFilter(QualityFilter):
    """Filter toxic content using keyword and pattern matching."""

    def __init__(self, toxicity_threshold: float = 0.5):
        self.toxicity_threshold = toxicity_threshold
        # Basic toxic patterns (production should use a trained classifier)
        self.toxic_patterns = [
            r"\b(hate|kill|die|rape|nazi|terrorist)\b",
            r"\b(fuck|shit|bitch|cunt|asshole)\b{4,}",  # excessive profanity
        ]
        self.pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
        ]

    def __call__(self, doc: Dict) -> bool:
        text = doc.get("text", "")

        # Check toxic patterns
        toxic_count = sum(1 for p in self.toxic_patterns if re.search(p, text, re.IGNORECASE))
        if toxic_count >= 2:
            return False

        # Check PII
        pii_count = sum(1 for p in self.pii_patterns if re.search(p, text))
        if pii_count > 0:
            return False

        return True


class PerplexityFilter(QualityFilter):
    """
    Filter by perplexity using a small reference model.
    Documents with very high or very low perplexity are likely low quality.
    """

    def __init__(
        self,
        min_perplexity: float = 20.0,
        max_perplexity: float = 5000.0,
        model_name: str = "gpt2",  # Small reference model
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

    def _load_model(self):
        if self.model is None:
            try:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
            except Exception:
                # Fallback: skip perplexity filtering if model unavailable
                self.model = "unavailable"

    def __call__(self, doc: Dict) -> bool:
        self._load_model()
        if self.model == "unavailable":
            return True

        text = doc.get("text", "")[:1024]  # Sample first 1K chars
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = math.exp(loss)

            return self.min_perplexity <= perplexity <= self.max_perplexity
        except Exception:
            return True


class CompositeFilter(QualityFilter):
    """Chain multiple filters."""

    def __init__(self, filters: List[QualityFilter]):
        self.filters = filters

    def __call__(self, doc: Dict) -> bool:
        return all(f(doc) for f in self.filters)
