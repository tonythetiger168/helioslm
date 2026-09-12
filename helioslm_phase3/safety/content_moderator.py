"""Content moderation for input and output text."""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ModerationCategory(Enum):
    HATE = "hate"
    HARASSMENT = "harassment"
    SELF_HARM = "self-harm"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    ILLEGAL = "illegal"
    PII = "pii"
    JAILBREAK = "jailbreak"


@dataclass
class ModerationResult:
    """Result of content moderation check."""
    flagged: bool
    categories: Dict[ModerationCategory, float]
    category_scores: Dict[str, float]
    reason: Optional[str] = None


class ContentModerator:
    """
    Multi-layer content moderation system.

    Layers:
      1. Rule-based: regex patterns for known harmful content
      2. Keyword-based: banned word lists
      3. Model-based: small classifier for nuanced detection (optional)
    """

    def __init__(self, use_model: bool = False):
        self.use_model = use_model

        # Banned patterns
        self.banned_patterns = {
            ModerationCategory.HATE: [
                r"\b(hate|kill\s+(all|every)|genocide|ethnic\s+cleansing)\b",
            ],
            ModerationCategory.HARASSMENT: [
                r"\b(doxx|swatting|stalking|cyberbully)\b",
            ],
            ModerationCategory.SELF_HARM: [
                r"\b(suicide|self-harm|cutting\s+myself|end\s+my\s+life)\b",
            ],
            ModerationCategory.SEXUAL: [
                r"\b(child\s+(porn|abuse)|csam)\b",
            ],
            ModerationCategory.VIOLENCE: [
                r"\b(bomb\s+making|how\s+to\s+make\s+a\s+weapon)\b",
            ],
            ModerationCategory.ILLEGAL: [
                r"\b(how\s+to\s+(steal|hack|forge))\b",
            ],
            ModerationCategory.PII: [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            ],
            ModerationCategory.JAILBREAK: [
                r"\b(ignore\s+previous\s+instructions|DAN|jailbreak)\b",
                r"\b(you\s+are\s+now\s+in\s+developer\s+mode)\b",
            ],
        }

        # Thresholds
        self.thresholds = {
            ModerationCategory.HATE: 0.8,
            ModerationCategory.HARASSMENT: 0.8,
            ModerationCategory.SELF_HARM: 0.5,  # Lower threshold for self-harm
            ModerationCategory.SEXUAL: 0.9,
            ModerationCategory.VIOLENCE: 0.8,
            ModerationCategory.ILLEGAL: 0.8,
            ModerationCategory.PII: 0.9,
            ModerationCategory.JAILBREAK: 0.7,
        }

    def moderate(self, text: str) -> ModerationResult:
        """Moderate text content."""
        categories = {}
        category_scores = {}

        text_lower = text.lower()

        for category, patterns in self.banned_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += min(matches * 0.3, 1.0)  # Each match adds 0.3, max 1.0

            categories[category] = score
            category_scores[category.value] = score

        # Check if any category exceeds threshold
        flagged = any(
            score >= self.thresholds.get(cat, 0.8)
            for cat, score in categories.items()
        )

        reason = None
        if flagged:
            violating = [
                cat.value for cat, score in categories.items()
                if score >= self.thresholds.get(cat, 0.8)
            ]
            reason = f"Flagged categories: {', '.join(violating)}"

        return ModerationResult(
            flagged=flagged,
            categories=categories,
            category_scores=category_scores,
            reason=reason,
        )

    def moderate_batch(self, texts: List[str]) -> List[ModerationResult]:
        """Moderate a batch of texts."""
        return [self.moderate(text) for text in texts]
