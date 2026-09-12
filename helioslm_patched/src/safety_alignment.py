"""Safety Alignment v2 + Constitutional AI - P7 (Production-Ready)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmfulnessClassifier(nn.Module):
    """Multi-level harmful content classifier with severity scoring."""

    HARM_CATEGORIES = [
        "violence", "hate_speech", "harassment", "self_harm",
        "sexual_content", "illegal_acts", "misinformation",
        "privacy_violation", "malware", "fraud", "discrimination", "toxicity"
    ]

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_categories = len(self.HARM_CATEGORIES)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                batch_first=True, dropout=0.1
            ),
            num_layers=2
        )

        self.category_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_categories),
            nn.Sigmoid()
        )

        self.severity_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 4)  # 4 severity levels
        )

    def forward(self, hidden_states):
        """
        Classify harmful content.

        Returns:
            dict with category scores, severity, and is_harmful flag
        """
        encoded = self.encoder(hidden_states)
        pooled = encoded.mean(dim=1)

        category_scores = self.category_head(pooled)
        severity_logits = self.severity_head(pooled)

        return {
            "category_scores": category_scores,
            "severity": F.softmax(severity_logits, dim=-1),
            "is_harmful": (category_scores > 0.5).any(dim=-1),
            "max_category": self.HARM_CATEGORIES[category_scores.argmax(dim=-1).item()] if category_scores.numel() > 0 else None,
        }


class ValueAlignmentHead(nn.Module):
    """HHH (Helpful, Harmless, Honest) value alignment scoring."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.hhh_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )

        self.alignment_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        """
        Score HHH values.

        Returns:
            dict with helpful, harmless, honest scores and overall alignment
        """
        pooled = hidden_states.mean(dim=1)
        hhh = torch.sigmoid(self.hhh_head(pooled))
        alignment = self.alignment_head(pooled)

        return {
            "helpful": hhh[:, 0],
            "harmless": hhh[:, 1],
            "honest": hhh[:, 2],
            "alignment_score": alignment,
        }


class ConstitutionalRuleEngine:
    """Configurable constitutional rule engine with embedding-based matching."""

    DEFAULT_RULES = [
        "Do not produce content that promotes violence or illegal acts.",
        "Respect user privacy and do not request sensitive personal information.",
        "Provide accurate information and acknowledge uncertainty.",
        "Avoid generating content that discriminates against protected groups.",
        "Do not assist in creating malware, scams, or fraudulent content.",
    ]

    def __init__(self, config):
        self.config = config
        self.rules = config.safety.constitutional_rules or self.DEFAULT_RULES

        # Simple rule embeddings (in production, use sentence transformer)
        self.rule_embeddings = None
        self._build_rule_embeddings()

    def _build_rule_embeddings(self):
        """Build simple rule embeddings."""
        # Placeholder: in production, use a text encoder
        import hashlib
        self.rule_embeddings = []
        for rule in self.rules:
            # Simple hash-based embedding for demonstration
            h = hashlib.md5(rule.encode()).hexdigest()
            emb = torch.tensor([int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)])
            self.rule_embeddings.append(emb)

    def evaluate(self, text_embedding):
        """
        Evaluate constitutional compliance.

        Returns:
            dict with rule compliance scores and overall score
        """
        scores = []
        for rule_emb in self.rule_embeddings:
            if isinstance(text_embedding, torch.Tensor):
                # Cosine similarity between text and rule
                sim = F.cosine_similarity(
                    text_embedding.flatten()[:len(rule_emb)],
                    rule_emb,
                    dim=0
                )
                scores.append(sim.item())
            else:
                scores.append(1.0)

        return {
            "rule_compliance": scores,
            "overall_compliance": sum(scores) / len(scores) if scores else 1.0,
            "rules": self.rules,
        }


class RewardModel(nn.Module):
    """Online RLHF reward model for preference learning."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.preference_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=16,
                batch_first=True, dropout=0.1
            ),
            num_layers=3
        )

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, hidden_states, response_length=None):
        """
        Compute reward score.

        Returns:
            reward: [B, 1]
        """
        encoded = self.preference_encoder(hidden_states)
        pooled = encoded.mean(dim=1)
        reward = self.reward_head(pooled)
        return reward


class SafetyAlignmentLayer(nn.Module):
    """
    Main safety alignment layer with full intervention support.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not config.safety.enabled:
            self.harm_classifier = None
            self.value_head = None
            self.rule_engine = None
            self.reward_model = None
            return

        self.harm_classifier = HarmfulnessClassifier(config)
        self.value_head = ValueAlignmentHead(config)
        self.rule_engine = ConstitutionalRuleEngine(config)
        self.reward_model = RewardModel(config)

        # Intervention threshold
        self.intervention_threshold = config.safety.harm_threshold

    def forward(self, hidden_states, text_embedding=None):
        """
        Full safety forward pass.

        Returns:
            dict with safety, values, constitutional, and reward info
        """
        outputs = {}

        if self.harm_classifier is not None:
            outputs["safety"] = self.harm_classifier(hidden_states)

        if self.value_head is not None:
            outputs["values"] = self.value_head(hidden_states)

        if text_embedding is not None and self.rule_engine is not None:
            outputs["constitutional"] = self.rule_engine.evaluate(text_embedding)

        return outputs

    def compute_reward(self, hidden_states):
        """Compute RLHF reward."""
        if self.reward_model is None:
            return torch.zeros(hidden_states.size(0), 1)
        return self.reward_model(hidden_states)

    def should_intervene(self, hidden_states):
        """
        Check if safety intervention is needed.

        Returns:
            bool: True if harmful content detected
        """
        if self.harm_classifier is None:
            return False

        safety_out = self.harm_classifier(hidden_states)
        return safety_out["is_harmful"].any().item()
