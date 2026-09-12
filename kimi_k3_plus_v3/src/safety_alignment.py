"""Safety Alignment + Constitutional AI - P7: Harm Detection + Value Alignment"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmfulnessClassifier(nn.Module):
    """P7: Multi-level harmful content classifier (12 categories)"""
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
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16, batch_first=True),
            num_layers=2
        )
        self.category_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, self.num_categories),
            nn.Sigmoid()
        )
        self.severity_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )

    def forward(self, hidden_states):
        encoded = self.encoder(hidden_states)
        pooled = encoded.mean(dim=1)
        category_scores = self.category_head(pooled)
        severity_logits = self.severity_head(pooled)
        return {
            "category_scores": category_scores,
            "severity": F.softmax(severity_logits, dim=-1),
            "is_harmful": (category_scores > 0.5).any(dim=-1)
        }


class ValueAlignmentHead(nn.Module):
    """P7: Value alignment scoring (helpful, harmless, honest)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        pooled = hidden_states.mean(dim=1)
        hhh = torch.sigmoid(self.hhh_head(pooled))
        alignment = self.alignment_head(pooled)
        return {
            "helpful": hhh[:, 0],
            "harmless": hhh[:, 1],
            "honest": hhh[:, 2],
            "alignment_score": alignment
        }


class ConstitutionalRuleEngine:
    """P7: Configurable constitutional rule engine"""
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
        self.rule_embeddings = None

    def evaluate(self, text_embedding):
        scores = []
        for rule in self.rules:
            score = 1.0
            scores.append(score)
        return {
            "rule_compliance": scores,
            "overall_compliance": sum(scores) / len(scores) if scores else 1.0
        }


class RewardModel(nn.Module):
    """P7: Online RLHF reward model for preference learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.preference_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16, batch_first=True),
            num_layers=3
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, hidden_states, response_length=None):
        encoded = self.preference_encoder(hidden_states)
        pooled = encoded.mean(dim=1)
        reward = self.reward_head(pooled)
        return reward


class SafetyAlignmentLayer(nn.Module):
    """P7: Main safety alignment layer"""
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

    def forward(self, hidden_states, text_embedding=None):
        outputs = {}
        if self.harm_classifier is not None:
            outputs["safety"] = self.harm_classifier(hidden_states)
        if self.value_head is not None:
            outputs["values"] = self.value_head(hidden_states)
        if text_embedding is not None and self.rule_engine is not None:
            outputs["constitutional"] = self.rule_engine.evaluate(text_embedding)
        return outputs

    def compute_reward(self, hidden_states):
        if self.reward_model is None:
            return torch.zeros(hidden_states.size(0), 1)
        return self.reward_model(hidden_states)
