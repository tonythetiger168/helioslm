"""Quality classifier model architecture."""
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class QualityDimensions:
    """Quality dimension scores."""
    grammar: float = 0.0      # 0-1, syntactic correctness
    knowledge: float = 0.0    # 0-1, information density
    coherence: float = 0.0    # 0-1, logical flow
    toxicity: float = 0.0     # 0-1, harmful content (inverted: higher = safer)
    diversity: float = 0.0    # 0-1, lexical diversity

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (
            0.25 * self.grammar +
            0.25 * self.knowledge +
            0.20 * self.coherence +
            0.15 * self.toxicity +
            0.15 * self.diversity
        )


class QualityClassifier(nn.Module):
    """
    Small transformer-based quality classifier (~0.5B params).

    Architecture:
      - 12 layers, 768 hidden, 12 heads
      - Shared encoder + 5 dimension-specific heads
      - Binary classification per dimension + regression for composite
    """

    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        num_dimensions: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_dimensions = num_dimensions

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Dimension-specific heads
        self.grammar_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.knowledge_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.coherence_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.toxicity_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.diversity_head = nn.Sequential(nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())

        # Composite score head (learned weighting)
        self.composite_head = nn.Sequential(
            nn.Linear(hidden_size * num_dimensions, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, seq = input_ids.shape

        # Embeddings
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)

        x = self.norm(x)

        # Pool (mean pooling with attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Dimension scores
        grammar = self.grammar_head(pooled)
        knowledge = self.knowledge_head(pooled)
        coherence = self.coherence_head(pooled)
        toxicity = self.toxicity_head(pooled)
        diversity = self.diversity_head(pooled)

        # Composite
        dim_features = torch.cat([grammar, knowledge, coherence, toxicity, diversity], dim=-1)
        composite = self.composite_head(dim_features)

        return {
            "grammar": grammar.squeeze(-1),
            "knowledge": knowledge.squeeze(-1),
            "coherence": coherence.squeeze(-1),
            "toxicity": toxicity.squeeze(-1),
            "diversity": diversity.squeeze(-1),
            "composite": composite.squeeze(-1),
        }

    def score_document(self, text: str, tokenizer) -> QualityDimensions:
        """Score a single document."""
        self.eval()
        tokens = tokenizer.encode(text, max_length=2048, truncation=True)
        input_ids = torch.tensor([tokens])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            scores = self.forward(input_ids, attention_mask)

        return QualityDimensions(
            grammar=scores["grammar"].item(),
            knowledge=scores["knowledge"].item(),
            coherence=scores["coherence"].item(),
            toxicity=scores["toxicity"].item(),
            diversity=scores["diversity"].item(),
        )
