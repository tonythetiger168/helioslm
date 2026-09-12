"""Multimodal fusion for vision + audio + text."""
import torch
import torch.nn as nn
from typing import Dict, Optional, List


class ModalityRouter(nn.Module):
    """
    Router that determines which modalities to process.

    Similar to MoE routing but for modalities.
    """

    def __init__(self, hidden_size: int, num_modalities: int = 3):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_modalities)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """Route based on text query intent."""
        # Use CLS token or pooled text feature
        pooled = text_features.mean(dim=1)  # [B, hidden]
        logits = self.router(pooled)  # [B, num_modalities]
        return torch.softmax(logits, dim=-1)


class MultimodalFusion(nn.Module):
    """
    Fuses vision, audio, and text features into unified representation.

    Architecture:
      1. Encode each modality independently
      2. Project to common space
      3. Cross-attention fusion
      4. Feed into language model
    """

    def __init__(
        self,
        lm_hidden_size: int = 12288,
        vision_embed_dim: int = 1024,
        audio_embed_dim: int = 768,
        num_fusion_layers: int = 4,
        num_heads: int = 16,
    ):
        super().__init__()
        self.lm_hidden_size = lm_hidden_size

        # Modality-specific projections
        self.vision_proj = nn.Linear(vision_embed_dim, lm_hidden_size)
        self.audio_proj = nn.Linear(audio_embed_dim, lm_hidden_size)

        # Modality router
        self.modality_router = ModalityRouter(lm_hidden_size)

        # Cross-attention fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=lm_hidden_size,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_fusion_layers)
        ])

        self.fusion_norm = nn.LayerNorm(lm_hidden_size)

        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(3, lm_hidden_size)  # 0=text, 1=vision, 2=audio

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse multimodal features.

        Args:
            text_features: [B, text_len, lm_hidden]
            vision_features: [B, num_patches, lm_hidden] or None
            audio_features: [B, audio_len, lm_hidden] or None

        Returns:
            fused: [B, total_len, lm_hidden]
        """
        features_list = [text_features]
        modality_ids = [torch.zeros(text_features.shape[:2], dtype=torch.long, device=text_features.device)]

        # Project and add vision features
        if vision_features is not None:
            vision_proj = self.vision_proj(vision_features)
            features_list.append(vision_proj)
            modality_ids.append(torch.ones(vision_features.shape[:2], dtype=torch.long, device=vision_features.device))

        # Project and add audio features
        if audio_features is not None:
            audio_proj = self.audio_proj(audio_features)
            features_list.append(audio_proj)
            modality_ids.append(torch.full(audio_features.shape[:2], 2, dtype=torch.long, device=audio_features.device))

        # Concatenate all features
        fused = torch.cat(features_list, dim=1)  # [B, total_len, hidden]
        modality_ids = torch.cat(modality_ids, dim=1)  # [B, total_len]

        # Add modality type embeddings
        fused = fused + self.modality_embeddings(modality_ids)

        # Cross-attention fusion
        for layer in self.fusion_layers:
            attn_out, _ = layer(fused, fused, fused)
            fused = self.fusion_norm(fused + attn_out)

        return fused
