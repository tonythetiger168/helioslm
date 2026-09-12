"""Native Multimodal Fusion v2 - P5 (Production-Ready)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class VisionEncoder(nn.Module):
    """ViT-style vision encoder with flexible resolution support."""

    def __init__(self, config):
        super().__init__()
        mm = config.multimodal
        self.patch_size = mm.vision_patch_size
        self.num_channels = mm.vision_channels
        self.hidden_size = mm.vision_hidden_size
        self.num_layers = mm.vision_num_layers
        self.image_size = mm.vision_image_size

        self.patch_embed = nn.Conv2d(
            self.num_channels, self.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )

        # Dynamic num_patches calculation
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=mm.vision_num_heads,
            dim_feedforward=mm.vision_mlp_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = RMSNorm(self.hidden_size)

    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]
        Returns:
            features: [B, num_patches+1, hidden]
        """
        B = images.shape[0]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        # Handle variable image sizes
        if x.shape[1] != self.num_patches:
            # Interpolate position embeddings for different resolutions
            import math
            h = w = int(math.sqrt(x.shape[1]))
            expected_h = expected_w = self.image_size // self.patch_size
            pos_embed = self.pos_embed[:, 1:, :].reshape(1, expected_h, expected_w, -1)
            pos_embed = F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(h, w), mode="bilinear", align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
            pos_embed = torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)
        return self.norm(x)


class AudioEncoder(nn.Module):
    """Conformer-based audio encoder."""

    def __init__(self, config):
        super().__init__()
        mm = config.multimodal
        self.n_mels = mm.audio_n_mels
        self.hidden_size = mm.audio_hidden_size
        self.num_layers = mm.audio_num_layers

        self.mel_proj = nn.Linear(self.n_mels, self.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3000, self.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=mm.audio_num_heads,
            dim_feedforward=mm.audio_mlp_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = RMSNorm(self.hidden_size)

    def forward(self, mel_specs):
        """
        Args:
            mel_specs: [B, T, n_mels]
        Returns:
            features: [B, T, hidden]
        """
        x = self.mel_proj(mel_specs)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x)


class CrossModalAttention(nn.Module):
    """Cross-modal fusion attention with learnable modality gates."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.multimodal.cross_modal_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Learnable modality gates
        self.modality_gates = nn.Parameter(torch.zeros(3))
        self.norm = RMSNorm(self.hidden_size)

    def forward(self, text_states, vision_states=None, audio_states=None, attention_mask=None):
        """
        Multi-modal cross attention.

        Args:
            text_states: [B, T, H]
            vision_states: [B, V, H] or None
            audio_states: [B, A, H] or None
        """
        modalities = [(text_states, 1.0)]
        if vision_states is not None:
            modalities.append((vision_states, 0.0))
        if audio_states is not None:
            modalities.append((audio_states, 0.0))

        q = self.q_proj(text_states)
        B, T, H = q.shape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        outputs = []
        gates = F.softmax(self.modality_gates[:len(modalities)], dim=0)

        for gate, (m, _) in zip(gates, modalities):
            k = self.k_proj(m)
            v = self.v_proj(m)
            k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            outputs.append(gate * out)

        fused = sum(outputs)
        fused = fused.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(fused)


class MultimodalProjector(nn.Module):
    """Project vision/audio features to language model hidden space."""

    def __init__(self, config):
        super().__init__()
        mm = config.multimodal
        self.vision_proj = nn.Linear(mm.vision_hidden_size, config.hidden_size)
        self.audio_proj = nn.Linear(mm.audio_hidden_size, config.hidden_size)
        self.fusion_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, vision_features=None, audio_features=None):
        outputs = []
        if vision_features is not None:
            outputs.append(self.vision_proj(vision_features))
        if audio_features is not None:
            outputs.append(self.audio_proj(audio_features))

        if len(outputs) == 0:
            return None
        if len(outputs) == 1:
            return outputs[0]

        concat = torch.cat(outputs, dim=-1)
        gate = torch.sigmoid(self.fusion_gate(concat))
        return gate * outputs[0] + (1 - gate) * outputs[1]


class MultimodalFusion(nn.Module):
    """
    Main multimodal fusion module.

    Note: In the full model, fusion happens at the input embedding layer.
    This module provides the encoders and projection for that integration.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not config.multimodal.enabled:
            self.vision_encoder = None
            self.audio_encoder = None
            self.cross_attn = None
            self.projector = None
            return

        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.cross_attn = CrossModalAttention(config)
        self.projector = MultimodalProjector(config)

    def encode_images(self, images):
        """Encode images to feature tokens."""
        if self.vision_encoder is None:
            return None
        return self.vision_encoder(images)

    def encode_audio(self, audio):
        """Encode audio to feature tokens."""
        if self.audio_encoder is None:
            return None
        return self.audio_encoder(audio)

    def forward(self, text_states, images=None, audio=None):
        """
        Legacy forward for standalone use.
        In full model, use encode_images/encode_audio directly.
        """
        vision_feat = self.encode_images(images) if images is not None else None
        audio_feat = self.encode_audio(audio) if audio is not None else None

        if vision_feat is None and audio_feat is None:
            return text_states

        projected = self.projector(vision_feat, audio_feat)
        return self.cross_attn(text_states, projected, None)
