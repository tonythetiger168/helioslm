"""Native Multimodal Fusion - P5: Vision + Audio + Cross-Modal Attention"""
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
    """P5: ViT-style vision encoder, supports 224x224 to 1024x1024"""
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=mm.vision_num_heads,
            dim_feedforward=mm.vision_mlp_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = RMSNorm(self.hidden_size)

    def forward(self, images):
        B = images.shape[0]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)


class AudioEncoder(nn.Module):
    """P5: Conformer-based audio encoder, supports 16kHz-48kHz"""
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
        x = self.mel_proj(mel_specs)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x)


class CrossModalAttention(nn.Module):
    """P5: Cross-modal fusion attention (text <-> vision <-> audio)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.multimodal.cross_modal_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.modality_gates = nn.Parameter(torch.zeros(3))
        self.norm = RMSNorm(self.hidden_size)

    def forward(self, text_states, vision_states=None, audio_states=None, attention_mask=None):
        modalities = [text_states]
        if vision_states is not None:
            modalities.append(vision_states)
        if audio_states is not None:
            modalities.append(audio_states)
        q = self.q_proj(text_states)
        all_kv = []
        for m in modalities:
            all_kv.append((self.k_proj(m), self.v_proj(m)))
        B, T, H = q.shape
        q = q.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)
        outputs = []
        gates = torch.softmax(self.modality_gates[:len(modalities)], dim=0)
        for gate, (k, v) in zip(gates, all_kv):
            k = k.view(B, -1, self.num_heads, H // self.num_heads).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, H // self.num_heads).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(H // self.num_heads)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            outputs.append(gate * out)
        fused = sum(outputs)
        fused = fused.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(fused)


class MultimodalProjector(nn.Module):
    """P5: Unified projection to language model hidden space"""
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
    """P5: Main multimodal fusion module"""
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

    def forward(self, text_states, images=None, audio=None):
        vision_feat = None
        audio_feat = None
        if images is not None and self.vision_encoder is not None:
            vision_feat = self.vision_encoder(images)
        if audio is not None and self.audio_encoder is not None:
            audio_feat = self.audio_encoder(audio)
        if vision_feat is None and audio_feat is None:
            return text_states
        projected = self.projector(vision_feat, audio_feat)
        return self.cross_attn(text_states, projected, None)
