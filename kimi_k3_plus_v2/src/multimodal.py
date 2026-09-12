"""
Unified Multimodal Encoder - Phase 2
文字 / 图像 / 视频 / 音频 统一编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTGiantEncoder(nn.Module):
    """ViT-Giant 视觉编码器"""
    def __init__(self, config):
        super().__init__()
        self.image_size = config.multimodal.vision_image_size
        self.patch_size = config.multimodal.vision_patch_size
        self.hidden_size = config.multimodal.vision_hidden_size
        num_patches = (self.image_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=16, dim_feedforward=6144, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=24)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, images):
        B = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x)


class VideoEncoder(nn.Module):
    """视频编码器 - 时序感知"""
    def __init__(self, config):
        super().__init__()
        self.temporal_resolution = config.multimodal.video_temporal_resolution
        self.frame_sample_rate = config.multimodal.video_frame_sample_rate
        self.motion_aware = config.multimodal.video_motion_aware

        self.frame_encoder = ViTGiantEncoder(config)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.multimodal.vision_hidden_size,
                nhead=16, dim_feedforward=6144, batch_first=True
            ),
            num_layers=8
        )

        if self.motion_aware:
            self.motion_head = nn.Linear(config.multimodal.vision_hidden_size, config.multimodal.vision_hidden_size)

    def forward(self, videos):
        """
        videos: [B, T, C, H, W]
        """
        B, T, C, H, W = videos.shape
        frames = videos.view(B * T, C, H, W)
        frame_features = self.frame_encoder(frames)[:, 0]  # [B*T, hidden]
        frame_features = frame_features.view(B, T, -1)

        # 时序建模
        temporal_out = self.temporal_transformer(frame_features)

        if self.motion_aware:
            motion = temporal_out[:, 1:] - temporal_out[:, :-1]
            motion_feat = self.motion_head(motion.mean(dim=1))
            temporal_out = temporal_out + motion_feat.unsqueeze(1)

        return temporal_out


class AudioEncoder(nn.Module):
    """音频编码器 - 支持情感与说话人识别"""
    def __init__(self, config):
        super().__init__()
        self.n_mels = config.multimodal.audio_n_mels
        self.enable_emotion = config.multimodal.audio_enable_emotion
        self.enable_speaker = config.multimodal.audio_enable_speaker_id

        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.n_mels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, batch_first=True),
            num_layers=12
        )

        if self.enable_emotion:
            self.emotion_head = nn.Linear(768, 8)  # 8种情感
        if self.enable_speaker:
            self.speaker_head = nn.Linear(768, 256)  # 说话人嵌入

    def forward(self, audio_features):
        """
        audio_features: [B, n_mels, T]
        """
        x = self.conv_layers(audio_features).transpose(1, 2)  # [B, T, 768]
        x = self.transformer(x)

        outputs = {"features": x}
        if self.enable_emotion:
            outputs["emotion"] = self.emotion_head(x.mean(dim=1))
        if self.enable_speaker:
            outputs["speaker"] = self.speaker_head(x.mean(dim=1))

        return outputs


class UnifiedMultimodalEncoder(nn.Module):
    """统一多模态编码器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.multimodal.enabled:
            self.vision_encoder = ViTGiantEncoder(config)
            self.video_encoder = VideoEncoder(config)
            self.audio_encoder = AudioEncoder(config)

            # 投影到统一维度
            self.vision_proj = nn.Linear(config.multimodal.vision_hidden_size, config.hidden_size)
            self.video_proj = nn.Linear(config.multimodal.vision_hidden_size, config.hidden_size)
            self.audio_proj = nn.Linear(768, config.hidden_size)

            # 跨模态对齐
            self.cross_modal_temp = config.multimodal.cross_modal_temperature

    def forward(self, input_ids=None, images=None, videos=None, audio=None):
        embeddings = []

        if input_ids is not None:
            embeddings.append(self.text_embed(input_ids))

        if images is not None and self.config.multimodal.enabled:
            vision_emb = self.vision_encoder(images)
            vision_emb = self.vision_proj(vision_emb)
            embeddings.append(vision_emb)

        if videos is not None and self.config.multimodal.enabled:
            video_emb = self.video_encoder(videos)
            video_emb = self.video_proj(video_emb)
            embeddings.append(video_emb)

        if audio is not None and self.config.multimodal.enabled:
            audio_out = self.audio_encoder(audio)
            audio_emb = self.audio_proj(audio_out["features"])
            embeddings.append(audio_emb)

        return torch.cat(embeddings, dim=1) if embeddings else None
