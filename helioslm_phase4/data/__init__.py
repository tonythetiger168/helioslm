"""HeliosLM Phase 4 — Multimodal Pretraining Data Pipeline

All known limitations FIXED:
  ✅ Video scene detection (not just uniform sampling)
  ✅ Whisper word-level timestamp alignment
  ✅ CLIP-based cross-modal quality filtering
  ✅ Interleaved document formatting
"""
from .image_text_pipeline import ImageTextDataset, ImageTextMixer
from .video_pipeline import VideoFrameExtractor, VideoCaptionDataset
from .video_scene_detect import SceneDetector, SceneAwareFrameExtractor
from .audio_text_pipeline import AudioTextAligner, SpeechCaptionDataset
from .whisper_alignment import WhisperAligner, WhisperAudioTextAligner
from .interleaved_formatter import InterleavedDocumentFormatter
from .mm_quality_filter import MultimodalQualityFilter

__all__ = [
    "ImageTextDataset", "ImageTextMixer",
    "VideoFrameExtractor", "VideoCaptionDataset",
    "SceneDetector", "SceneAwareFrameExtractor",
    "AudioTextAligner", "SpeechCaptionDataset",
    "WhisperAligner", "WhisperAudioTextAligner",
    "InterleavedDocumentFormatter",
    "MultimodalQualityFilter",
]
