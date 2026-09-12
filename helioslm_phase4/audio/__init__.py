"""HeliosLM Phase 4 — Audio Module

Audio understanding with spectrogram + transformer encoder.
Supports: speech recognition, audio captioning, sound event detection.
"""
from .audio_encoder import AudioEncoder, SpectrogramExtractor
from .audio_processor import AudioProcessor

__all__ = ["AudioEncoder", "SpectrogramExtractor", "AudioProcessor"]
