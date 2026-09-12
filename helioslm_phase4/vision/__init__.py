"""HeliosLM Phase 4 — Vision Module

Image understanding with Vision Transformer (ViT) encoder.
Supports: image classification, captioning, VQA, document OCR.
"""
from .vision_encoder import VisionEncoder, PatchEmbedding
from .image_processor import ImageProcessor

__all__ = ["VisionEncoder", "PatchEmbedding", "ImageProcessor"]
