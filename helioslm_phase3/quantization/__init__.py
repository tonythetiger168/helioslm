"""HeliosLM Phase 3 — Model Quantization

Production quantization for reduced memory and faster inference:
  - FP8 (H100 native, 2x memory reduction)
  - INT8 (smoothquant, 2x memory reduction)
  - AWQ (Activation-aware Weight Quantization, 4-bit)
  - GPTQ (post-training quantization, 4-bit)
  - Dynamic quantization for on-the-fly compression
"""
from .fp8_quantizer import FP8Quantizer, convert_to_fp8
from .smoothquant import SmoothQuantConverter
from .awq_quantizer import AWQQuantizer
from .gptq_quantizer import GPTQQuantizer
from .dynamic_quant import DynamicQuantizer

__all__ = [
    "FP8Quantizer", "convert_to_fp8",
    "SmoothQuantConverter",
    "AWQQuantizer",
    "GPTQQuantizer",
    "DynamicQuantizer",
]
