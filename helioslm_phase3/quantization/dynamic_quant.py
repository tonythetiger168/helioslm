"""Dynamic quantization for on-the-fly inference optimization.

Applies quantization dynamically during forward pass without 
pre-computing scales. Useful for quick deployment without calibration.
"""
import torch
import torch.nn as nn


class DynamicQuantizer:
    """
    Dynamic INT8 quantization.

    Quantizes activations on-the-fly during inference.
    """

    def __init__(self, dtype: torch.dtype = torch.qint8):
        self.dtype = dtype

    def quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamically quantize activations."""
        scale = x.abs().max() / 127.0
        if scale == 0:
            return x

        quantized = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def apply_to_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to a model."""
        # Use PyTorch's built-in dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return model
