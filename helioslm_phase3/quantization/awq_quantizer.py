"""AWQ: Activation-aware Weight Quantization (4-bit).

Reference: "AWQ: Activation-aware Weight Quantization for LLM Compression 
            and Acceleration" (Lin et al., 2023)

Key insight: Protect salient weight channels (important for activation 
magnitudes) by scaling them before quantization.
"""
import torch
import torch.nn as nn
from typing import Dict, List


class AWQQuantizer:
    """
    AWQ 4-bit quantizer.

    Protects salient channels by searching for optimal per-channel scales.
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.qmax = 2 ** (bits - 1) - 1  # 7 for 4-bit
        self.qmin = -(2 ** (bits - 1))   # -8 for 4-bit

    def find_salient_channels(self, weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """
        Find salient weight channels based on activation magnitudes.

        Channels with larger activation magnitudes are more important.
        """
        # Activation magnitude per channel (input dimension)
        act_magnitude = activation.abs().mean(dim=0)  # [in_features]

        # Weight magnitude per channel
        weight_magnitude = weight.abs().mean(dim=0)  # [in_features]

        # Combined saliency
        saliency = act_magnitude * weight_magnitude

        return saliency

    def search_scale(self, weight: torch.Tensor, activation: torch.Tensor, n_grid: int = 20) -> torch.Tensor:
        """
        Search for optimal per-channel scaling factors.

        Grid search over possible scales to minimize quantization error.
        """
        saliency = self.find_salient_channels(weight, activation)

        # Grid search
        best_scales = torch.ones(weight.shape[1], device=weight.device)
        best_error = float('inf')

        for s in torch.linspace(0.5, 1.5, n_grid):
            scale = torch.where(saliency > saliency.median(), s, 1.0)

            # Apply scale
            scaled_weight = weight * scale.unsqueeze(0)

            # Quantize
            w_quant = self._quantize(scaled_weight)
            w_dequant = self._dequantize(w_quant)

            # Error
            error = (weight - w_dequant / scale.unsqueeze(0)).abs().mean()

            if error < best_error:
                best_error = error
                best_scales = scale

        return best_scales

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Group-wise quantization."""
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, self.group_size)

        scale = tensor.abs().max(dim=1, keepdim=True).values / self.qmax
        scale = scale.clamp(min=1e-5)

        quantized = (tensor / scale).round().clamp(self.qmin, self.qmax)

        return quantized.reshape(original_shape), scale.reshape(original_shape[0], -1)

    def _dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize."""
        return quantized * scale

    def quantize_layer(self, layer: nn.Linear, activation: torch.Tensor) -> nn.Module:
        """Quantize a single layer with AWQ."""
        # Search optimal scales
        scales = self.search_scale(layer.weight, activation)

        # Apply scales to weights
        scaled_weight = layer.weight * scales.unsqueeze(0)

        # Quantize
        w_quant, w_scale = self._quantize(scaled_weight)

        # Store quantized weights
        layer.register_buffer("awq_weight", w_quant.to(torch.int8))
        layer.register_buffer("awq_scale", w_scale)
        layer.register_buffer("awq_channel_scale", scales)

        return layer
