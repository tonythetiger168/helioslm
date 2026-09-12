"""SmoothQuant: Accurate and Efficient Post-Training Quantization.

Reference: "SmoothQuant: Accurate and Efficient Post-Training Quantization 
            for Large Language Models" (Xiao et al., 2023)

Key insight: Migrate quantization difficulty from activations to weights
using a per-channel scaling factor.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class SmoothQuantConverter:
    """
    SmoothQuant INT8 converter.

    Formula: X' = X * s^(-1), W' = W * s
    where s is computed to balance activation and weight magnitudes.
    """

    def __init__(self, alpha: float = 0.5, migration_strength: float = 0.5):
        """
        Args:
            alpha: Smoothing strength (0 = all weight, 1 = all activation)
            migration_strength: How much to migrate difficulty to weights
        """
        self.alpha = alpha
        self.migration_strength = migration_strength

    def compute_smooth_scale(self, activations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute per-channel smooth scale.

        s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
        """
        # Activation max per channel
        act_max = activations.abs().max(dim=0).values

        # Weight max per channel
        weight_max = weights.abs().max(dim=0).values

        # Compute scale
        scale = (act_max ** self.alpha) / (weight_max ** (1 - self.alpha) + 1e-8)

        # Clamp to avoid extreme values
        scale = scale.clamp(min=1e-5, max=1e5)

        return scale

    def smooth_layer(self, layer: nn.Linear, calibration_data: torch.Tensor) -> tuple:
        """
        Smooth a single linear layer.

        Returns: (smoothed_layer, scale)
        """
        # Get activations
        with torch.no_grad():
            activations = calibration_data

        # Compute scale
        scale = self.compute_smooth_scale(activations, layer.weight)

        # Apply smoothing
        # X' = X * s^(-1)
        # W' = W * s
        smoothed_weight = layer.weight * scale.unsqueeze(0)

        # Create new layer with smoothed weights
        new_layer = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        new_layer.weight = nn.Parameter(smoothed_weight)
        if layer.bias is not None:
            new_layer.bias = nn.Parameter(layer.bias.clone())

        return new_layer, scale

    def convert_model(self, model: nn.Module, calibration_loader, device: str = "cuda") -> nn.Module:
        """
        Convert entire model using SmoothQuant.

        Args:
            model: Model to quantize
            calibration_loader: DataLoader with calibration data
            device: Device for computation
        """
        model.eval()

        # Collect activation statistics
        activation_stats = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                activation_stats[name] = input.detach()
            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run calibration
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    model(**batch)
                else:
                    batch = batch.to(device)
                    model(batch)
                break  # One batch is enough for stats

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Apply smoothing
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activation_stats:
                smoothed, scale = self.smooth_layer(module, activation_stats[name])
                # Replace module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, smoothed)
                else:
                    setattr(model, child_name, smoothed)

                # Store scale for inference
                module.register_buffer("smooth_scale", scale)

        # Quantize to INT8
        self._quantize_to_int8(model)

        return model

    def _quantize_to_int8(self, model: nn.Module):
        """Quantize smoothed weights to INT8."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Simple symmetric quantization
                w = module.weight
                scale = w.abs().max() / 127.0

                w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
                module.register_buffer("weight_int8", w_int8)
                module.register_buffer("quant_scale", torch.tensor(scale))
