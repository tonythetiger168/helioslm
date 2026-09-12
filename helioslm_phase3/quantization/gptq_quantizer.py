"""GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

Reference: "GPTQ: Accurate Post-Training Quantization for Generative 
            Pre-trained Transformers" (Frantar et al., 2023)

Key insight: Use approximate second-order information (OBS) to quantize
weights layer-by-layer with minimal error accumulation.
"""
import torch
import torch.nn as nn
from typing import Optional


class GPTQQuantizer:
    """
    GPTQ layer-wise quantization.

    Uses Hessian information to quantize weights with optimal compensation.
    """

    def __init__(self, bits: int = 4, group_size: int = 128, actorder: bool = True):
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder  # Reorder by activation magnitude
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -(2 ** (bits - 1))

    def quantize_layer(self, layer: nn.Linear, inputs: torch.Tensor) -> nn.Module:
        """
        Quantize a single layer using GPTQ.

        Args:
            layer: Linear layer to quantize
            inputs: Calibration inputs [batch, seq, in_features]
        """
        W = layer.weight.data.clone()  # [out_features, in_features]

        # Flatten inputs
        X = inputs.reshape(-1, inputs.shape[-1])  # [batch*seq, in_features]

        # Compute Hessian: H = X^T X
        H = torch.matmul(X.T, X)  # [in_features, in_features]
        H = H / X.shape[0]

        # Add damping
        damp = 0.01 * torch.diag(H).mean()
        H += damp * torch.eye(H.shape[0], device=H.device)

        # Cholesky decomposition for inverse
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except:
            # Fallback if not positive definite
            H_inv = torch.eye(H.shape[0], device=H.device)

        # Quantize column by column
        Q = torch.zeros_like(W)

        for i in range(W.shape[1]):  # For each input channel
            w_col = W[:, i]

            # Quantize
            scale = w_col.abs().max() / self.qmax
            if scale > 0:
                w_quant = (w_col / scale).round().clamp(self.qmin, self.qmax)
            else:
                w_quant = w_col

            Q[:, i] = w_quant * scale

            # Quantization error
            err = (w_col - Q[:, i]).unsqueeze(1)  # [out_features, 1]

            # Update remaining weights using Hessian information
            if i < W.shape[1] - 1:
                H_inv_diag = H_inv[i, i].clamp(min=1e-8)
                W[:, i+1:] -= err * (H_inv[i, i+1:] / H_inv_diag).unsqueeze(0)

        # Store quantized weights
        layer.register_buffer("gptq_weight", Q)

        return layer
