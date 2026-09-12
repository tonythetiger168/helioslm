"""Long context extension: 4K → 32K → 128K → 1M via YaRN/NTK

Phase 2 Strategy:
  Stage 1: Pre-train at 4K context
  Stage 2: Continue pre-train at 32K (adjust RoPE base)
  Stage 3: Continue pre-train at 128K (YaRN)
  Stage 4: Continue pre-train at 1M (NTK-aware + YaRN)

Reference: "YaRN: Efficient Context Window Extension of Large Language Models"
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class YaRNConfig:
    """YaRN configuration for context extension."""
    original_max_position: int = 4096
    target_max_position: int = 32768
    scale_factor: float = 8.0  # target / original
    beta_slow: float = 0.25
    beta_fast: float = 0.1
    mscale: float = 1.0  # Attention temperature scaling


class YaRNRoPE:
    """
    YaRN (Yet another RoPE extensioN) implementation.

    Key idea: Interpolate RoPE frequencies with NTK-aware scaling
    and add attention temperature scaling to maintain stability.
    """

    def __init__(self, config: YaRNConfig, head_dim: int, base: float = 10000.0):
        self.config = config
        self.head_dim = head_dim
        self.base = base

        # Compute frequency scaling
        self._compute_scaling()

    def _compute_scaling(self):
        """Compute YaRN frequency scaling factors."""
        dim = self.head_dim
        beta_fast = self.config.beta_fast
        beta_slow = self.config.beta_slow
        scale = self.config.scale_factor

        # NTK-aware scaling: scale different dimensions differently
        # High-frequency dims (small indices) get less scaling
        # Low-frequency dims (large indices) get more scaling
        freqs = torch.arange(0, dim, 2).float()

        # YaRN interpolation
        # For dimensions where wavelength < beta_fast: no scaling
        # For dimensions where wavelength > beta_slow: full scaling
        # In between: linear interpolation
        wavelengths = 2 * math.pi * (self.base ** (freqs / dim))

        # Compute per-dimension scale
        ramp = torch.clamp(
            (wavelengths - beta_fast) / (beta_slow - beta_fast),
            min=0.0, max=1.0,
        )

        # Mix between no scaling and full scaling
        self.scale_factors = 1.0 + ramp * (scale - 1.0)

        # Attention temperature scaling (mscale)
        # Prevents attention scores from becoming too sharp at long contexts
        self.attn_scale = self.config.mscale * math.sqrt(
            1 - math.log(scale) / math.log(self.config.target_max_position)
        ) if scale > 1 else 1.0

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> tuple:
        """Generate cos/sin embeddings for given sequence length."""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Apply per-dimension scaling
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        inv_freq = inv_freq / self.scale_factors

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        return cos, sin, self.attn_scale


class LongContextTrainer:
    """
    Progressive context length extension trainer.

    Usage:
        trainer = LongContextTrainer(model, config)

        # Stage 1: 4K pre-training
        trainer.set_context_length(4096)
        trainer.train(dataloader_4k, steps=100000)

        # Stage 2: 32K extension
        trainer.extend_context(32768, yarn_config)
        trainer.train(dataloader_32k, steps=20000)

        # Stage 3: 128K extension
        trainer.extend_context(131072, yarn_config)
        trainer.train(dataloader_128k, steps=10000)
    """

    def __init__(self, model, base_config):
        self.model = model
        self.base_config = base_config
        self.current_max_position = base_config.max_position_embeddings
        self.yarn_config = None

    def set_context_length(self, seq_len: int):
        """Set the current training context length."""
        self.current_max_position = seq_len
        # Update model config
        self.model.config.max_position_embeddings = seq_len
        print(f"📏 Context length set to {seq_len:,}")

    def extend_context(self, target_seq_len: int, yarn_config: Optional[YaRNConfig] = None):
        """
        Extend model context length using YaRN.

        This adjusts the RoPE base and optionally applies YaRN scaling.
        """
        scale = target_seq_len / self.current_max_position

        if yarn_config is None:
            yarn_config = YaRNConfig(
                original_max_position=self.current_max_position,
                target_max_position=target_seq_len,
                scale_factor=scale,
            )

        self.yarn_config = yarn_config

        # Update RoPE in attention layers
        for layer in self.model.layers:
            attn = layer.attention
            attn.rope_theta = attn.rope_theta * scale

            # If using YaRN, replace standard RoPE with YaRN RoPE
            if scale > 2.0:
                attn.yarn_rope = YaRNRoPE(
                    yarn_config,
                    head_dim=attn.head_dim,
                    base=attn.rope_theta,
                )
                print(f"🧶 YaRN applied: scale={scale:.1f}, attn_scale={attn.yarn_rope.attn_scale:.3f}")

        self.set_context_length(target_seq_len)

    def prepare_data_for_length(
        self,
        data_iterator,
        tokenizer,
        seq_len: int,
    ):
        """Prepare data loader for specific context length."""
        from ..data_pipeline.streaming_dataset import StreamingPretrainDataset

        dataset = StreamingPretrainDataset(
            data_iterator=data_iterator,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pack_sequences=True,
        )

        return dataset
