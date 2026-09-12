"""
Kimi K3+ Main Model
"""

import torch
import torch.nn as nn
from .attention import KimiDeltaAttention, RMSNorm
from .moe import StableLatentMoE


class KimiK3PlusLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = KimiDeltaAttention(config)
        self.moe = StableLatentMoE(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, past_residuals=None, attention_mask=None):
        attn_out, residual = self.attention(hidden_states, self.layer_idx, past_residuals, attention_mask)
        hidden_states = hidden_states + attn_out
        normed = self.post_attention_norm(hidden_states)
        moe_out, aux_loss = self.moe(normed)
        hidden_states = hidden_states + moe_out
        return hidden_states, residual, aux_loss


class KimiK3Plus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            KimiK3PlusLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, task_type=None):
        hidden_states = self.embed_tokens(input_ids)
        past_residuals = []
        total_aux_loss = 0
        for layer in self.layers:
            hidden_states, residual, aux_loss = layer(hidden_states, past_residuals, attention_mask)
            past_residuals.append(residual)
            if len(past_residuals) > 4:
                past_residuals.pop(0)
            total_aux_loss += aux_loss
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, total_aux_loss
