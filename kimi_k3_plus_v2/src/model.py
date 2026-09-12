"""
Kimi K3+ Unified Architecture
一个架构，四个尺寸
"""

import torch
import torch.nn as nn
from .attention import KimiDeltaAttention, RMSNorm
from .moe import StableLatentMoE
from .multimodal import UnifiedMultimodalEncoder
from .speculative_decoding import SpeculativeDecoder, DraftModel
from .agentic import AgenticLayer
from .memory import LongTermMemory


class KimiK3PlusLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = KimiDeltaAttention(config)
        self.moe = StableLatentMoE(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, past_residuals=None, attention_mask=None, task_type=None):
        attn_out, residual = self.attention(hidden_states, self.layer_idx, past_residuals, attention_mask)
        hidden_states = hidden_states + attn_out
        normed = self.post_attention_norm(hidden_states)
        moe_out, aux_loss = self.moe(normed, task_type=task_type)
        hidden_states = hidden_states + moe_out
        return hidden_states, residual, aux_loss


class KimiK3Plus(nn.Module):
    """统一架构 - 支持四个尺寸配置"""
    def __init__(self, config, size="ultra"):
        super().__init__()
        self.config = config
        self.size = size

        # 多模态编码器
        self.embed_tokens = UnifiedMultimodalEncoder(config)

        # Transformer 层
        self.layers = nn.ModuleList([
            KimiK3PlusLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # 输出层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Phase 1: 推测解码
        if config.speculative.enabled and size in ["ultra", "pro", "lite"]:
            self.draft_model = DraftModel(config)
            self.speculative_decoder = SpeculativeDecoder(self, self.draft_model, config)
        else:
            self.speculative_decoder = None

        # Phase 3: Agentic
        if config.agentic.enabled:
            self.agentic_layer = AgenticLayer(config)
            self.memory = LongTermMemory(config)
        else:
            self.agentic_layer = None
            self.memory = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, images=None, videos=None, audio=None,
                attention_mask=None, task_type=None, use_cache=False):
        # 多模态编码
        hidden_states = self.embed_tokens(input_ids, images, videos, audio)
        if hidden_states is None:
            raise ValueError("At least one input modality must be provided")

        past_residuals = []
        total_aux_loss = 0

        for layer in self.layers:
            hidden_states, residual, aux_loss = layer(
                hidden_states, past_residuals, attention_mask, task_type
            )
            past_residuals.append(residual)
            if len(past_residuals) > 4:
                past_residuals.pop(0)
            total_aux_loss += aux_loss

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, total_aux_loss

    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, 
                 use_speculative=True, task_type=None):
        """生成文本 - 支持推测解码"""
        if use_speculative and self.speculative_decoder is not None:
            return self.speculative_decoder.generate(input_ids, max_new_tokens, temperature)[0]

        # 标准自回归生成
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(input_ids, task_type=task_type)
                next_token = torch.multinomial(
                    torch.softmax(logits[:, -1, :] / temperature, dim=-1), 
                    num_samples=1
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def agentic_generate(self, goal, max_steps=10):
        """Agentic 生成 - 支持工具调用和长期记忆"""
        if self.agentic_layer is None:
            raise RuntimeError("Agentic layer not enabled")

        # 检索相关记忆
        if self.memory is not None:
            relevant_memories = self.memory.retrieve(goal, top_k=5)

        # 任务规划
        plan = self.agentic_layer.planner.plan(goal)

        results = []
        for step in range(max_steps):
            # 执行子任务
            # ... (简化实现)
            pass

        return results
