
"""
Kimi K3+ 核心架構實現
基於 Kimi K3 框架的增強版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

# ============================================
# 1. Kimi Delta Attention (KDA) + Gated MLA
# ============================================

class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention: 3:1 比例交替線性注意力與全局注意力
    無需位置編碼 (NoPE) 即可外推至 1M tokens
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kda_ratio = config.attention.kda_ratio  # "3:1"

        # Gated Multi-Layer Attention (Gated MLA)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Gating mechanism for MLA
        self.gate = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Linear attention kernel transformation
        self.linear_q_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_k_kernel = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Attention Residuals (AttnRes)
        self.attn_res_enabled = config.attention.attention_residuals.enabled
        self.residual_depth = config.attention.attention_residuals.residual_depth
        if self.attn_res_enabled:
            self.residual_gates = nn.Parameter(torch.zeros(config.num_hidden_layers, self.residual_depth))
            self.residual_proj = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(self.residual_depth)
            ])

    def linear_attention(self, q, k, v, mask=None):
        """線性複雜度注意力 (O(n) instead of O(n²))"""
        # Apply feature map (elu+1 for positive values)
        q = F.elu(self.linear_q_kernel(q)) + 1
        k = F.elu(self.linear_k_kernel(k)) + 1

        # KV state accumulation
        kv_state = torch.einsum('bhsk,bhsv->bhkv', k, v)

        # Compute output
        z = torch.einsum('bhqd,bhkd->bhq', q, k.sum(dim=2)) + 1e-6
        out = torch.einsum('bhqd,bhkd,bhkv->bhqv', q, k, kv_state) / z.unsqueeze(-1)

        return out

    def global_attention(self, q, k, v, mask=None):
        """標準全局注意力 (O(n²))"""
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkv->bhqv', attn_weights, v)
        return out

    def forward(self, hidden_states, layer_idx, past_residuals=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Gating for MLA
        gate_values = torch.sigmoid(self.gate(hidden_states))
        gate_values = gate_values.unsqueeze(-1).transpose(1, 2)  # [batch, num_heads, seq_len, 1]

        # Determine attention type based on KDA ratio (3:1)
        # Every 4th layer uses global attention, others use linear
        use_global = (layer_idx % 4 == 0)

        if use_global:
            attn_output = self.global_attention(q, k, v, attention_mask)
        else:
            attn_output = self.linear_attention(q, k, v, attention_mask)

        # Apply gating
        attn_output = attn_output * gate_values

        # Attention Residuals (AttnRes)
        if self.attn_res_enabled and past_residuals is not None:
            residual_output = torch.zeros_like(attn_output)
            for i, (residual, proj) in enumerate(zip(past_residuals[-self.residual_depth:], self.residual_proj)):
                gate = torch.sigmoid(self.residual_gates[layer_idx, i])
                residual_output += gate * proj(residual).view_as(attn_output)
            attn_output = attn_output + residual_output

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, hidden_states  # Return hidden_states as residual for next layers


# ============================================
# 2. Stable LatentMoE with Dynamic Sparsity
# ============================================

class StableLatentMoE(nn.Module):
    """
    穩定潛在混合專家模型
    - 896 個專家，每 token 激活 16 個（可動態調整 8~16）
    - 領域專家分群（程式/數學/科學/創意）
    - SiTU-GLU 激活函數
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.num_shared_experts = config.moe.num_shared_experts
        self.top_k = config.moe.num_activated_experts
        self.min_experts = config.moe.dynamic_sparsity.min_experts
        self.max_experts = config.moe.dynamic_sparsity.max_experts
        self.expert_hidden_size = config.moe.expert_hidden_size

        # Router network
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Expert networks (each is a feed-forward network)
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(self.num_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            self._create_expert() for _ in range(self.num_shared_experts)
        ])

        # Expert grouping for domain specialization
        self.expert_groups = config.moe.expert_grouping.groups

        # Load balancing loss coefficient
        self.load_balance_loss_coef = config.moe.load_balancing.loss_coef

        # Dynamic sparsity estimator
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _create_expert(self):
        """創建單個專家網絡 (SiTU-GLU activation)"""
        return nn.ModuleDict({
            'gate_proj': nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False),
            'up_proj': nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False),
            'down_proj': nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False),
        })

    def situ_glu(self, x, expert):
        """SiTU-GLU: Sigmoid-Tanh Unit Gated Linear Unit"""
        gate = torch.sigmoid(expert['gate_proj'](x))
        up = torch.tanh(expert['up_proj'](x))
        return expert['down_proj'](gate * up)

    def forward(self, hidden_states, task_type=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Dynamic sparsity: estimate task difficulty
        difficulty = self.difficulty_estimator(hidden_states_flat).squeeze(-1)
        dynamic_k = self.min_experts + (self.max_experts - self.min_experts) * difficulty
        dynamic_k = dynamic_k.round().long().clamp(self.min_experts, self.max_experts)

        # Routing
        router_logits = self.router(hidden_states_flat)

        # Apply domain bias if task_type is known
        if task_type is not None:
            for group in self.expert_groups:
                if group['specialty'] == task_type:
                    start, end = map(int, group['expert_indices'].split('-'))
                    router_logits[:, start:end+1] += 2.0  # Boost domain experts

        # Top-k routing with load balancing
        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_k_values, dim=-1)

        # Compute load balancing loss
        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = self.num_experts * (
            router_prob.mean(dim=0) * router_prob.mean(dim=0)
        ).sum() * self.load_balance_loss_coef

        # Compute expert outputs
        output = torch.zeros_like(hidden_states_flat)

        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = routing_weights[:, i:i+1]

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.situ_glu(expert_input, self.experts[expert_id])
                    output[mask] += expert_weights[mask] * expert_output

        # Add shared experts (always active)
        for shared_expert in self.shared_experts:
            output += self.situ_glu(hidden_states_flat, shared_expert)

        output = output.view(batch_size, seq_len, hidden_size)
        return output, aux_loss


# ============================================
# 3. 多模態編碼器
# ============================================

class MultimodalEncoder(nn.Module):
    """
    統一多模態編碼器：文字 / 圖像 / 影片 / 音訊
    """
    def __init__(self, config):
        super().__init__()
        self.text_embed = nn.Embedding(config.architecture.vocab_size, config.hidden_size)

        # Vision encoder (ViT)
        if config.multimodal.enabled:
            self.vision_encoder = ViTEncoder(config.multimodal.vision_encoder)
            self.video_encoder = TimeSformerEncoder(config.multimodal.video_encoder)
            self.audio_encoder = WhisperEncoder(config.multimodal.audio_encoder)

            # Cross-modal projection
            self.vision_proj = nn.Linear(config.multimodal.vision_encoder.hidden_size, config.hidden_size)
            self.video_proj = nn.Linear(config.multimodal.vision_encoder.hidden_size, config.hidden_size)
            self.audio_proj = nn.Linear(config.multimodal.audio_encoder.hidden_size, config.hidden_size)

    def forward(self, input_ids=None, images=None, videos=None, audio=None):
        embeddings = []

        if input_ids is not None:
            text_emb = self.text_embed(input_ids)
            embeddings.append(text_emb)

        if images is not None:
            vision_emb = self.vision_encoder(images)
            vision_emb = self.vision_proj(vision_emb)
            embeddings.append(vision_emb)

        if videos is not None:
            video_emb = self.video_encoder(videos)
            video_emb = self.video_proj(video_emb)
            embeddings.append(video_emb)

        if audio is not None:
            audio_emb = self.audio_encoder(audio)
            audio_emb = self.audio_proj(audio_emb)
            embeddings.append(audio_emb)

        # Concatenate all modalities
        return torch.cat(embeddings, dim=1)


# ============================================
# 4. 推測解碼（Speculative Decoding）
# ============================================

class SpeculativeDecoder:
    """
    推測解碼加速：使用 30B 草稿模型預測，K3+ 驗證
    """
    def __init__(self, target_model, draft_model, max_draft_tokens=5):
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_tokens = max_draft_tokens

    def generate(self, input_ids, max_new_tokens):
        generated = input_ids.clone()

        while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            # Draft model generates candidate tokens
            draft_tokens = self.draft_model.generate(
                generated, 
                max_new_tokens=min(self.max_draft_tokens, 
                    input_ids.shape[1] + max_new_tokens - generated.shape[1])
            )

            # Target model verifies in parallel
            with torch.no_grad():
                target_logits = self.target_model(draft_tokens).logits

            # Accept/reject draft tokens
            accepted = 0
            for i in range(draft_tokens.shape[1] - generated.shape[1]):
                pos = generated.shape[1] + i
                draft_token = draft_tokens[0, pos]
                target_token = target_logits[0, pos].argmax()

                if draft_token == target_token:
                    generated = torch.cat([generated, draft_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    accepted += 1
                else:
                    generated = torch.cat([generated, target_token.unsqueeze(0).unsqueeze(0)], dim=1)
                    break

            if accepted == 0:
                # Fallback to target model single token
                next_token = target_logits[0, -1].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        return generated


# ============================================
# 5. 長期記憶機制
# ============================================

class LongTermMemory:
    """
    外部長期記憶：向量資料庫 + 壓縮摘要
    """
    def __init__(self, config):
        self.embedding_dim = config.agent.long_term_memory.memory_bank.embedding_dim
        self.max_memories = config.agent.long_term_memory.memory_bank.max_memories
        self.compression_ratio = config.agent.long_term_memory.memory_bank.compression_ratio

        # FAISS vector store
        import faiss
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.memories = []

    def add(self, key, value, importance_score=1.0):
        """添加記憶"""
        if len(self.memories) >= self.max_memories:
            # Remove least important memory
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i]['importance'])
            self.memories.pop(min_idx)

        # Compress if too long
        if len(value) > 1000:
            value = self._compress(value)

        self.memories.append({
            'key': key,
            'value': value,
            'importance': importance_score,
            'timestamp': time.time()
        })

        # Add to FAISS
        key_vector = key.detach().cpu().numpy()
        self.index.add(key_vector.reshape(1, -1))

    def retrieve(self, query, top_k=5):
        """檢索相關記憶"""
        query_vector = query.detach().cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.memories):
                results.append(self.memories[idx])

        return results

    def _compress(self, text):
        """壓縮長文本為摘要"""
        # Use internal summarization
        # Simplified: extract first and last sentences
        sentences = text.split('.')
        if len(sentences) > 3:
            return '.'.join(sentences[:2]) + '...' + '.'.join(sentences[-2:])
        return text


# ============================================
# 6. 主模型：KimiK3Plus
# ============================================

class KimiK3Plus(nn.Module):
    """
    Kimi K3+ 主模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = MultimodalEncoder(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            KimiK3PlusLayer(config, layer_idx)
            for layer_idx in range(config.architecture.num_hidden_layers)
        ])

        # Norm and output
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.architecture.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.architecture.vocab_size, bias=False)

        # Reasoning mode
        self.always_on_thinking = config.reasoning.always_on_thinking

        # Long-term memory
        if config.agent.long_term_memory.enabled:
            self.memory = LongTermMemory(config)

    def forward(self, input_ids=None, images=None, videos=None, audio=None, 
                attention_mask=None, use_reasoning=True):
        # Encode inputs
        hidden_states = self.embed_tokens(input_ids, images, videos, audio)

        # Pass through layers
        past_residuals = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, layer_idx, past_residuals, attention_mask)
            past_residuals.append(residual)
            if len(past_residuals) > 4:  # Keep last 4 for AttnRes
                past_residuals.pop(0)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


class KimiK3PlusLayer(nn.Module):
    """單個 Transformer 層：Attention + MoE + AttnRes"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = KimiDeltaAttention(config)
        self.moe = StableLatentMoE(config)
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.architecture.rms_norm_eps)
        self.post_attention_norm = nn.RMSNorm(config.hidden_size, eps=config.architecture.rms_norm_eps)

    def forward(self, hidden_states, layer_idx, past_residuals, attention_mask):
        # Pre-norm attention
        normed = self.input_norm(hidden_states)
        attn_out, residual = self.attention(normed, layer_idx, past_residuals, attention_mask)
        hidden_states = hidden_states + attn_out

        # Pre-norm MoE
        normed = self.post_attention_norm(hidden_states)
        moe_out, aux_loss = self.moe(normed)
        hidden_states = hidden_states + moe_out

        return hidden_states, residual
