"""HeliosLM v1.0.2 Unified Model — P0+P1+P2+P3 Production (KV Cache Integration)

P0 FIX (v1.0.1 → v1.0.2):
  - Integrated KVCache for O(L) autoregressive generation
  - Updated forward() to pass past_key_values through layers
  - generate() now uses KV cache for all decoding steps
  - Added tie_word_embeddings support
  - Removed multimodal stub parameters (images, audio) until P5 implementation
"""
import torch
import torch.nn as nn
from .attention import HeliosAttention, RMSNorm, KVCache
from .moe import StableLatentMoE
from .speculative_decoding import SpeculativeDecoder, DraftModel
from .rag import RAGModule
from .cot_compiler import CoTCompiler
from .agentic import AgenticLayer
from .memory import LongTermMemory
from .reasoning_budget import ReasoningBudgetController


class HeliosLMLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = HeliosAttention(config)
        self.moe = StableLatentMoE(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, past_residuals=None, attention_mask=None,
                task_type=None, past_key_value=None, use_cache=False):
        attn_out, residual, present_kv = self.attention(
            hidden_states, self.layer_idx, past_residuals, attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out
        moe_out, aux_loss = self.moe(self.post_attention_norm(hidden_states), task_type=task_type)
        hidden_states = hidden_states + moe_out
        return hidden_states, residual, aux_loss, present_kv


class HeliosLM(nn.Module):
    def __init__(self, config, size="ultra"):
        super().__init__()
        self.config = config
        self.size = size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            HeliosLMLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # P0: Speculative decoding
        if config.speculative.enabled and size != "nano":
            shared_embed = self.embed_tokens if config.tie_word_embeddings else None
            self.draft_model = DraftModel(config, shared_embedding=shared_embed)
            self.speculative_decoder = SpeculativeDecoder(self, self.draft_model, config)
        else:
            self.draft_model = None
            self.speculative_decoder = None

        # P1-P4 modules
        self.rag_module = RAGModule(config) if config.rag.enabled else None
        self.cot_compiler = CoTCompiler(config) if config.cot.enabled else None
        self.agentic_layer = AgenticLayer(config) if config.agentic.enabled else None
        self.memory = LongTermMemory(config) if config.agentic.enabled else None
        self.budget_controller = ReasoningBudgetController(config) if config.reasoning_budget.enabled else None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, task_type=None,
                past_key_values=None, use_cache=False, knowledge_index=None):
        hidden_states = self.embed_tokens(input_ids)
        past_residuals = []
        total_aux_loss = 0.0
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, residual, aux_loss, present_kv = layer(
                hidden_states, past_residuals, attention_mask, task_type,
                past_key_value=past_kv, use_cache=use_cache,
            )
            past_residuals.append(residual)
            if len(past_residuals) > 4:
                past_residuals.pop(0)
            total_aux_loss += aux_loss
            if present_key_values is not None:
                present_key_values.append(present_kv)

        # P1-P4 side modules (non-causal, applied to final hidden states)
        rag_info = self.rag_module(hidden_states, input_ids, knowledge_index=knowledge_index) if self.rag_module else None
        cot_info = self.cot_compiler.compile_reasoning(hidden_states) if self.cot_compiler else None
        agent_info = self.agentic_layer(hidden_states) if self.agentic_layer else None
        budget_info = None
        if self.budget_controller is not None:
            budget, task_diff = self.budget_controller.get_budget(hidden_states, total_token_budget=4096)
            budget_info = {"budget": budget, "difficulty": task_diff}

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "rag_info": rag_info,
            "cot_info": cot_info,
            "agent_info": agent_info,
            "budget_info": budget_info,
        }
        if present_key_values is not None:
            output["past_key_values"] = present_key_values
        return output

    def generate(self, input_ids, max_new_tokens=100, temperature=0.7,
                 use_speculative=True, task_type=None, use_cache=True):
        """Autoregressive generation with optional KV cache and speculative decoding."""
        if use_speculative and self.speculative_decoder is not None:
            return self.speculative_decoder.generate(input_ids, max_new_tokens, temperature)[0]

        self.eval()
        generated = input_ids.clone()
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(
                    generated,
                    task_type=task_type,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
                logits = out["logits"]
                past_key_values = out.get("past_key_values", None)

                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated
