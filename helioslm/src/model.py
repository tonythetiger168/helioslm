"""HeliosLM v1.0 Unified Model - P0+P1+P2+P3 Production"""
import torch
import torch.nn as nn
from .attention import HeliosAttention, RMSNorm
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

    def forward(self, hidden_states, past_residuals=None, attention_mask=None, task_type=None):
        attn_out, residual = self.attention(hidden_states, self.layer_idx, past_residuals, attention_mask)
        hidden_states = hidden_states + attn_out
        moe_out, aux_loss = self.moe(self.post_attention_norm(hidden_states), task_type=task_type)
        hidden_states = hidden_states + moe_out
        return hidden_states, residual, aux_loss


class HeliosLM(nn.Module):
    def __init__(self, config, size="ultra"):
        super().__init__()
        self.config = config
        self.size = size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HeliosLMLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # P0: Speculative decoding
        if config.speculative.enabled and size != "nano":
            self.draft_model = DraftModel(config)
            self.speculative_decoder = SpeculativeDecoder(self, self.draft_model, config)
        else:
            self.draft_model = None
            self.speculative_decoder = None

        # P1: RAG
        if config.rag.enabled:
            self.rag_module = RAGModule(config)
        else:
            self.rag_module = None

        # P2: CoT
        if config.cot.enabled:
            self.cot_compiler = CoTCompiler(config)
        else:
            self.cot_compiler = None

        # P3: Agentic
        if config.agentic.enabled:
            self.agentic_layer = AgenticLayer(config)
            self.memory = LongTermMemory(config)
        else:
            self.agentic_layer = None
            self.memory = None

        # P4: Budget control
        if config.reasoning_budget.enabled:
            self.budget_controller = ReasoningBudgetController(config)
        else:
            self.budget_controller = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, task_type=None,
                images=None, audio=None, user_id=None, knowledge_index=None):
        hidden_states = self.embed_tokens(input_ids)
        past_residuals = []
        total_aux_loss = 0

        for layer in self.layers:
            hidden_states, residual, aux_loss = layer(hidden_states, past_residuals, attention_mask, task_type)
            past_residuals.append(residual)
            if len(past_residuals) > 4:
                past_residuals.pop(0)
            total_aux_loss += aux_loss

        # P1: RAG enhancement
        rag_info = None
        if self.rag_module is not None:
            rag_info = self.rag_module(hidden_states, input_ids, knowledge_index=knowledge_index)

        # P2: CoT compilation
        cot_info = None
        if self.cot_compiler is not None:
            cot_info = self.cot_compiler.compile_reasoning(hidden_states)

        # P3: Agentic layer
        agent_info = None
        if self.agentic_layer is not None:
            agent_info = self.agentic_layer(hidden_states)

        # P4: Budget control
        budget_info = None
        if self.budget_controller is not None:
            budget, task_diff = self.budget_controller.get_budget(hidden_states, total_token_budget=4096)
            budget_info = {"budget": budget, "difficulty": task_diff}

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "rag_info": rag_info,
            "cot_info": cot_info,
            "agent_info": agent_info,
            "budget_info": budget_info,
        }

    def generate(self, input_ids, max_new_tokens=100, temperature=0.7,
                 use_speculative=True, task_type=None):
        if use_speculative and self.speculative_decoder is not None:
            return self.speculative_decoder.generate(input_ids, max_new_tokens, temperature)[0]

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(input_ids, task_type=task_type)
                logits = out["logits"]
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
