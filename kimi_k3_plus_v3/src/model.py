"""Kimi K3+ v4.0 Unified Model - P0+P1+P2+P3+P4+P5+P6+P7+P8"""
import torch, torch.nn as nn
from .attention import KimiDeltaAttention, RMSNorm
from .moe import StableLatentMoE
from .speculative_decoding import SpeculativeDecoder, DraftModel
from .rag import RAGModule
from .cot_compiler import CoTCompiler
from .agentic import AgenticLayer
from .memory import LongTermMemory
from .reasoning_budget import ReasoningBudgetController
from .multimodal import MultimodalFusion
from .adaptive_reasoning_v2 import AdaptiveReasoningController, MCTSReasoningSearch
from .online_learning import OnlineLearningManager
from .safety_alignment import SafetyAlignmentLayer
from .compression import CompressionManager

class KimiK3PlusLayer(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__(); self.layer_idx=layer_idx
        self.attention=KimiDeltaAttention(config); self.moe=StableLatentMoE(config)
        self.post_attention_norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    def forward(self,hidden_states,past_residuals=None,attention_mask=None,task_type=None):
        attn_out,residual=self.attention(hidden_states,self.layer_idx,past_residuals,attention_mask)
        hidden_states=hidden_states+attn_out
        moe_out,aux_loss=self.moe(self.post_attention_norm(hidden_states),task_type=task_type)
        hidden_states=hidden_states+moe_out
        return hidden_states,residual,aux_loss

class KimiK3Plus(nn.Module):
    def __init__(self,config,size="ultra"):
        super().__init__(); self.config=config; self.size=size
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([KimiK3PlusLayer(config,i) for i in range(config.num_hidden_layers)])
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        # P0
        if config.speculative.enabled and size!="nano":
            self.draft_model=DraftModel(config); self.speculative_decoder=SpeculativeDecoder(self,self.draft_model,config)
        else: self.speculative_decoder=None
        # P1
        if config.rag.enabled: self.rag_module=RAGModule(config)
        else: self.rag_module=None
        # P2
        if config.cot.enabled: self.cot_compiler=CoTCompiler(config)
        else: self.cot_compiler=None
        # P3
        if config.agentic.enabled:
            self.agentic_layer=AgenticLayer(config); self.memory=LongTermMemory(config)
        else: self.agentic_layer=None; self.memory=None
        # P4
        if config.reasoning_budget.enabled: self.budget_controller=ReasoningBudgetController(config)
        else: self.budget_controller=None
        # P5: Multimodal Fusion
        if config.multimodal.enabled:
            self.multimodal_fusion=MultimodalFusion(config)
        else: self.multimodal_fusion=None
        # P6: Adaptive Reasoning V2 + Online Learning
        if config.adaptive_reasoning_v2.enabled:
            self.reasoning_controller=AdaptiveReasoningController(config)
            self.mcts_search=MCTSReasoningSearch(config)
        else: self.reasoning_controller=None; self.mcts_search=None
        if config.online_learning.enabled:
            self.online_learning=OnlineLearningManager(config)
        else: self.online_learning=None
        # P7: Safety Alignment
        if config.safety.enabled:
            self.safety_layer=SafetyAlignmentLayer(config)
        else: self.safety_layer=None
        # P8: Compression
        if config.compression.enabled:
            self.compression=CompressionManager(config)
        else: self.compression=None
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,mean=0.0,std=0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0.0,std=0.02)

    def forward(self, input_ids, attention_mask=None, task_type=None, images=None, audio=None, user_id=None):
        hidden_states = self.embed_tokens(input_ids)
        past_residuals = []
        total_aux_loss = 0
        for layer in self.layers:
            hidden_states, residual, aux_loss = layer(hidden_states, past_residuals, attention_mask, task_type)
            past_residuals.append(residual)
            if len(past_residuals) > 4:
                past_residuals.pop(0)
            total_aux_loss += aux_loss
        # P5: Multimodal fusion
        if self.multimodal_fusion is not None and (images is not None or audio is not None):
            hidden_states = self.multimodal_fusion(hidden_states, images=images, audio=audio)
        # P1: RAG
        if self.rag_module is not None:
            rag_out = self.rag_module(hidden_states, input_ids)
            if rag_out.get("uncertain"):
                pass
        # P2: CoT
        if self.cot_compiler is not None:
            cot_out = self.cot_compiler.compile_reasoning(hidden_states)
        # P3: Agentic
        if self.agentic_layer is not None:
            agent_out = self.agentic_layer(hidden_states)
        # P4: Budget control
        if self.budget_controller is not None:
            budget, task_diff = self.budget_controller.get_budget(hidden_states, total_token_budget=4096)
        # P6: Adaptive reasoning depth
        if self.reasoning_controller is not None:
            budget_v2, level = self.reasoning_controller.get_budget(hidden_states)
        # P6: Online learning user adaptation
        if self.online_learning is not None and user_id is not None:
            hidden_states = self.online_learning.process_feedback(user_id, hidden_states, feedback_score=0.5)
        # P7: Safety alignment
        if self.safety_layer is not None:
            safety_out = self.safety_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, total_aux_loss

    def generate(self,input_ids,max_new_tokens=100,temperature=0.7,use_speculative=True,task_type=None):
        if use_speculative and self.speculative_decoder is not None:
            return self.speculative_decoder.generate(input_ids,max_new_tokens,temperature)[0]
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits,_=self.forward(input_ids,task_type=task_type)
                next_token=torch.multinomial(torch.softmax(logits[:,-1,:]/temperature,dim=-1),num_samples=1)
                input_ids=torch.cat([input_ids,next_token],dim=1)
        return input_ids
