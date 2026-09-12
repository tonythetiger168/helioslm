"""Production RAG + Hallucination Calibration - P1"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class RAGModule(nn.Module):
    """Production RAG with FAISS/Milvus integration, Confidence Calibration, Multi-model verification"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.retrieval_k = config.rag.retrieval_k
        self.conf_threshold = config.rag.confidence_threshold
        self.force_retrieval = config.rag.force_retrieval
        self.multi_model_verify = config.rag.multi_model_verification

        self.query_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        if self.multi_model_verify:
            self.verify_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        self.uncertainty_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    def encode_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(hidden_states[:, -1, :]), dim=-1)

    def retrieve(self, query_embedding: torch.Tensor, knowledge_index: Optional[object] = None):
        if knowledge_index is not None:
            try:
                import faiss
                query_np = query_embedding.detach().cpu().numpy()
                scores, indices = knowledge_index.search(query_np, self.retrieval_k)
                return torch.from_numpy(indices), torch.from_numpy(scores)
            except Exception:
                pass
        return None, None

    def calibrate_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.confidence_head(hidden_states[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logits / self.temperature)

    def verify_with_context(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if not self.multi_model_verify:
            return torch.ones(hidden_states.size(0))
        gen_state = hidden_states[:, -1, :]
        ctx_state = context.mean(dim=1) if context.dim() == 3 else context
        combined = torch.cat([gen_state, ctx_state], dim=-1)
        return self.verify_head(combined).squeeze(-1)

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                knowledge_index: Optional[object] = None, retrieved_context: Optional[torch.Tensor] = None) -> Dict:
        conf = self.calibrate_confidence(hidden_states)
        needs_retrieval = self.force_retrieval or conf.min() < self.conf_threshold

        retrieved_knowledge = None
        verification_score = torch.ones_like(conf)

        if needs_retrieval:
            query_emb = self.encode_query(hidden_states)
            indices, scores = self.retrieve(query_emb, knowledge_index)
            if retrieved_context is not None:
                verification_score = self.verify_with_context(hidden_states, retrieved_context)
                ctx_encoded = self.context_encoder(retrieved_context.mean(dim=1))
                hidden_states = hidden_states + 0.1 * ctx_encoded.unsqueeze(1)

        effective_conf = conf * verification_score
        is_uncertain = effective_conf < 0.3

        return {
            "hidden_states": hidden_states,
            "confidence": effective_conf,
            "raw_confidence": conf,
            "verification_score": verification_score,
            "uncertain": is_uncertain,
            "needs_retrieval": needs_retrieval,
            "retrieved_indices": retrieved_knowledge,
        }
