"""HeliosLM v1.0.2 — P0+P1+P2+P3: Speed + Accuracy + Reasoning + Agentic

P0 FIX: Added KVCache and updated exports for v1.0.2.
"""

__version__ = "1.0.2"
__license__ = "Apache License 2.0"
__author__ = "HeliosLM Team"

from .model import HeliosLM, HeliosLMLayer
from .attention import HeliosAttention, RMSNorm, KVCache
from .moe import StableLatentMoE
from .speculative_decoding import SpeculativeDecoder, DraftModel
from .rag import RAGModule
from .cot_compiler import CoTCompiler
from .agentic import AgenticLayer, MCPToolRouter, SelfReflectionModule, DockerSandboxExecutor
from .memory import LongTermMemory
from .tokenizer import HeliosTokenizer
from .data_pipeline import PretrainDataset, SFTDataset, RLHFDataset, create_dataloader

__all__ = [
    "HeliosLM",
    "HeliosLMLayer",
    "HeliosAttention",
    "RMSNorm",
    "KVCache",
    "StableLatentMoE",
    "SpeculativeDecoder",
    "DraftModel",
    "RAGModule",
    "CoTCompiler",
    "AgenticLayer",
    "MCPToolRouter",
    "SelfReflectionModule",
    "DockerSandboxExecutor",
    "LongTermMemory",
    "HeliosTokenizer",
    "PretrainDataset",
    "SFTDataset",
    "RLHFDataset",
    "create_dataloader",
]
