"""HeliosLM v1.0 - P0+P1+P2+P3: Speed + Accuracy + Reasoning + Agentic"""

__version__ = "1.0.0"
__license__ = "Apache License 2.0"
__author__ = "HeliosLM Team"

from .model import HeliosLM, HeliosLMLayer
from .attention import HeliosAttention, RMSNorm
from .moe import StableLatentMoE
from .speculative_decoding import SpeculativeDecoder, DraftModel
from .rag import RAGModule
from .cot_compiler import CoTCompiler
from .agentic import AgenticLayer, MCPToolRouter, SelfReflectionModule, CodeExecutionValidator
from .memory import LongTermMemory
from .tokenizer import HeliosTokenizer
from .data_pipeline import PretrainDataset, SFTDataset, RLHFDataset, create_dataloader

__all__ = [
    "HeliosLM",
    "HeliosLMLayer",
    "HeliosAttention",
    "RMSNorm",
    "StableLatentMoE",
    "SpeculativeDecoder",
    "DraftModel",
    "RAGModule",
    "CoTCompiler",
    "AgenticLayer",
    "MCPToolRouter",
    "SelfReflectionModule",
    "CodeExecutionValidator",
    "LongTermMemory",
    "HeliosTokenizer",
    "PretrainDataset",
    "SFTDataset",
    "RLHFDataset",
    "create_dataloader",
]
