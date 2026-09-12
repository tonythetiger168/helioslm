"""HeliosLM Phase 3 — Inference Optimization

Production inference stack:
  - vLLM integration with PagedAttention
  - Continuous batching for throughput maximization
  - Speculative decoding with draft model
  - Tensor parallelism for multi-GPU serving
  - Prefix caching for common prompts
"""
from .vllm_engine import VLLMEngine, HeliosLMvLLM
from .continuous_batching import ContinuousBatcher, BatchRequest
from .speculative_service import SpeculativeService
from .prefix_cache import PrefixCache
from .tensor_parallel import TensorParallelEngine

__all__ = [
    "VLLMEngine", "HeliosLMvLLM",
    "ContinuousBatcher", "BatchRequest",
    "SpeculativeService",
    "PrefixCache",
    "TensorParallelEngine",
]
