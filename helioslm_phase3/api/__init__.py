"""HeliosLM Phase 3 — API Service

Production API layer:
  - FastAPI RESTful with OpenAI-compatible endpoints
  - gRPC for high-throughput internal communication
  - Server-Sent Events (SSE) for streaming responses
  - Request validation, rate limiting, authentication
"""
from .rest_server import create_app, ChatCompletionRequest
from .grpc_server import HeliosLMServiceServicer
from .streaming import stream_generator
from .auth import APIKeyValidator, RateLimiter

__all__ = [
    "create_app", "ChatCompletionRequest",
    "HeliosLMServiceServicer",
    "stream_generator",
    "APIKeyValidator", "RateLimiter",
]
