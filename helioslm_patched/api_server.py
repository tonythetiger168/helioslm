"""HeliosLM v4.1 Production API Server - OpenAI Compatible"""
import sys
import os
import json
import time
import uuid
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from helioslm_patched.src.model import KimiK3Plus
from helioslm_patched.src.tokenizer import KimiTokenizer
from helioslm_patched.configs import ultra_config, pro_config, lite_config, nano_config


SIZE_CONFIGS = {
    "helioslm-ultra": ultra_config,
    "helioslm-pro": pro_config,
    "helioslm-lite": lite_config,
    "helioslm-nano": nano_config,
}


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""
    model: str = "helioslm-ultra"
    messages: List[Dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None
    images: Optional[List[str]] = None
    audio: Optional[str] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None
    safety_check: bool = True


@dataclass
class CompletionRequest:
    """Legacy completion request."""
    model: str = "helioslm-ultra"
    prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None


class InferenceEngine:
    """Production inference engine with model caching."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = KimiTokenizer(vocab_size=160000)
        self.max_cache_size = 4

    def load_model(self, model_name: str):
        """Load model into cache."""
        if model_name in self.models:
            return self.models[model_name]

        if model_name not in SIZE_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        config = SIZE_CONFIGS[model_name]
        size = model_name.replace("helioslm-", "")

        model = KimiK3Plus(config, size=size).to(self.device)
        model.eval()

        # Cache management
        if len(self.models) >= self.max_cache_size:
            oldest = next(iter(self.models))
            del self.models[oldest]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.models[model_name] = model
        return model

    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """Non-streaming generation."""
        model = self.load_model(model_name)

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                use_speculative=kwargs.get("use_speculative", True),
                safety_check=kwargs.get("safety_check", True),
            )

        output_ids = output[0, input_tensor.shape[1]:].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    async def generate_stream(self, model_name: str, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Streaming generation with SSE."""
        model = self.load_model(model_name)

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        generated = input_tensor.clone()
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _, past_key_values, metadata = model.forward(
                    generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                next_logits = logits[:, -1, :] / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                yield token_text

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == model.config.eos_token_id:
                    break


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_minute=60, tokens_per_minute=100000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.requests = {}
        self.tokens = {}

    def check(self, api_key: str, num_tokens: int = 0) -> bool:
        """Check if request is within rate limit."""
        now = time.time()

        # Request limit
        if api_key not in self.requests:
            self.requests[api_key] = []
        self.requests[api_key] = [t for t in self.requests[api_key] if now - t < 60]

        if len(self.requests[api_key]) >= self.rpm:
            return False

        # Token limit
        if api_key not in self.tokens:
            self.tokens[api_key] = []
        self.tokens[api_key] = [(t, c) for t, c in self.tokens[api_key] if now - t < 60]

        total_tokens = sum(c for _, c in self.tokens[api_key])
        if total_tokens + num_tokens > self.tpm:
            return False

        self.requests[api_key].append(now)
        self.tokens[api_key].append((now, num_tokens))
        return True


class AuthValidator:
    """Simple API key validator."""

    def __init__(self, valid_keys=None):
        self.valid_keys = set(valid_keys or ["sk-test-key"])

    def validate(self, api_key: str) -> bool:
        return api_key in self.valid_keys


# Global engine
engine = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 HeliosLM v4.1 API Server starting...")
    yield
    print("🛑 Server shutting down...")


app = FastAPI(
    title="HeliosLM v4.1 API",
    description="Production LLM API with OpenAI-compatible endpoints",
    version="4.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
auth_validator = AuthValidator()
rate_limiter = RateLimiter()


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate API key."""
    api_key = credentials.credentials
    if not auth_validator.validate(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "4.1.0",
        "models": list(SIZE_CONFIGS.keys()),
        "device": str(engine.device),
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "helioslm",
                "permission": [],
                "root": name,
            }
            for name in SIZE_CONFIGS.keys()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    """OpenAI-compatible chat completion endpoint."""
    # Rate limiting
    if not rate_limiter.check(api_key, request.max_tokens):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Format prompt
    prompt = engine.tokenizer.apply_chat_template(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(request, prompt),
            media_type="text/event-stream",
        )
    else:
        return _generate_chat_response(request, prompt)


def _generate_chat_response(request: ChatCompletionRequest, prompt: str) -> Dict:
    """Generate non-streaming chat response."""
    start_time = time.time()

    output_text = engine.generate(
        request.model,
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        safety_check=request.safety_check,
    )

    prompt_tokens = len(engine.tokenizer.encode(prompt))
    completion_tokens = len(engine.tokenizer.encode(output_text))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_chat_response(request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    """Generate streaming SSE response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Send role
    yield f"data: {json.dumps({
        'id': response_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': request.model,
        'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
    })}\n\n"

    # Stream tokens
    async for token in engine.generate_stream(
        request.model, prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    ):
        yield f"data: {json.dumps({
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]
        })}\n\n"

    # Finish
    yield f"data: {json.dumps({
        'id': response_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': request.model,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
    })}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completion(request: CompletionRequest, api_key: str = Depends(get_api_key)):
    """Legacy completion endpoint."""
    if not rate_limiter.check(api_key, request.max_tokens):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    output_text = engine.generate(
        request.model,
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    prompt_tokens = len(engine.tokenizer.encode(request.prompt))
    completion_tokens = len(engine.tokenizer.encode(output_text))

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": output_text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/embeddings")
async def embeddings(request: Dict, api_key: str = Depends(get_api_key)):
    """Text embedding endpoint."""
    model = request.get("model", "helioslm-ultra")
    input_text = request.get("input", "")

    if isinstance(input_text, list):
        input_text = input_text[0]

    # Simple embedding (in production, use model's hidden states)
    tokens = engine.tokenizer.encode(input_text)
    embedding = [float(t) / 160000 for t in tokens[:256]]
    embedding += [0.0] * (256 - len(embedding))

    return {
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": embedding,
        }],
        "model": model,
        "usage": {
            "prompt_tokens": len(tokens),
            "total_tokens": len(tokens),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
