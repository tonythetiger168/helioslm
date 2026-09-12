"""FastAPI RESTful server with OpenAI-compatible endpoints."""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import time
import uuid


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""
    model: str = "helioslm-ultra"
    messages: List[Dict[str, str]] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]


class HeliosLMFastAPI:
    """FastAPI application wrapper."""

    def __init__(self, engine, auth_validator=None, rate_limiter=None):
        self.engine = engine
        self.auth_validator = auth_validator
        self.rate_limiter = rate_limiter
        self.app = FastAPI(title="HeliosLM API", version="1.0.0")
        self.security = HTTPBearer()

        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "model": "helioslm-ultra"}

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "helioslm-ultra",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "helioslm",
                    },
                    {
                        "id": "helioslm-pro",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "helioslm",
                    },
                ]
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completion(request: ChatCompletionRequest):
            # Authentication
            if self.auth_validator:
                # Validate API key
                pass

            # Rate limiting
            if self.rate_limiter:
                # Check rate limit
                pass

            # Format prompt from messages
            prompt = self._format_messages(request.messages)

            if request.stream:
                return StreamingResponse(
                    self._stream_response(prompt, request),
                    media_type="text/event-stream",
                )
            else:
                return self._generate_response(prompt, request)

        @self.app.post("/v1/completions")
        async def completion(request: Dict):
            """Legacy completion endpoint."""
            prompt = request.get("prompt", "")
            req = ChatCompletionRequest(
                model=request.get("model", "helioslm-ultra"),
                messages=[{"role": "user", "content": prompt}],
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 256),
                stream=request.get("stream", False),
            )
            return await chat_completion(req)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _generate_response(self, prompt: str, request: ChatCompletionRequest) -> Dict:
        """Generate non-streaming response."""
        # Tokenize (placeholder)
        prompt_tokens = [ord(c) % 160000 for c in prompt[:200]]

        # Generate
        start_time = time.time()
        output_text = self.engine.generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        completion_tokens = len(output_text.split())

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
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_tokens) + completion_tokens,
            }
        }

    async def _stream_response(self, prompt: str, request: ChatCompletionRequest):
        """Generate streaming SSE response."""
        # Simulate streaming
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Send role first
        yield f"data: {json.dumps({\n            'id': response_id,\n            'object': 'chat.completion.chunk',\n            'created': created,\n            'model': request.model,\n            'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]\n        })}\n\n"

        # Stream tokens
        tokens = ["Hello", ",", " this", " is", " a", " streaming", " response", "."]
        for token in tokens:
            yield f"data: {json.dumps({\n                'id': response_id,\n                'object': 'chat.completion.chunk',\n                'created': created,\n                'model': request.model,\n                'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]\n            })}\n\n"

        # Send finish
        yield f"data: {json.dumps({\n            'id': response_id,\n            'object': 'chat.completion.chunk',\n            'created': created,\n            'model': request.model,\n            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]\n        })}\n\n"

        yield "data: [DONE]\n\n"


def create_app(engine, auth_validator=None, rate_limiter=None):
    """Create FastAPI application."""
    api = HeliosLMFastAPI(engine, auth_validator, rate_limiter)
    return api.app
