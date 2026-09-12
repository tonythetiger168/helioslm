"""gRPC server for high-throughput internal communication."""
from concurrent import futures
import grpc
from typing import Optional


class HeliosLMServiceServicer:
    """gRPC servicer for HeliosLM."""

    def __init__(self, engine):
        self.engine = engine

    def Generate(self, request, context):
        """Handle generation request."""
        # This would use generated protobuf definitions
        # Placeholder implementation
        response = type('Response', (), {
            'text': self.engine.generate(request.prompt, max_tokens=request.max_tokens),
            'tokens_generated': 0,
        })()
        return response

    def StreamGenerate(self, request, context):
        """Handle streaming generation request."""
        # Yield tokens as they're generated
        for token in ["Hello", " world", "!"]:
            yield type('Chunk', (), {'token': token})()


def serve_grpc(engine, port: int = 50051, max_workers: int = 10):
    """Start gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # In real implementation, add generated servicer to server
    # add_HeliosLMServiceServicer_to_server(HeliosLMServiceServicer(engine), server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"🚀 gRPC server started on port {port}")
    return server
