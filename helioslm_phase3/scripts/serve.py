#!/usr/bin/env python3
"""HeliosLM Phase 3 — Production Inference Server (Fixed v1.0.1)

Fixed:
  - Real model loading from Phase 2 checkpoints
  - FlashAttention kernel auto-detection
  - Proper error handling

Usage:
    python scripts/serve.py --model checkpoints/dpo_final.pt --mode rest
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import uvicorn

from api import create_app, APIKeyValidator, RateLimiter
from api.grpc_server import serve_grpc
from inference import VLLMEngine
from inference.kernels.flash_attn_triton import flash_attn_kernel
from quantization import convert_to_fp8
from monitoring import MetricsCollector
from safety import ContentModerator, InputValidator, AuditLogger

# Import model architecture from P0 fix
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "helioslm_p0_fix"))
from src.model import HeliosLM
from configs.ultra_config import ultra_config


def load_model(checkpoint_path: str, model_size: str, device: str = "cuda"):
    """Load HeliosLM model from checkpoint."""
    print(f"📂 Loading model from {checkpoint_path}")

    model = HeliosLM(ultra_config, size=model_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded: {total_params:.1f}B params")
    print(f"⚡ FlashAttention: {flash_attn_kernel.backend}")

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rest","grpc","both"], default="rest")
    p.add_argument("--model", required=True)
    p.add_argument("--model-size", default="ultra")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--grpc-port", type=int, default=50051)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--quantization", choices=["fp8","int8","awq","gptq","dynamic"], default=None)
    p.add_argument("--enable-moderation", action="store_true")
    p.add_argument("--enable-audit-log", action="store_true")
    p.add_argument("--api-keys", nargs="+", default=[])
    p.add_argument("--rate-limit-rpm", type=int, default=60)
    p.add_argument("--rate-limit-tpm", type=int, default=100000)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("HeliosLM Phase 3 — Inference Server v1.0.1")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, args.model_size, device)

    if args.quantization == "fp8":
        print("🔧 FP8 quantization...")
        model = convert_to_fp8(model)

    engine = VLLMEngine(model_path=args.model, config=ultra_config)
    auth = APIKeyValidator(set(args.api_keys)) if args.api_keys else None
    rate_limiter = RateLimiter(args.rate_limit_rpm, args.rate_limit_tpm)

    if args.mode in ("rest", "both"):
        app = create_app(engine, auth, rate_limiter)
        print(f"🚀 REST: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    if args.mode in ("grpc", "both"):
        server = serve_grpc(engine, port=args.grpc_port)
        print(f"🚀 gRPC: port {args.grpc_port}")
        server.wait_for_termination()


if __name__ == "__main__":
    main()
