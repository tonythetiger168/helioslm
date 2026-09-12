#!/usr/bin/env python3
"""HeliosLM Inference Script - Production Inference with vLLM-style Serving

Usage:
    # Single prompt inference
    python scripts/inference.py --size nano --prompt "Explain quantum computing"

    # Batch inference from JSONL
    python scripts/inference.py --size lite --batch inputs.jsonl --output outputs.jsonl

    # With speculative decoding
    python scripts/inference.py --size lite --speculative --prompt "Write a poem"

    # Launch API server
    python scripts/inference.py --size pro --serve --port 8000

    # Interactive chat
    python scripts/inference.py --size nano --chat
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import HeliosLMConfig, ultra_config, pro_config, lite_config, nano_config
from src import HeliosLM


def get_config(size: str) -> HeliosLMConfig:
    configs = {"ultra": ultra_config, "pro": pro_config, "lite": lite_config, "nano": nano_config}
    if size not in configs:
        raise ValueError(f"Unknown size: {size}")
    return configs[size]


def load_model(size: str, checkpoint_path: str = None, device: str = "cuda"):
    """Load HeliosLM model."""
    config = get_config(size)
    model = HeliosLM(config, size=size)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint loaded. Using random initialization.")
        print("For production use, train or download pretrained weights.")

    model.to(device)
    model.eval()
    return model, config


def generate_single(model: HeliosLM, prompt: str, max_tokens: int = 100,
                    temperature: float = 0.7, use_speculative: bool = False,
                    device: str = "cuda") -> str:
    """Generate text from a single prompt."""
    # Tokenize (placeholder - use real tokenizer in production)
    # For demo, convert prompt to token IDs using simple hash
    token_ids = [ord(c) % 160000 for c in prompt[:512]]
    input_ids = torch.tensor([token_ids], device=device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_speculative=use_speculative,
        )
    elapsed = time.time() - start_time

    # Decode (placeholder - use real tokenizer in production)
    generated = output_ids[0, len(token_ids):].tolist()
    text = "".join([chr(t % 128) if 32 <= t % 128 <= 126 else " " for t in generated])

    speed = len(generated) / elapsed
    print(f"Generated {len(generated)} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)")
    return text


def batch_inference(model: HeliosLM, input_file: str, output_file: str,
                    max_tokens: int = 100, temperature: float = 0.7,
                    device: str = "cuda"):
    """Batch inference from JSONL file."""
    results = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            text = generate_single(model, prompt, max_tokens, temperature, device=device)
            results.append({"prompt": prompt, "output": text})

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Batch inference complete. Results saved to {output_file}")


def serve_api(model: HeliosLM, config: HeliosLMConfig, port: int = 8000):
    """Launch FastAPI serving endpoint."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return

    app = FastAPI(title="HeliosLM API", version="1.0.0")

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.7)

        text = generate_single(model, prompt, max_tokens, temperature)
        return JSONResponse({
            "id": "helioslm-completion",
            "object": "text_completion",
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        })

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{"id": config.model_name, "object": "model"}],
        })

    print(f"Starting HeliosLM API server on port {port}")
    print(f"Model: {config.model_name}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def interactive_chat(model: HeliosLM, config: HeliosLMConfig, device: str = "cuda"):
    """Interactive chat mode."""
    print(f"=== HeliosLM {config.model_name} Chat ===")
    print("Type 'exit' to quit, 'clear' to clear history.")
    history = []

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "exit":
            break
        if prompt.lower() == "clear":
            history = []
            print("History cleared.")
            continue

        full_prompt = "\n".join(history + [f"User: {prompt}", "Assistant:"])
        response = generate_single(model, full_prompt, max_tokens=200, device=device)

        history.append(f"User: {prompt}")
        history.append(f"Assistant: {response}")

        print(f"Assistant: {response}")


def main():
    parser = argparse.ArgumentParser(description="HeliosLM Inference")
    parser.add_argument("--size", type=str, default="nano",
                        choices=["ultra", "pro", "lite", "nano"],
                        help="Model size")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for inference")
    parser.add_argument("--batch", type=str, default=None,
                        help="Path to JSONL file for batch inference")
    parser.add_argument("--output", type=str, default="outputs.jsonl",
                        help="Output file for batch inference")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable speculative decoding")
    parser.add_argument("--serve", action="store_true",
                        help="Launch API server")
    parser.add_argument("--port", type=int, default=8000,
                        help="API server port")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.size, args.checkpoint, args.device)
    print(f"Loaded {config.model_name} on {args.device}")

    # Run inference mode
    if args.serve:
        serve_api(model, config, port=args.port)
    elif args.chat:
        interactive_chat(model, config, device=args.device)
    elif args.batch:
        batch_inference(model, args.batch, args.output,
                        args.max_tokens, args.temperature, args.device)
    elif args.prompt:
        text = generate_single(model, args.prompt, args.max_tokens,
                               args.temperature, args.speculative, args.device)
        print(f"\nOutput: {text}")
    else:
        print("No input provided. Use --prompt, --batch, --serve, or --chat.")


if __name__ == "__main__":
    main()
