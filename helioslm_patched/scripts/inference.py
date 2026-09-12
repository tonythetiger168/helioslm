#!/usr/bin/env python3
"""Kimi K3+ v4.1 Inference Script"""
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from configs import ultra_config, pro_config, lite_config, nano_config
from src.model import KimiK3Plus

SIZE_CONFIGS = {"ultra": ultra_config, "pro": pro_config, "lite": lite_config, "nano": nano_config}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="lite", choices=["ultra", "pro", "lite", "nano"])
    parser.add_argument("--prompt", default="解釋量子計算")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = SIZE_CONFIGS[args.size]
    device = torch.device(args.device)

    print(f"K3+ v4.1 Inference | Size:{args.size.upper()} | Device:{device}")

    model = KimiK3Plus(config, size=args.size).to(device)
    model.eval()

    # Mock tokenization (replace with real tokenizer)
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_speculative=args.speculative
        )

    print(f"Output shape: {output.shape} | Tokens: {output.shape[1] - input_ids.shape[1]}")
    print("Note: Use real tokenizer for text output")


if __name__ == "__main__":
    main()
