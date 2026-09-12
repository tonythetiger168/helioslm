#!/usr/bin/env python3
"""Kimi K3+ v2.0 推理入口"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus

SIZE_CONFIGS = {
    "ultra": ultra_config,
    "pro": pro_config,
    "lite": lite_config,
    "nano": nano_config,
}


def main():
    parser = argparse.ArgumentParser(description="Kimi K3+ Inference")
    parser.add_argument("--size", type=str, default="lite", choices=["ultra", "pro", "lite", "nano"])
    parser.add_argument("--prompt", type=str, default="解释量子计算的基本原理")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--speculative", action="store_true", help="使用推测解码")
    args = parser.parse_args()

    config = SIZE_CONFIGS[args.size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Kimi K3+ v2.0 Inference")
    print(f"  Size: {args.size.upper()}")
    print(f"  Model: {config.model_name}")
    print(f"  Speculative: {args.speculative}")

    model = KimiK3Plus(config, size=args.size).to(device)
    model.eval()

    # 模拟输入
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)

    print(f"\nPrompt: {args.prompt}")
    print("Generating...")

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_speculative=args.speculative
        )

    print(f"Output shape: {output.shape}")
    print(f"Generated tokens: {output.shape[1] - input_ids.shape[1]}")
    print("\n推理完成！")


if __name__ == "__main__":
    main()
