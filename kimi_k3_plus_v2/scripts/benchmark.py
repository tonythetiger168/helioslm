#!/usr/bin/env python3
"""Kimi K3+ v2.0 基准测试"""

import sys
import time
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


def benchmark_speed(model, config, device, num_tokens=100):
    """测试生成速度"""
    input_ids = torch.randint(0, config.vocab_size, (1, 128)).to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, use_speculative=False)

    # Benchmark standard
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=num_tokens, use_speculative=False)
    torch.cuda.synchronize() if device.type == "cuda" else None
    std_time = time.time() - start
    std_speed = num_tokens / std_time

    # Benchmark speculative
    if model.speculative_decoder is not None:
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=num_tokens, use_speculative=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        spec_time = time.time() - start
        spec_speed = num_tokens / spec_time
        speedup = spec_speed / std_speed
    else:
        spec_speed = 0
        speedup = 0

    return std_speed, spec_speed, speedup


def main():
    print("Kimi K3+ v2.0 Benchmark")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for size_name, config in SIZE_CONFIGS.items():
        print(f"\n📊 Testing {size_name.upper()}...")
        model = KimiK3Plus(config, size=size_name).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")

        std_speed, spec_speed, speedup = benchmark_speed(model, config, device)
        print(f"   Standard Speed: {std_speed:.1f} tok/s")
        if spec_speed > 0:
            print(f"   Speculative Speed: {spec_speed:.1f} tok/s")
            print(f"   Speedup: {speedup:.2f}x")
        else:
            print(f"   Speculative: N/A")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print("\n" + "=" * 50)
    print("基准测试完成！")


if __name__ == "__main__":
    main()
