#!/usr/bin/env python3
"""Kimi K3+ v4.1 Training Script"""
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from configs import ultra_config, pro_config, lite_config, nano_config
from src.model import KimiK3Plus
from src.trainer import PretrainingTrainer, SFTTrainer

SIZE_CONFIGS = {"ultra": ultra_config, "pro": pro_config, "lite": lite_config, "nano": nano_config}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="lite", choices=["ultra", "pro", "lite", "nano"])
    parser.add_argument("--phase", default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = SIZE_CONFIGS[args.size]
    device = torch.device(args.device)

    print(f"K3+ v4.1 Training | Size:{args.size.upper()} | Model:{config.model_name} | Device:{device}")

    model = KimiK3Plus(config, size=args.size).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # TODO: Load dataloader
    print("Ready for training! Use --phase pretrain or --phase sft")


if __name__ == "__main__":
    main()
