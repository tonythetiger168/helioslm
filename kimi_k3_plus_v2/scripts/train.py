#!/usr/bin/env python3
"""Kimi K3+ v2.0 训练入口 - 支持四个尺寸"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus
from trainer import PretrainingTrainer, DistillationTrainer, SFTTrainer


SIZE_CONFIGS = {
    "ultra": ultra_config,
    "pro": pro_config,
    "lite": lite_config,
    "nano": nano_config,
}


def main():
    parser = argparse.ArgumentParser(description="Kimi K3+ Training")
    parser.add_argument("--size", type=str, default="ultra", choices=["ultra", "pro", "lite", "nano"])
    parser.add_argument("--phase", type=str, default="pretrain", choices=["pretrain", "distill", "sft"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = SIZE_CONFIGS[args.size]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Kimi K3+ v2.0 Training")
    print(f"  Size: {args.size.upper()}")
    print(f"  Model: {config.model_name}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Context: {config.max_position_embeddings:,}")
    print(f"  Device: {device}")

    model = KimiK3Plus(config, size=args.size).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total:,}")

    if args.phase == "pretrain":
        print("\n启动预训练...")
        # trainer = PretrainingTrainer(model, config, device)
        # trainer.train(dataloader, num_steps=1000000)
    elif args.phase == "distill" and args.size == "nano":
        print("\n启动知识蒸馏...")
        teacher = KimiK3Plus(ultra_config, size="ultra").to(device)
        # trainer = DistillationTrainer(teacher, model, config, device)
        # trainer.train(dataloader)
    elif args.phase == "sft":
        print("\n启动监督微调...")
        # trainer = SFTTrainer(model, config, device)
        # trainer.train(dataloader)

    print("\n训练准备完成！")


if __name__ == "__main__":
    main()
