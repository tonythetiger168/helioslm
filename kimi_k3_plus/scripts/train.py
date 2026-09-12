#!/usr/bin/env python3
"""Kimi K3+ 训练入口"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.model_config import default_config
from model import KimiK3Plus
from trainer import PretrainingTrainer, SFTTrainer


def main():
    print("Kimi K3+ Training")
    print(f"  Model: {default_config.model_name}")
    print(f"  Context: {default_config.max_position_embeddings:,} tokens")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KimiK3Plus(default_config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total:,}")
    print("模型初始化完成，准备训练...")


if __name__ == "__main__":
    main()
