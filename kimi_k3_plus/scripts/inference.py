#!/usr/bin/env python3
"""Kimi K3+ 推理入口"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.model_config import default_config
from model import KimiK3Plus


def generate(model, input_ids, max_new_tokens=100, temperature=0.7):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :] / temperature, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


def main():
    print("Kimi K3+ Inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KimiK3Plus(default_config).to(device)
    input_ids = torch.randint(0, default_config.vocab_size, (1, 10)).to(device)
    output = generate(model, input_ids, max_new_tokens=20)
    print(f"Output shape: {output.shape}")
    print("推理测试完成")


if __name__ == "__main__":
    main()
