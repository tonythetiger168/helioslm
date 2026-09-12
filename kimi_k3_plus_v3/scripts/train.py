#!/usr/bin/env python3
"""Kimi K3+ v3.0 Training"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"src"))
from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus
from trainer import PretrainingTrainer, SFTTrainer

SIZE_CONFIGS={"ultra":ultra_config,"pro":pro_config,"lite":lite_config,"nano":nano_config}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--size",default="ultra",choices=["ultra","pro","lite","nano"])
    parser.add_argument("--phase",default="pretrain",choices=["pretrain","sft"])
    args=parser.parse_args()
    config=SIZE_CONFIGS[args.size]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"K3+ v3.0 Training | Size:{args.size.upper()} | Model:{config.model_name} | Device:{device}")
    model=KimiK3Plus(config,size=args.size).to(device)
    print(f"Parameters:{sum(p.numel() for p in model.parameters()):,}")
    print("Ready for training!")

if __name__=="__main__":main()
