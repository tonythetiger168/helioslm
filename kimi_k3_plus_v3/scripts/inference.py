#!/usr/bin/env python3
"""Kimi K3+ v3.0 Inference"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"src"))
from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus

SIZE_CONFIGS={"ultra":ultra_config,"pro":pro_config,"lite":lite_config,"nano":nano_config}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--size",default="lite",choices=["ultra","pro","lite","nano"])
    parser.add_argument("--prompt",default="解释量子计算")
    parser.add_argument("--max-tokens",type=int,default=200)
    parser.add_argument("--speculative",action="store_true")
    args=parser.parse_args()
    config=SIZE_CONFIGS[args.size]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=KimiK3Plus(config,size=args.size).to(device); model.eval()
    input_ids=torch.randint(0,config.vocab_size,(1,10)).to(device)
    with torch.no_grad():
        output=model.generate(input_ids,max_new_tokens=args.max_tokens,use_speculative=args.speculative)
    print(f"Output shape:{output.shape} | Tokens:{output.shape[1]-input_ids.shape[1]}")

if __name__=="__main__":main()
