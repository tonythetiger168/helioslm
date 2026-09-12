#!/usr/bin/env python3
"""Kimi K3+ v3.0 Benchmark"""
import sys, time, torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"src"))
from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus

SIZE_CONFIGS={"ultra":ultra_config,"pro":pro_config,"lite":lite_config,"nano":nano_config}

def benchmark(model,config,device,num_tokens=100):
    input_ids=torch.randint(0,config.vocab_size,(1,128)).to(device)
    # warmup
    with torch.no_grad(): _=model.generate(input_ids,max_new_tokens=10,use_speculative=False)
    # standard
    if device.type=="cuda": torch.cuda.synchronize()
    t0=time.time()
    with torch.no_grad(): _=model.generate(input_ids,max_new_tokens=num_tokens,use_speculative=False)
    if device.type=="cuda": torch.cuda.synchronize()
    std_speed=num_tokens/(time.time()-t0)
    # speculative
    spec_speed=0
    if model.speculative_decoder is not None:
        if device.type=="cuda": torch.cuda.synchronize()
        t0=time.time()
        with torch.no_grad(): _=model.generate(input_ids,max_new_tokens=num_tokens,use_speculative=True)
        if device.type=="cuda": torch.cuda.synchronize()
        spec_speed=num_tokens/(time.time()-t0)
    return std_speed,spec_speed

def main():
    print("K3+ v3.0 Benchmark"); print("="*50)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name,cfg in SIZE_CONFIGS.items():
        print(f"\n📊 {name.upper()}")
        model=KimiK3Plus(cfg,size=name).to(device)
        print(f"   Params:{sum(p.numel() for p in model.parameters()):,}")
        std,spec=benchmark(model,cfg,device)
        print(f"   Standard:{std:.1f} tok/s")
        if spec>0: print(f"   Speculative:{spec:.1f} tok/s | Speedup:{spec/std:.2f}x")
        del model; torch.cuda.empty_cache() if device.type=="cuda" else None
    print("\n"+"="*50+"\nDone!")

if __name__=="__main__":main()
