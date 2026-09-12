import os, sys, torch
sys.path.insert(0, "/mnt/agents/output")
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
    cfg = HeliosLMv5Config(size="lite")
    torch.manual_seed(42)
    moe = DeviceLimitedMoE(cfg)
    # give each rank a DIFFERENT load
    moe.expert_load.fill_(float(rank + 1))
    moe.update_bias()
    b = moe.route_bias.detach().clone()
    dist.broadcast(b, src=0)
    same = torch.equal(b, moe.route_bias)
    print(f"rank{rank}: bias identical across ranks after all_reduce update: {same}", flush=True)
    # device-limited routing smoke: forward under distributed
    x = torch.randn(2, 3, cfg.hidden_size)
    out = moe(x)
    print(f"rank{rank}: dist forward ok {tuple(out.shape)} finite={torch.isfinite(out).all().item()}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
