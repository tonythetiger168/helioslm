"""moe_dispatch_bench.py — blocked vs batched MoE dispatch (v5.106, #42 prep).

Times one DeviceLimitedMoE layer, forward + backward, with
``moe.dispatch`` "blocked" (fixed 64-row blocks, one call per expert and
block) and "batched" (two bmm calls over all active experts), on the same
weights and inputs, and checks that the outputs agree.

    python examples/moe_dispatch_bench.py                 # CPU, small
    python examples/moe_dispatch_bench.py --device cuda --dtype bf16 \\
        --tokens 8192 --hidden 640 --experts 16 --top-k 2 --expert-hidden 448

Writes JSON to --out if given. Wall-clock numbers are telemetry: they
depend on the machine and are not compared by any test.
"""
import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config, MoEConfig  # noqa: E402
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench(m, x, dtype, steps, warmup):
    dev = x.device
    times = []
    for i in range(warmup + steps):
        _sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            y = m(x)
        y.float().pow(2).mean().backward()
        _sync(dev)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
        m.zero_grad(set_to_none=True)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cfg = HeliosLMv5Config(size="lite", hidden_size=args.hidden, moe=MoEConfig(
        num_experts=args.experts, num_activated_experts=args.top_k,
        expert_hidden_size=args.expert_hidden, device_group_size=1))
    torch.manual_seed(args.seed)
    blocked = DeviceLimitedMoE(cfg).to(dev).train()
    batched = copy.deepcopy(blocked)
    batched.dispatch = "batched"
    x = torch.randn(1, args.tokens, args.hidden, device=dev)
    with torch.no_grad():
        diff = (blocked(x) - batched(x)).abs().max().item()
    res = {"blocked_s": bench(blocked, x, dtype, args.steps, args.warmup),
           "batched_s": bench(batched, x, dtype, args.steps, args.warmup)}
    res["speedup"] = res["blocked_s"] / res["batched_s"]
    payload = {"args": vars(args), "max_abs_diff_fp32": diff,
               "median_fwd_bwd": res,
               "environment": {"torch": torch.__version__,
                               "platform": platform.platform(),
                               "device": torch.cuda.get_device_name(0)
                               if dev.type == "cuda" else platform.processor()}}
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
