"""bench_kda_chunked.py — token loop vs chunked KDA scan (v5.93).

    python examples/bench_kda_chunked.py --device cuda --out benchmarks/kda_chunked_cuda.json

Times one training step (forward + backward) of a single per-channel
GatedDeltaAttention layer at the pilot width, for the token loop
(chunk 0) and each --chunks size, and checks each chunked output against
the loop. On CUDA the step runs under bf16 autocast, as in the pilot.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.attention.linear_attention import GatedDeltaAttention  # noqa: E402
from helioslm_v5.src.training.pretrain import pilot_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--chunks", default="16,32,64,128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = pilot_config("k3_block", hidden_size=args.hidden)
    torch.manual_seed(0)
    base = GatedDeltaAttention(cfg).to(dev)
    base.chunk_size = 0
    x = torch.randn(args.batch, args.seq, args.hidden, device=dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def step(m):
        xi = x.clone().requires_grad_(True)
        with amp:
            y, _ = m(xi)
        y.float().square().mean().backward()
        return y.detach().float()

    def timed(m):
        step(m)                                   # warm-up
        best = float("inf")
        for _ in range(args.repeats):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            step(m)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return best

    ref = step(base)
    rows = [{"chunk": 0, "seconds": timed(base), "max_abs_diff": 0.0}]
    for c in map(int, args.chunks.split(",")):
        m = copy.deepcopy(base)
        m.chunk_size = c
        diff = (step(m) - ref).abs().max().item()
        rows.append({"chunk": c, "seconds": timed(m), "max_abs_diff": diff})
    tokens = args.batch * args.seq
    for r in rows:
        r["speedup"] = rows[0]["seconds"] / r["seconds"]
        r["tokens_per_s"] = tokens / r["seconds"]
        print(f"chunk {r['chunk']:4d}  {r['seconds']:.3f} s  "
              f"{r['tokens_per_s']:9.0f} tok/s  x{r['speedup']:.1f}  "
              f"diff {r['max_abs_diff']:.1e}")
    if args.out:
        info = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                "torch": torch.__version__, "batch": args.batch, "seq": args.seq,
                "hidden": args.hidden, "heads": base.num_heads,
                "head_dim": base.head_dim, "rows": rows}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
