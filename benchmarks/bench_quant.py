"""Reproducible fused vs unfused AWQ/GPTQ/MXFP4 quantization benchmark (Issue #2).

Example:
    python benchmarks/bench_quant.py --in-features 4096 --out-features 14336 --group-size 128 --repeats 7 --out benchmarks/quant_results.json

Start with --in-features 256 --out-features 512 on low-memory laptops.
Times are CPU wall-clock medians; no speedup is assumed.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("awq", "gptq", "mxfp4")
sys.path.insert(0, str(ROOT))


def _run_case(kind, fused, args):
    import torch
    from evastar.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, MXFP4Linear)
    torch.manual_seed(args.seed)
    linear = torch.nn.Linear(args.in_features, args.out_features, bias=False)
    if kind == "mxfp4":
        # MXFP4 blocks are fixed at 32 by the OCP MX spec; --group-size is
        # the AWQ/GPTQ group only
        quant = MXFP4Linear.from_linear(linear, block_size=32)
    else:
        cls = {"awq": AWQLinear, "gptq": GPTQLinear}[kind]
        quant = cls.from_linear(linear, group_size=args.group_size)
    del linear
    quant.fused = fused
    quant.eval()
    x = torch.randn(args.batch, args.in_features)
    # Validate fused/unfused numerical parity on the SAME quantized weights
    # before measuring; fail loudly rather than publishing invalid speedups.
    with torch.inference_mode():
        quant.fused = False
        oracle = quant(x)
        quant.fused = True
        candidate = quant(x)
        max_abs_error = (candidate - oracle).abs().max().item()
        torch.testing.assert_close(candidate, oracle, atol=1e-4, rtol=1e-4)
        quant.fused = fused
        for _ in range(args.warmup):
            quant(x)
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            quant(x)
            durations.append((time.perf_counter() - start) * 1000)
    return {"method": kind, "fused": fused, "median_ms": statistics.median(durations),
            "samples_ms": durations, "max_abs_error_vs_reference": max_abs_error, "torch": torch.__version__}


def _rss_kb():
    # Linux ru_maxrss is KiB, macOS is bytes; Windows resource may be absent.
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value / 1024) if sys.platform == "darwin" else int(value)
    except ImportError:
        return None


def _child(args):
    row = _run_case(args.method, args.fused == "yes", args)
    row["peak_rss_kib"] = _rss_kb()
    print(json.dumps(row))


def _parent(args):
    results = []
    for method in args.methods:
        for fused in ("no", "yes"):
            cmd = [sys.executable, __file__, "--child", "--method", method,
                   "--fused", fused, "--in-features", str(args.in_features),
                   "--out-features", str(args.out_features), "--group-size", str(args.group_size),
                   "--batch", str(args.batch), "--warmup", str(args.warmup),
                   "--repeats", str(args.repeats), "--seed", str(args.seed)]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    summary = {}
    for method in args.methods:
        base = next(r for r in results if r["method"] == method and not r["fused"])
        fused = next(r for r in results if r["method"] == method and r["fused"])
        summary[method] = {"speedup_x": base["median_ms"] / fused["median_ms"],
                           "unfused_median_ms": base["median_ms"],
                           "fused_median_ms": fused["median_ms"],
                           "unfused_peak_rss_kib": base["peak_rss_kib"],
                           "fused_peak_rss_kib": fused["peak_rss_kib"]}
    payload = {"schema": 1, "platform": platform.platform(), "python": sys.version.split()[0],
               "shape": {"in": args.in_features, "out": args.out_features,
                         "batch": args.batch, "group_size": args.group_size},
               "warmup": args.warmup, "repeats": args.repeats,
               "results": results, "summary": summary}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-features", type=int, default=4096)
    ap.add_argument("--out-features", type=int, default=14336)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--out", default="benchmarks/quant_results.json")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--method", choices=METHODS, default="awq", help=argparse.SUPPRESS)
    ap.add_argument("--fused", choices=["yes", "no"], default="yes", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if min(args.in_features, args.out_features, args.group_size, args.batch, args.repeats) <= 0 or args.warmup < 0:
        ap.error("dimensions, batch and repeats must be positive; warmup must be nonnegative")
    if args.child:
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
