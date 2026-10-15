#!/usr/bin/env python3
"""
Benchmark script for comparing fused vs unfused quantization performance.
Compares wall time and peak RSS for AWQ/GPTQ/MXFP4 quantization methods.
"""

import argparse
import json
import os
import sys
import time
import torch

# --- CROSS-PLATFORM MEMORY CHECKER ---
try:
    # Untuk Linux/macOS
    import resource
    def get_peak_rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:
    # Untuk Windows
    import psutil
    def get_peak_rss_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
# ---------------------------------------


def benchmark_quantization(shape=(4096, 14336), group_size=128, num_runs=10, method="awq"):
    """
    Benchmark fused vs unfused quantization.
    
    Args:
        shape: Tuple of (in_features, out_features)
        group_size: Group size for group-wise quantization
        num_runs: Number of runs for median calculation
        method: Quantization method (awq, gptq, or mxfp4)
    """
    in_features, out_features = shape
    
    results = {
        "shape": shape,
        "group_size": group_size,
        "method": method,
        "num_runs": num_runs,
        "fused": {},
        "unfused": {}
    }
    
    # Create dummy weight tensor
    weight = torch.randn(in_features, out_features, dtype=torch.float16)
    
    # Benchmark UNFUSED (mod.fused = False)
    print(f"\n{'='*60}")
    print(f"Benchmarking UNFUSED {method.upper()} quantization...")
    print(f"{'='*60}")
    
    unfused_times = []
    unfused_rss = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Simulate unfused quantization (dequantize then matmul)
        # In real implementation, this would call the actual quantization kernel
        with torch.no_grad():
            # Placeholder: simulate computation
            _ = weight * 0.1  # Replace with actual quantization op
            _ = torch.matmul(weight.t(), weight)
        
        elapsed = time.perf_counter() - start_time
        unfused_times.append(elapsed)
        unfused_rss.append(get_peak_rss_mb())
        
        if i < 3:  # Print first 3 runs
            print(f"  Run {i+1}: {elapsed*1000:.2f} ms, RSS: {unfused_rss[-1]:.2f} MB")
    
    # Calculate median for unfused
    unfused_times.sort()
    median_unfused_time = unfused_times[len(unfused_times)//2]
    median_unfused_rss = max(unfused_rss)
    
    results["unfused"]["median_time_ms"] = median_unfused_time * 1000
    results["unfused"]["peak_rss_mb"] = median_unfused_rss
    
    # Benchmark FUSED (mod.fused = True)
    print(f"\n{'='*60}")
    print(f"Benchmarking FUSED {method.upper()} quantization...")
    print(f"{'='*60}")
    
    fused_times = []
    fused_rss = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Simulate fused quantization (dequant×matmul in one kernel)
        with torch.no_grad():
            # Placeholder: simulate fused operation
            _ = torch.matmul(weight.t(), weight) * 0.1  # Replace with actual fused op
        
        elapsed = time.perf_counter() - start_time
        fused_times.append(elapsed)
        fused_rss.append(get_peak_rss_mb())
        
        if i < 3:
            print(f"  Run {i+1}: {elapsed*1000:.2f} ms, RSS: {fused_rss[-1]:.2f} MB")
    
    # Calculate median for fused
    fused_times.sort()
    median_fused_time = fused_times[len(fused_times)//2]
    median_fused_rss = max(fused_rss)
    
    results["fused"]["median_time_ms"] = median_fused_time * 1000
    results["fused"]["peak_rss_mb"] = median_fused_rss
    
    # Calculate speedup
    speedup = median_unfused_time / median_fused_time if median_fused_time > 0 else 1.0
    results["speedup"] = speedup
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS - {method.upper()}")
    print(f"{'='*60}")
    print(f"Shape: {shape}, Group Size: {group_size}")
    print(f"Unfused: {median_unfused_time*1000:.2f} ms, {median_unfused_rss:.2f} MB")
    print(f"Fused:   {median_fused_time*1000:.2f} ms, {median_fused_rss:.2f} MB")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused vs unfused quantization")
    parser.add_argument("--shape", type=int, nargs=2, default=[4096, 14336],
                       help="Input shape (in_features, out_features)")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for group-wise quantization")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of runs for median calculation")
    parser.add_argument("--method", type=str, choices=["awq", "gptq", "mxfp4"], default="awq",
                       help="Quantization method")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_quantization(
        shape=tuple(args.shape),
        group_size=args.group_size,
        num_runs=args.runs,
        method=args.method
    )
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()