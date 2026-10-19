"""Small single-GPU latency measurements; no whole-model quantization claim."""
import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_cuda import model_on  # noqa: E402


def measure(fn, device, warmup=2, repeats=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples, output_shapes = [], []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        output_shapes.append(list(out.shape))
        assert torch.isfinite(out).all()
        del out
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'output_shapes': output_shapes,
            'baseline_allocated_bytes': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'warmup': warmup, 'repeats': repeats}


def model_bench(device, dtype):
    model, _ = model_on(device, dtype)
    ids = torch.tensor([[4, 5, 6, 7] * 16], device=device)
    with torch.inference_mode():
        prefill = measure(lambda: model(ids)[0], device)
        decode = measure(lambda: model.generate(ids, max_new_tokens=16, temperature=0), device)
    prefill['input_tokens'] = 64
    counts = [shape[1] - 64 for shape in decode['output_shapes']]
    decode.update(prompt_tokens=64, max_new_tokens=16, actual_new_tokens=counts,
                  tokens_per_second=statistics.median(n * 1000 / ms for n, ms in zip(counts, decode['samples_ms'])),
                  scope='end-to-end generate including prefill; batch=1')
    return {'prefill': prefill, 'generate': decode}


def quant_bench(device, name):
    from helioslm_v5.src.quantization.standard_quant import AWQLinear, GPTQLinear, MXFP4Linear
    torch.manual_seed(23)
    linear = torch.nn.Linear(512, 1024, device=device)
    x = torch.randn(16, 512, device=device)
    if name == 'awq':
        mod = AWQLinear.from_linear(linear, group_size=128, activations=x)
    elif name == 'gptq':
        mod = GPTQLinear.from_linear(linear, group_size=128,
                                     calibration_data=torch.randn(96, 512, device=device), act_order=True)
    else:
        mod = MXFP4Linear.from_linear(linear, block_size=32)
    with torch.inference_mode():
        mod.fused = False
        ref = mod(x)
        unfused = measure(lambda: mod(x), device)
        mod.fused = True
        out = mod(x)
        error = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        assert torch.isfinite(out).all() and error < 1e-4
        fused = measure(lambda: mod(x), device)
    return {'shape': [16, 512, 1024], 'dtype': 'float32', 'max_scaled_error': error,
            'tolerance': 1e-4, 'unfused': unfused, 'fused': fused,
            'latency_ratio_unfused_over_fused': unfused['median_ms'] / fused['median_ms'],
            'scope': 'small Linear only; preparation excluded; no required speedup threshold'}


def fp8_profile(device):
    from helioslm_v5.src.training.fp8_trainer import FP8Linear
    mod = FP8Linear(128, 128).to(device)
    x = torch.randn(4, 128, device=device, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        out = mod(x)
        out.square().mean().backward()
        torch.cuda.synchronize(device)
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
    ops = sorted({row.key for row in prof.key_averages()})
    return {'device': str(device), 'input_dtype': str(x.dtype), 'output_dtype': str(out.dtype),
            'operators': ops, 'scope': 'CPU operator dispatch of a CUDA FP8Linear; not GPU kernel trace',
            'native_fp8_tensor_core_verified': False,
            'reason': 'Current FP8Linear uses grid simulation + STE + ordinary matmul, not a native FP8 GEMM.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    report = {'cuda_benchmark_passed': False, 'rows': [],
              'methodology': '2 warmups, 7 synchronized wall-clock samples; small lite model / Linear; no performance threshold',
              'unverified': ['multi-GPU/NCCL', 'native FP8 product kernels', 'large-model throughput']}
    if not torch.cuda.is_available():
        report['status'] = 'blocked_no_cuda'
        code = 2
    else:
        device = torch.device('cuda:0')
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['environment'] = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(device)}
        jobs = [('model_' + n, lambda d=getattr(torch, n): model_bench(device, d))
                for n in ('float32', 'float16', 'bfloat16')]
        jobs += [('quant_' + n, lambda n=n: quant_bench(device, n)) for n in ('awq', 'gptq', 'mxfp4')]
        jobs.append(('fp8_operator_profile', lambda: fp8_profile(device)))
        for name, fn in jobs:
            row = {'name': name}
            try:
                row.update(status='passed', metrics=fn())
            except Exception:
                row.update(status='failed', traceback=traceback.format_exc())
            report['rows'].append(row)
            print(name, row['status'], flush=True)
            torch.cuda.empty_cache()
        report['cuda_benchmark_passed'] = all(r['status'] == 'passed' for r in report['rows'])
        report['status'] = 'passed' if report['cuda_benchmark_passed'] else 'failed_or_partial'
        code = 0 if report['cuda_benchmark_passed'] else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
