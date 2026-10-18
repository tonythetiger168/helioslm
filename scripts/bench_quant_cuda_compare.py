"""Compare old fused, vectorized tiled and dense-reference quantized Linear."""
import argparse
import copy
import importlib.util
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helioslm_v5.src.quantization import standard_quant as current  # noqa: E402


def timing(fn, device):
    for _ in range(2):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(7):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        out = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
        assert torch.isfinite(out).all()
        del out
    return {'samples_ms': values, 'median_ms': statistics.median(values),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}


def job(old, method, width, outputs, device):
    torch.manual_seed(23)
    lin = torch.nn.Linear(width, outputs)
    calibration = torch.randn(96, width)
    if method == 'awq':
        original = old.AWQLinear.from_linear(lin, group_size=128, activations=calibration)
    elif method == 'gptq':
        original = old.GPTQLinear.from_linear(lin, group_size=128, calibration_data=calibration, act_order=True)
    else:
        original = old.MXFP4Linear.from_linear(lin, block_size=32)
    new = copy.deepcopy(original)
    # Identical registered tensors and all quantization metadata; only code differs.
    new.__class__ = getattr(current, type(original).__name__)
    original, new = original.to(device), new.to(device)
    x = torch.randn(16, width).to(device)
    original.fused = False
    with torch.inference_mode():
        reference = original(x)
        original.fused = True
        bias = new.bias
        def scaled_input():
            return x / new.act_scale if method == 'awq' and new.act_scale is not None else x
        # Direct tile call also permits an explicitly labeled CPU harness check.
        arms = {'legacy_fused': lambda: original(x),
                'tiled': lambda: new._tiled_forward(scaled_input(), bias),
                'unfused': lambda: torch.nn.functional.linear(
                    scaled_input(), new._dequantize(), bias)}
        errors = {}
        for name, fn in arms.items():
            out = fn()
            error = (out - reference).abs().max().item() / (reference.abs().max().item() + 1e-8)
            assert torch.isfinite(out).all() and error < 1e-4, (method, name, error)
            errors[name] = error
        trials, combined = [], {name: [] for name in arms}
        for trial in range(3):
            order = list(arms)
            random.Random(400 + trial).shuffle(order)
            rows = {}
            for name in order:
                rows[name] = timing(arms[name], device)
                combined[name].extend(rows[name]['samples_ms'])
            trials.append({'order': order, 'arms': rows})
    medians = {name: statistics.median(samples) for name, samples in combined.items()}
    return {'shape': [16, width, outputs], 'max_scaled_errors': errors,
            'trials': trials, 'median_ms_21_samples': medians,
            'legacy_over_tiled_latency': medians['legacy_fused'] / medians['tiled'],
            'unfused_over_tiled_latency': medians['unfused'] / medians['tiled']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cpu-smoke', action='store_true')
    args = ap.parse_args()
    report = {'cuda_benchmark_passed': False, 'rows': [],
              'scope': '3 trials, randomized arm order, 2 warmups +7 synchronized samples per arm/trial; same process and tensors; no minimum speedup asserted',
              'memory_scope': 'PyTorch allocated peak includes both module copies and retained reference, not total VRAM; reserved allocator is shared',
              'preparation': 'calibration on CPU excluded from timing; identical tensors copied to each arm'}
    if not args.cpu_smoke and not torch.cuda.is_available():
        report['status'] = 'blocked_no_cuda'
        code = 2
    else:
        device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
        torch.set_num_threads(1)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            report['gpu'] = torch.cuda.get_device_name(device)
        report.update(torch=str(torch.__version__), device=str(device))
        spec = importlib.util.spec_from_file_location('quant_baseline', args.baseline)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        shapes = [(65, 33)] if args.cpu_smoke else [(512, 1024), (1031, 65)]
        for width, outputs in shapes:
            for method in ('awq', 'gptq', 'mxfp4'):
                row = {'method': method, 'input_features': width, 'output_features': outputs}
                try:
                    row.update(status='passed', metrics=job(old, method, width, outputs, device))
                except Exception:
                    row.update(status='failed', traceback=traceback.format_exc())
                report['rows'].append(row)
                print(method, width, row['status'], flush=True)
        ok = all(r['status'] == 'passed' for r in report['rows'])
        report.update(status=('cpu_smoke_passed' if args.cpu_smoke else 'cuda_comparison_passed') if ok else 'failed_or_partial',
                      cuda_benchmark_passed=ok and device.type == 'cuda')
        code = 0 if ok else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
