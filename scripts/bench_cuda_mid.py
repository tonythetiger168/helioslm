"""Synchronized mid-model measurements; no model-quality claim."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 1024])
    ap.add_argument('--skip-generation', action='store_true')
    args = ap.parse_args()
    if args.warmup < 0 or args.repeats < 1 or min(args.lengths) < 1:
        ap.error('warmup must be >= 0; repeats and lengths must be positive')
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    device = torch.device('cuda:0')
    cfg = HeliosLMv5Config(size='mid')
    model = HeliosLMv5(cfg).to(device=device, dtype=getattr(torch, args.dtype)).eval()
    report = {'status': 'running', 'gpu': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'dtype': args.dtype, 'batch': 1,
              'parameters': sum(p.numel() for p in model.parameters()),
              'warmup': args.warmup, 'repeats': args.repeats, 'rows': [],
              'scope': 'Untrained 360M model; synchronized end-to-end wall clock. '
                       'Other desktop applications remain running; not an isolated throughput comparison.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def measure(name, fn, input_tokens, generation=False):
        with torch.inference_mode():
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples, counts = [], []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
                if generation:
                    counts.append(out.shape[1] - input_tokens)
                    assert 0 < counts[-1] <= 16
                else:
                    assert torch.isfinite(out).all()
                del out
        row = {'name': name, 'input_tokens': input_tokens, 'samples_ms': samples,
               'median_ms': statistics.median(samples),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        if generation:
            row.update(actual_new_tokens=counts,
                       median_tokens_per_second=statistics.median(
                           n * 1000 / ms for n, ms in zip(counts, samples)),
                       scope='generate includes prompt prefill; max_new_tokens=16')
        else:
            row['prefill_tokens_per_second'] = input_tokens * 1000 / row['median_ms']
        report['rows'].append(row)
        print(name + ': %.2fms' % row['median_ms'], flush=True)

    for length in args.lengths:
        ids = torch.randint(4, cfg.vocab_size, (1, length), device=device)
        measure('prefill_%d' % length, lambda: model(ids)[0], length)
    if not args.skip_generation:
        prompt = torch.randint(4, cfg.vocab_size, (1, 128), device=device)
        measure('generate_128_plus_16',
                lambda: model.generate(prompt, max_new_tokens=16, temperature=0), 128, True)
    report['status'] = 'passed'
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
