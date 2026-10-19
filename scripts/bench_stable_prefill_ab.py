"""Interleaved stable-prefill A/B against the pre-optimization token schedule.

The reference is intentionally limited to the default text model layout.
Both modes use identical weights, inputs and low-precision arithmetic.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.attention.mla import MLA
from helioslm_v5.src.model_v5 import HeliosLMv5


def reference_projection(self, hidden, positions, position_start=None):
    qn, qr = self._project_queries(hidden, positions)
    c, kr = self._project_kv(hidden, positions)
    return qn, qr, c, kr


@contextmanager
def reference_projections(model):
    saved = []
    try:
        for module in model.modules():
            if isinstance(module, MLA):
                saved.append((module, module._project_new_tokens))
                module._project_new_tokens = types.MethodType(reference_projection, module)
        yield
    finally:
        for module, original in saved:
            module._project_new_tokens = original


def reference_prefill(model, ids):
    past, logits, hidden = None, [], []
    with reference_projections(model):
        for i in range(ids.shape[1]):
            out, state, past = model._forward_impl(
                ids[:, i:i + 1], past_key_values=past, use_cache=True)
            logits.append(out)
            hidden.append(state)
    return torch.cat(logits, 1), torch.cat(hidden, 1), past


def tensors(output):
    return list(output[:2]) + [t for cache in output[2] for t in cache]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', choices=['lite', 'mid'], default='mid')
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.tokens < 1 or args.warmup < 0 or args.repeats < 1:
        ap.error('tokens/repeats must be positive and warmup nonnegative')
    if not torch.cuda.is_available():
        ap.error('This A/B requires CUDA')
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    config = HeliosLMv5Config(size=args.size)
    model = HeliosLMv5(config).cuda().eval()
    rows = []
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            model.to(dtype=dtype)
            torch.manual_seed(83)
            ids = torch.randint(4, config.vocab_size, (1, args.tokens), device='cuda')
            functions = {'reference': lambda: reference_prefill(model, ids),
                         'optimized': lambda: model(ids, use_cache=True)}
            for fn in functions.values():
                for _ in range(args.warmup):
                    fn()
            samples = {key: [] for key in functions}
            expected = None
            for i in range(args.repeats):
                order = ('reference', 'optimized') if i % 2 == 0 else ('optimized', 'reference')
                for tag in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = functions[tag]()
                    torch.cuda.synchronize()
                    samples[tag].append((time.perf_counter() - start) * 1000)
                    if expected is None:
                        expected = [t.clone() for t in tensors(output)]
                    if not all(torch.equal(a, b) for a, b in zip(tensors(output), expected)):
                        raise AssertionError('Logits/hidden/cache changed in ' + tag)
            medians = {k: statistics.median(v) for k, v in samples.items()}
            row = {'dtype': str(dtype), 'samples_ms': samples, 'median_ms': medians,
                   'latency_reduction_percent': 100 * (1 - medians['optimized'] / medians['reference']),
                   'bit_identical_logits_hidden_cache': True}
            rows.append(row)
            print(json.dumps(row), flush=True)
    source = ['helioslm_v5/src/model_v5.py', 'helioslm_v5/src/attention/mla.py',
              'helioslm_v5/src/moe/sigmoid_moe.py', 'scripts/bench_stable_prefill_ab.py']
    report = {'schema': 'windows-cuda-prefill-ab/1', 'verification_tier': 'measured',
              'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
              'size': args.size, 'batch': 1, 'tokens': args.tokens, 'warmup': args.warmup,
              'repeats': args.repeats, 'rows': rows,
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source},
              'scope': 'Untrained model; interleaved synchronized wall clock on a shared desktop. '
                       'No isolated throughput, convergence, or original parallel-prefill claim.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
