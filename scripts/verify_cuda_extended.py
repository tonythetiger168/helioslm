"""Bounded mid-model CUDA acceptance and soak; hardware limits remain explicit."""
import argparse
import datetime
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--soak-seconds', type=int, default=600)
    args = ap.parse_args()
    if args.soak_seconds < 1:
        ap.error('--soak-seconds must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'running', 'checks': [], 'soak': {},
              'environment': {'python': platform.python_version(),
                              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                              'gpu_count': torch.cuda.device_count(),
                              'nccl_available': torch.distributed.is_nccl_available()},
              'limits': ['Bounded synthetic test, not a production training quality result.',
                         'SGD, batch=1; no full-model AdamW or large-batch claim.',
                         'FP8 training check uses a lite synthetic model, not mid convergence.',
                         'No multi-GPU or multi-hour stability claim.']}

    def save():
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def check(name, fn):
        start = time.monotonic()
        try:
            row = dict(name=name, status='passed')
            row.update(fn())
        except Exception as exc:
            row = dict(name=name, status='failed', error=repr(exc),
                       traceback=traceback.format_exc())
        row['seconds'] = time.monotonic() - start
        report['checks'].append(row)
        save()
        print(name + ': ' + row['status'], flush=True)
        return row

    if not torch.cuda.is_available():
        report.update(status='blocked_no_cuda')
        save()
        return 2
    device = torch.device('cuda:0')
    report['environment'].update(gpu=torch.cuda.get_device_name(0),
                                 compute_capability=list(torch.cuda.get_device_capability(0)),
                                 initial_free_total_bytes=list(torch.cuda.mem_get_info(0)))
    report['checks'].append({'name': 'multi_gpu_nccl', 'status': 'blocked',
                             'reason': 'Run torchrun --standalone --nproc_per_node=2 '
                                       'scripts/train_native_fp8.py on NCCL-enabled multi-GPU Linux.',
                             'observed_gpu_count': torch.cuda.device_count(),
                             'observed_nccl_available': torch.distributed.is_nccl_available()})

    def native_fp8():
        a = (torch.randn(64, 128, device=device) * 0.1).to(torch.float8_e4m3fn)
        b = (torch.randn(64, 128, device=device) * 0.1).to(torch.float8_e4m3fn).t()
        scale = torch.ones((), device=device)
        try:
            out = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                   out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            if any(s in message for s in ('not supported', 'only supported',
                                          'not implemented', 'no kernel image',
                                          'cublas_status_not_supported')):
                return {'status': 'blocked', 'reason': str(exc),
                        'operator': 'aten::_scaled_mm'}
            raise
        ref = a.float() @ b.float()
        assert out.is_cuda and torch.isfinite(out).all()
        assert torch.allclose(out.float(), ref, rtol=0.02, atol=0.02)
        return {'operator': 'aten::_scaled_mm', 'input_dtype': str(a.dtype),
                'output_dtype': str(out.dtype), 'max_abs_error': (out.float()-ref).abs().max().item(),
                'shape_mkn': [64, 128, 64]}

    # A supported primitive and the product's implementation are separate claims.
    check('native_fp8_gemm_probe', native_fp8)
    def native_training():
        from helioslm_v5.src.training.fp8_trainer import FP8Linear, FP8Trainer
        config = HeliosLMv5Config(size='lite')
        fp8_model = HeliosLMv5(config).to(device).train()
        trainer = FP8Trainer(fp8_model, config, backend='native')
        before = fp8_model.lm_head.weight.detach().clone()
        losses = []
        for _ in range(3):
            ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
            losses.append(trainer.train_step({'input_ids': ids}, labels=ids))
        delta = (fp8_model.lm_head.weight - before).abs().max().item()
        assert trainer.skipped_steps == 0 and delta > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)
        return {'size': 'lite', 'steps': 3, 'losses': losses,
                'lm_head_max_update': delta, 'operator': 'aten::_scaled_mm',
                'native_layers': sum(isinstance(m, FP8Linear) for m in fp8_model.modules())}

    check('product_native_fp8_training', native_training)
    save()
    model = None
    try:
        torch.manual_seed(20261019)
        torch.cuda.manual_seed_all(20261019)
        cfg = HeliosLMv5Config(size='mid')
        model = HeliosLMv5(cfg).to(device).eval()
        count = sum(p.numel() for p in model.parameters())
        assert count == 359992352, count
        report['mid_parameters'] = count
        report['inference_policy'] = {
            'stable_low_precision_prefill': cfg.stable_low_precision_prefill,
            'scope': 'eval/no-grad monotonic text, same batch layout',
            'tradeoff': 'FP16/BF16 prefill uses one-token scheduling; slower than parallel prefill.'}
        report['model_source_sha256'] = hashlib.sha256(
            (ROOT / 'helioslm_v5/src/model_v5.py').read_bytes()).hexdigest()

        def inference(dtype):
            model.to(dtype=dtype).eval()
            tol = {torch.float32: 1e-4, torch.float16: 0.02, torch.bfloat16: 0.05}[dtype]
            lengths = []
            with torch.inference_mode():
                for length in (32, 128, 512, 1024):
                    ids = torch.randint(4, cfg.vocab_size, (1, length), device=device)
                    logits, _, _ = model(ids)
                    assert logits.device == device and logits.dtype == dtype
                    assert torch.isfinite(logits).all()
                    lengths.append(length)
                    del logits
                ids = torch.randint(4, cfg.vocab_size, (1, 32), device=device)
                full = model(ids)[0]
                past, steps = None, []
                for i in range(ids.size(1)):
                    logits, _, past = model(ids[:, i:i+1], past_key_values=past, use_cache=True)
                    steps.append(logits)
                cached = torch.cat(steps, dim=1)
                cache_dtypes = sorted({str(t.dtype) for layer in past for t in layer})
                assert cached.dtype == dtype
                assert all(t.dtype == dtype for layer in past for t in layer)
                scaled = ((cached.float()-full.float()).abs().max() /
                          full.float().abs().max().clamp_min(1e-8)).item()
                cache_passed = scaled < tol
            torch.cuda.synchronize()
            return {'status': 'passed' if cache_passed else 'failed',
                    'forward_finite_passed': True, 'cache_parity_passed': cache_passed,
                    'dtype': str(dtype), 'batch': 1, 'sequence_lengths': lengths,
                    'cache_dtypes': cache_dtypes,
                    'cache_scaled_max_error': scaled, 'cache_tolerance': tol}

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            row = check('mid_inference_' + str(dtype).split('.')[-1], lambda d=dtype: inference(d))
            # A parity failure is recorded without preventing other precisions or the soak.

        model.float().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        ids = torch.randint(4, cfg.vocab_size, (1, 64), device=device)

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(ids)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                                   ids[:, 1:].reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
            grad = model.lm_head.weight.grad
            assert grad is not None and grad.abs().max() > 0
            before = model.lm_head.weight.detach().clone()
            optimizer.step()
            changed = (before - model.lm_head.weight).abs().max().item()
            assert changed > 0
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return {'loss': float(loss.detach()), 'max_head_update': changed,
                    'optimizer': 'SGD', 'dtype': 'float32', 'batch': 1, 'sequence': 64}

        row = check('mid_cuda_forward_backward_update', train_step)
        if row['status'] != 'passed':
            raise RuntimeError('Mid training check failed; see recorded traceback')

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        baseline = torch.cuda.memory_allocated()
        report['soak'] = {'status': 'running', 'target_seconds': args.soak_seconds,
                          'inference_iterations': 0, 'training_iterations': 0,
                          'baseline_allocated_bytes': baseline, 'samples': []}
        index, next_sample = 0, 0.0
        with torch.no_grad():
            reference = model.eval()(ids)[0].detach().clone()
        while time.monotonic() - start < args.soak_seconds:
            if index % 10 == 9:
                model.train()
                train_step()
                report['soak']['training_iterations'] += 1
                with torch.no_grad():
                    reference = model.eval()(ids)[0].detach().clone()
            else:
                model.eval()
                with torch.inference_mode():
                    logits = model(ids)[0]
                    assert torch.isfinite(logits).all()
                    assert torch.equal(logits, reference), 'Repeated inference changed without weight update'
                    del logits
                report['soak']['inference_iterations'] += 1
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            if elapsed >= next_sample:
                sample = {'elapsed_seconds': elapsed, 'allocated_bytes': torch.cuda.memory_allocated(),
                          'reserved_bytes': torch.cuda.memory_reserved(),
                          'free_total_bytes': list(torch.cuda.mem_get_info())}
                report['soak']['samples'].append(sample)
                report['soak']['elapsed_seconds'] = elapsed
                save()
                print('soak: %.1fs, inference=%d, training=%d' % (
                    elapsed, report['soak']['inference_iterations'],
                    report['soak']['training_iterations']), flush=True)
                next_sample = elapsed + 30
            index += 1
        optimizer.zero_grad(set_to_none=True)
        del reference
        gc.collect()
        torch.cuda.synchronize()
        final = torch.cuda.memory_allocated()
        # Parameters/buffers persist; outputs/gradients must not grow per iteration.
        assert final <= baseline + 64 * 1024**2, (baseline, final)
        report['soak'].update(status='passed', elapsed_seconds=time.monotonic()-start,
                              final_allocated_bytes=final,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        report['checks'].append({'name': 'mid_soak_or_setup', 'status': 'failed',
                                 'error': repr(exc), 'traceback': traceback.format_exc()})
        if report['soak'].get('status') == 'running':
            report['soak']['status'] = 'failed'
    report['status'] = ('failed' if any(r['status'] == 'failed' for r in report['checks'])
                        else 'passed_available_checks_with_blocked_scope')
    save()
    print('Status: ' + report['status'], flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
