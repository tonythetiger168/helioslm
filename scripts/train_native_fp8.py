"""Synthetic full HeliosLM training acceptance; single CUDA or torchrun NCCL."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.training.fp8_trainer import FP8Linear, FP8Trainer


def training_config(size, distributed=False):
    config = HeliosLMv5Config(size=size)
    if distributed:
        # This launcher replicates the entire model; it is data parallel,
        # not expert parallel. Every replica must route across all experts.
        config.moe.device_group_size = 1
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--size', choices=['lite', 'mid'], default='lite')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--out', type=Path, default=Path('extended_results/native_fp8_training.json'))
    args = ap.parse_args()
    if args.steps < 2 or args.tokens < 2:
        ap.error('steps and tokens must be >= 2')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Native FP8 requires CUDA')
    if world > 1 and not dist.is_nccl_available():
        raise RuntimeError('Multi-GPU mode requires NCCL-enabled Linux PyTorch')
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'status': 'running', 'world_size': world}),
                            encoding='utf-8')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    try:
        if world > 1:
            dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
        torch.manual_seed(1234)
        config = training_config(args.size, distributed=world > 1)
        model = HeliosLMv5(config).to(device).train()
        trainer = FP8Trainer(model, config, backend='native', distributed=world > 1)
        original = model.lm_head.weight.detach().clone()
        generator = torch.Generator(device=device).manual_seed(2026 + rank)
        losses = []
        start = time.monotonic()
        for step in range(args.steps):
            ids = torch.randint(0, config.vocab_size, (1, args.tokens),
                                device=device, generator=generator)
            loss = trainer.train_step({'input_ids': ids}, labels=ids)
            if not torch.isfinite(torch.tensor(loss)) or trainer.skipped_steps:
                raise RuntimeError('Nonfinite loss/gradient or skipped update')
            losses.append(loss)
            print(f'rank={rank} step={step + 1} loss={loss:.6f}', flush=True)
        torch.cuda.synchronize()
        delta = (model.lm_head.weight - original).abs().max().item()
        if delta <= 0:
            raise AssertionError('LM head did not update')
        max_replica_error = 0.0
        if world > 1:
            for param in model.parameters():
                reference = param.detach().clone()
                dist.broadcast(reference, 0)
                max_replica_error = max(max_replica_error,
                                        (param - reference).abs().max().item())
            error = torch.tensor(max_replica_error, device=device)
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            max_replica_error = error.item()
            if max_replica_error != 0:
                raise AssertionError('Replica weights diverged')
        report = {'status': 'passed', 'backend': 'native', 'world_size': world,
                  'collective_backend': 'nccl' if world > 1 else None,
                  'size': args.size, 'steps': args.steps, 'tokens': args.tokens,
                  'losses_rank0': losses, 'seconds': time.monotonic() - start,
                  'lm_head_max_update': delta, 'max_replica_error': max_replica_error,
                  'native_fp8_layers': sum(isinstance(m, FP8Linear) for m in model.modules()),
                  'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                  'scope': 'Synthetic acceptance; not convergence or throughput certification'}
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print('Status: passed', flush=True)
    except Exception as exc:
        if rank == 0:
            args.out.write_text(json.dumps({'status': 'failed', 'error': repr(exc),
                                            'world_size': world}), encoding='utf-8')
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
