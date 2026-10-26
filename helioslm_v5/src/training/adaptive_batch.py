"""Micro-batch size from the GPU's memory (v5.161).

``get_adaptive_batch_size(device)``: 4 on a CUDA device with less than
30 GB (L4 22 GB, RTX 4090 24 GB), else 8 (the pilot default; also CPU).
A rule of thumb from the first L4 k3 run, which ran out of memory at
batch 8 before v5.157's ``--kda-recompute``; with recompute batch 8 fits
in 10-14 GB on a 24 GB card.

``fit_micro_batch(try_step, batch, grad_accum)`` measures instead: it
runs ``try_step(micro_batch)`` and on a CUDA out-of-memory error halves
the micro-batch and doubles the accumulation, so the tokens per
optimizer step stay the same. ``examples/pilot_100m.py --auto-batch``
uses it.
"""
import torch


def get_adaptive_batch_size(device, small_gb=30, small=4, large=8):
    device = torch.device(device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        if total / 1024 ** 3 < small_gb:
            return small
    return large


def fit_micro_batch(try_step, batch, grad_accum, log=print):
    """Largest ``batch / 2**i`` micro-batch for which ``try_step`` runs.

    Returns (micro_batch, grad_accum) with micro_batch * grad_accum ==
    batch * grad_accum (the input product). Raises if even 1 does not
    fit, or if ``batch`` is not a power of two times the result.
    """
    total = batch * grad_accum
    mb = batch
    while True:
        try:
            try_step(mb)
            return mb, total // mb
        except torch.cuda.OutOfMemoryError:
            if mb == 1 or mb % 2:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"micro-batch {mb} ran out of memory; trying {mb // 2} "
                f"with grad_accum {total // (mb // 2)}")
            mb //= 2
