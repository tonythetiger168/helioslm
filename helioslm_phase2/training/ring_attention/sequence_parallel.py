"""Sequence splitting and gathering for context parallelism."""
import torch
import torch.distributed as dist
from .context_parallel_group import get_cp_group, get_cp_world_size, get_cp_rank

def split_sequence(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Split sequence dimension across CP group."""
    cp_size = get_cp_world_size()
    cp_rank = get_cp_rank()
    seq_len = tensor.size(dim)
    assert seq_len % cp_size == 0, f"seq_len {seq_len} must be divisible by cp_size {cp_size}"
    chunk_size = seq_len // cp_size
    start = cp_rank * chunk_size
    end = start + chunk_size
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)].contiguous()

def gather_sequence(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather sequence chunks from all CP ranks."""
    cp_group = get_cp_group()
    cp_size = get_cp_world_size()
    if cp_size == 1:
        return tensor

    # All-gather along sequence dimension
    tensor_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    dist.all_gather(tensor_list, tensor, group=cp_group)
    return torch.cat(tensor_list, dim=dim)
