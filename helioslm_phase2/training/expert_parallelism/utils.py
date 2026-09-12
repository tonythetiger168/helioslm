"""Utilities for expert parallelism."""
import torch
import torch.distributed as dist

def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int):
    """Split tensor along last dimension for model parallelism."""
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size(last_dim)
    assert last_dim_size % num_partitions == 0
    stride = last_dim_size // num_partitions
    tensor_list = torch.split(tensor, stride, dim=last_dim)
    return tensor_list

def gather_from_model_parallel_region(tensor: torch.Tensor):
    """Gather tensor from model parallel region."""
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    dim_size = list(tensor.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(output, tensor)
    return output
