"""Context Parallel group management for Ring Attention."""
import torch.distributed as dist

_CP_GROUP = None
_CP_WORLD_SIZE = None
_CP_RANK = None

def initialize_context_parallel_group(cp_size: int):
    global _CP_GROUP, _CP_WORLD_SIZE, _CP_RANK
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_cp_groups = world_size // cp_size
    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _CP_GROUP = group
            _CP_WORLD_SIZE = len(ranks)
            _CP_RANK = ranks.index(rank)
    print(f"CP init: rank={rank}, cp_rank={_CP_RANK}, cp_size={_CP_WORLD_SIZE}")

def get_cp_group():
    if _CP_GROUP is None:
        raise RuntimeError("CP group not initialized")
    return _CP_GROUP

def get_cp_world_size() -> int:
    return _CP_WORLD_SIZE if _CP_WORLD_SIZE is not None else 1
def get_cp_rank() -> int:
    return _CP_RANK if _CP_RANK is not None else 0
