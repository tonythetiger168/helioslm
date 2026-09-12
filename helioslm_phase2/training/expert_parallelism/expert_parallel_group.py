"""Expert Parallel Group management."""
import torch.distributed as dist
_EP_GROUP = None
_EP_WORLD_SIZE = None
_EP_RANK = None

def initialize_expert_parallel_group(ep_size: int):
    global _EP_GROUP, _EP_WORLD_SIZE, _EP_RANK
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_ep_groups = world_size // ep_size
    for i in range(num_ep_groups):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _EP_GROUP = group
            _EP_WORLD_SIZE = len(ranks)
            _EP_RANK = ranks.index(rank)
    print(f"EP init: rank={rank}, ep_rank={_EP_RANK}, ep_size={_EP_WORLD_SIZE}")

def get_ep_group():
    if _EP_GROUP is None:
        raise RuntimeError("EP group not initialized")
    return _EP_GROUP

def get_ep_world_size() -> int:
    return _EP_WORLD_SIZE if _EP_WORLD_SIZE is not None else 1
def get_ep_rank() -> int:
    return _EP_RANK if _EP_RANK is not None else 0

def get_expert_assignment(num_experts: int, ep_world_size: int, ep_rank: int):
    experts_per_rank = num_experts // ep_world_size
    start = ep_rank * experts_per_rank
    return list(range(start, start + experts_per_rank))
