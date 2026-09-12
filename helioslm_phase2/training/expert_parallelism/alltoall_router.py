"""All-to-All communication for expert parallel routing."""
import torch
import torch.distributed as dist
from typing import Tuple, Optional

def all_to_all(input_tensor: torch.Tensor, output_split_sizes=None, input_split_sizes=None, group=None):
    if group is None:
        from .expert_parallel_group import get_ep_group
        group = get_ep_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_tensor
    if hasattr(dist, "all_to_all_single"):
        if output_split_sizes is None or input_split_sizes is None:
            output = torch.empty_like(input_tensor)
            dist.all_to_all_single(output, input_tensor, group=group)
            return output
        output = input_tensor.new_empty(sum(output_split_sizes), input_tensor.shape[1])
        dist.all_to_all_single(output, input_tensor, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group)
        return output
    # Fallback
    rank = dist.get_rank(group)
    output_tensors = []
    for i in range(world_size):
        if i == rank:
            output_tensors.append(input_tensor)
        else:
            ss = input_split_sizes[i] if input_split_sizes else input_tensor.shape[0] // world_size
            rs = output_split_sizes[i] if output_split_sizes else input_tensor.shape[0] // world_size
            recv_buf = input_tensor.new_empty(rs, input_tensor.shape[1])
            send_buf = input_tensor[:ss]
            dist.send(send_buf, dst=i, group=group)
            dist.recv(recv_buf, src=i, group=group)
            output_tensors.append(recv_buf)
    return torch.cat(output_tensors, dim=0)

class AllToAllRouter:
    def __init__(self, num_experts: int, ep_world_size: int, ep_rank: int):
        self.num_experts = num_experts
        self.ep_world_size = ep_world_size
        self.ep_rank = ep_rank
        self.experts_per_rank = num_experts // ep_world_size
        self.local_start = ep_rank * self.experts_per_rank
        self.local_end = self.local_start + self.experts_per_rank

    def dispatch(self, hidden_states: torch.Tensor, router_probs: torch.Tensor, expert_indices: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        top_k = expert_indices.shape[1]
        flat_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])
        flat_probs = router_probs.reshape(-1)
        flat_experts = expert_indices.reshape(-1)
        target_ranks = flat_experts // self.experts_per_rank
        sorted_order = torch.argsort(target_ranks)
        sorted_hidden = flat_hidden[sorted_order]
        sorted_probs = flat_probs[sorted_order]
        sorted_experts = flat_experts[sorted_order]
        split_sizes = [(target_ranks == r).sum().item() for r in range(self.ep_world_size)]
        from .expert_parallel_group import get_ep_group
        disp_hidden = all_to_all(sorted_hidden, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=get_ep_group())
        disp_probs = all_to_all(sorted_probs.unsqueeze(-1), output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=get_ep_group()).squeeze(-1)
        return disp_hidden, disp_probs, sorted_experts, split_sizes

    def combine(self, expert_output: torch.Tensor, split_sizes: list, num_tokens: int):
        from .expert_parallel_group import get_ep_group
        combined = all_to_all(expert_output, output_split_sizes=split_sizes, input_split_sizes=split_sizes, group=get_ep_group())
        return combined[:num_tokens]
