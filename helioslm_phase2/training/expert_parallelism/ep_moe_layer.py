"""Expert Parallel MoE Layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert_parallel_group import get_ep_group, get_ep_world_size, get_ep_rank, get_expert_assignment
from .alltoall_router import AllToAllRouter
from .load_balancer import compute_load_balance_loss, LoadBalancer

class EPMoELayer(nn.Module):
    """
    MoE layer with Expert Parallelism.

    Each GPU holds a subset of experts. Tokens are dispatched via All-to-All.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.num_activated_experts
        self.ep_world_size = get_ep_world_size()
        self.ep_rank = get_ep_rank()

        # Router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Local experts (only the ones assigned to this GPU)
        self.local_expert_ids = get_expert_assignment(self.num_experts, self.ep_world_size, self.ep_rank)
        self.experts = nn.ModuleDict({
            str(eid): nn.Sequential(
                nn.Linear(self.hidden_size, config.moe.expert_hidden_size),
                nn.GELU(),
                nn.Linear(config.moe.expert_hidden_size, self.hidden_size),
            )
            for eid in self.local_expert_ids
        })

        # All-to-All router
        self.router_op = AllToAllRouter(self.num_experts, self.ep_world_size, self.ep_rank)
        self.load_balancer = LoadBalancer(self.num_experts)

        # Shared experts (always local, not parallelized)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, config.moe.expert_hidden_size),
                nn.GELU(),
                nn.Linear(config.moe.expert_hidden_size, self.hidden_size),
            )
            for _ in range(config.moe.num_shared_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, task_type=None):
        B, seq, h = hidden_states.shape
        flat = hidden_states.view(-1, h)

        # Router logits
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Dispatch via All-to-All
        disp_hidden, disp_probs, disp_experts, split_sizes = self.router_op.dispatch(
            flat, top_k_probs, top_k_indices
        )

        # Compute local expert outputs
        output = torch.zeros_like(disp_hidden)
        for eid in self.local_expert_ids:
            mask = (disp_experts == eid)
            if mask.any():
                expert_input = disp_hidden[mask]
                expert_output = self.experts[str(eid)](expert_input)
                expert_weight = disp_probs[mask].unsqueeze(-1)
                output[mask] = expert_weight * expert_output

        # Combine back
        combined = self.router_op.combine(output, split_sizes, flat.shape[0])
        combined = combined.view(B, seq, h)

        # Shared experts
        for shared in self.shared_experts:
            combined = combined + shared(hidden_states)

        # Load balance loss
        aux_loss = compute_load_balance_loss(router_probs, top_k_indices, self.num_experts)

        return combined, aux_loss
