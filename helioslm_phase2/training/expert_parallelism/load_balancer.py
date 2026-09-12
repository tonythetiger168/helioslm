"""Load balancing for Expert Parallelism."""
import torch
import torch.nn as nn

def compute_load_balance_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int, alpha: float = 0.01) -> torch.Tensor:
    """
    Compute auxiliary load balance loss.

    Encourages uniform distribution of tokens across experts.

    Loss = alpha * N * sum_j (f_j * P_j)^2
    where f_j is the fraction of tokens routed to expert j,
    and P_j is the average router probability for expert j.
    """
    # Fraction of tokens routed to each expert
    router_mask = torch.zeros_like(router_probs).scatter_(1, expert_indices, 1.0)
    tokens_per_expert = router_mask.sum(dim=0)  # [num_experts]
    f = tokens_per_expert / tokens_per_expert.sum()

    # Average router probability per expert
    P = router_probs.mean(dim=0)  # [num_experts]

    # Load balance loss
    loss = alpha * num_experts * (f * P).sum()
    return loss

class LoadBalancer:
    """Dynamic load balancer that monitors and adjusts expert capacity."""

    def __init__(self, num_experts: int, capacity_factor: float = 1.25, min_capacity: int = 4):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.expert_history = {i: [] for i in range(num_experts)}

    def compute_capacity(self, num_tokens: int, top_k: int) -> int:
        """Compute per-expert capacity."""
        base_capacity = int(self.capacity_factor * num_tokens * top_k / self.num_experts)
        return max(base_capacity, self.min_capacity)

    def update_history(self, expert_indices: torch.Tensor):
        """Update token distribution history."""
        for eid in range(self.num_experts):
            count = (expert_indices == eid).sum().item()
            self.expert_history[eid].append(count)
            # Keep last 1000 steps
            if len(self.expert_history[eid]) > 1000:
                self.expert_history[eid].pop(0)

    def get_imbalance_ratio(self) -> float:
        """Compute current load imbalance ratio (max/min)."""
        recent_counts = [sum(self.expert_history[e][-100:]) for e in range(self.num_experts)]
        nonzero = [c for c in recent_counts if c > 0]
        if not nonzero:
            return 1.0
        return max(nonzero) / min(nonzero)
