"""Sigmoid MoE with Device-Limited Routing - DeepSeek-V3 Style

Key properties:
- Sigmoid routing (independent per-expert scores, no softmax competition).
- Aux-free load balancing (DeepSeek-V3 style): ``route_bias`` is added ONLY
  for top-k expert selection; the gating weights that scale expert outputs
  are the bias-free sigmoid scores. ``route_bias`` is not gradient-trained
  (requires_grad=False); call ``update_bias(expert_load)`` periodically to
  nudge it by a fixed step toward balance.
- Device-limited routing: experts are partitioned across
  ``config.moe.device_group_size`` devices; under torch.distributed each rank
  only routes to and computes its local experts.
- Shared experts are always active.

The transformer layer is responsible for normalizing the input exactly once
before calling this module (there is intentionally no input norm here).
"""
import torch
import torch.nn as nn


class DeviceLimitedMoE(nn.Module):
    """MoE with sigmoid routing, aux-free bias balancing, device-limited experts."""

    # Fixed expert-call block size; see the dispatch comment in forward().
    _EXPERT_BLOCK = 64

    def __init__(self, config, device_map=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.num_shared = config.moe.num_shared_experts
        self.top_k = config.moe.num_activated_experts
        self.expert_hidden = config.moe.expert_hidden_size
        self.bias_update_rate = config.moe.bias_update_rate

        # Device mapping: expert_id -> device_id (uses device_group_size).
        self.device_map = device_map or self._default_device_map()
        self.num_devices = max(self.device_map.values()) + 1

        # Router: sigmoid activation (NOT softmax).
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Aux-free load-balancing bias: used ONLY for top-k selection, never
        # in the gating weights, and never updated by gradient descent.
        self.route_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        # Accumulated selection counts since the last update_bias() call.
        self.register_buffer("expert_load", torch.zeros(self.num_experts))
        # Lazy cache for the distributed local-expert id list (O3): the
        # expert->device map is fixed at construction and a rank's partition
        # never changes within a process, so the list is built at most once
        # per (rank, device) instead of on every forward.
        self._local_experts_cache = None

        # Routed experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.expert_hidden),
                nn.GELU(),
                nn.Linear(self.expert_hidden, self.hidden_size),
            )
            for _ in range(self.num_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.expert_hidden),
                nn.GELU(),
                nn.Linear(self.expert_hidden, self.hidden_size),
            )
            for _ in range(self.num_shared)
        ])

    def _default_device_map(self):
        """Evenly partition experts across ``device_group_size`` devices."""
        g = self.config.moe.device_group_size
        per_device = max(1, self.num_experts // g)
        return {eid: min(eid // per_device, g - 1) for eid in range(self.num_experts)}

    def _local_expert_ids(self, rank, device):
        """This rank's local expert ids, lazily built and cached.

        The device map and the rank's partition are fixed for the lifetime
        of the process, so the cache is keyed only on (rank, device) — the
        tensor is rebuilt at most once per device move, not per forward.
        """
        cache = self._local_experts_cache
        if cache is not None and cache[0] == rank and cache[1].device == device:
            return cache[1]
        local_experts = torch.tensor(
            [eid for eid, did in self.device_map.items()
             if did == rank % self.num_devices],
            dtype=torch.long, device=device,
        )
        self._local_experts_cache = (rank, local_experts)
        return local_experts

    def _route(self, flat):
        """Return (topk_indices [N, K], topk_gates [N, K]).

        Selection scores include route_bias; gating weights do not.
        """
        router_logits = self.router(flat)
        scores = torch.sigmoid(router_logits)                    # bias-free gates
        scores_for_select = scores + self.route_bias             # selection only

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Device-limited path: route only among this rank's local experts.
            world_size = torch.distributed.get_world_size()
            if world_size < self.num_devices:
                raise RuntimeError(
                    f"device-limited routing requires world_size >= "
                    f"device_group_size ({self.num_devices}), got world_size "
                    f"{world_size}: ranks would map to nonexistent expert "
                    "partitions"
                )
            rank = torch.distributed.get_rank()
            local_experts = self._local_expert_ids(rank, scores.device)
            local_sel = scores_for_select.index_select(1, local_experts)
            k_local = min(self.top_k, local_experts.numel())
            _, local_idx = torch.topk(local_sel, k_local, dim=-1)
            topk_indices = local_experts[local_idx]  # gather on-device
        else:
            topk_indices = torch.topk(scores_for_select, self.top_k, dim=-1).indices

        topk_gates = scores.gather(1, topk_indices)  # no bias in the weights
        return topk_indices, topk_gates

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, seq, hidden] (already normalized by the caller).
        Returns:
            output: [B, seq, hidden]
        """
        B, seq, H = hidden_states.shape
        flat = hidden_states.reshape(-1, H)
        N = flat.shape[0]

        topk_indices, topk_gates = self._route(flat)
        K = topk_indices.shape[1]

        # Accumulate routing-load statistics for aux-free bias updates.
        # Training-only (M-C2): eval / inference forwards (e.g. RL rollouts)
        # must not pollute the statistics that drive update_bias().
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                counts_all = torch.bincount(
                    topk_indices.reshape(-1), minlength=self.num_experts
                )
                self.expert_load += counts_all.to(self.expert_load.dtype)

        # Dispatch: sort tokens by expert id once, then process contiguous
        # segments — O(num_experts) iterations and a single .tolist() sync,
        # instead of top_k * num_experts masked gathers with per-expert syncs.
        #
        # Each expert call runs on FIXED-shape blocks of _EXPERT_BLOCK rows
        # (the tail block is zero-padded and its padding discarded). A
        # batched GEMM's per-row result is bit-reproducible only for a fixed
        # GEMM shape (kernel blocking depends on the row count), so variable
        # -length segments would make a token's expert output depend at the
        # ~1e-7 level on which OTHER tokens were routed to the same expert.
        # Fixed-shape blocks keep each token's output bit-identical
        # regardless of its batchmates — required for strict isolation
        # guarantees (e.g. packed-sequence cross-document perturbation
        # tests assert bit-identical outputs at the model level). Padding
        # waste is bounded by _EXPERT_BLOCK - 1 rows per expert.
        flat_expert = topk_indices.reshape(-1)                       # [N*K]
        flat_weight = topk_gates.reshape(-1)                         # [N*K]
        token_idx = torch.arange(N, device=flat.device).repeat_interleave(K)
        order = flat_expert.argsort()
        sorted_tokens = token_idx[order]
        sorted_weights = flat_weight[order]
        counts = torch.bincount(flat_expert, minlength=self.num_experts).tolist()

        output = torch.zeros_like(flat)
        start = 0
        block = self._EXPERT_BLOCK
        for eid, cnt in enumerate(counts):
            if cnt:
                idx = sorted_tokens[start : start + cnt]
                rows = flat[idx]
                outs = []
                for b0 in range(0, cnt, block):
                    chunk = rows[b0 : b0 + block]
                    n_real = chunk.shape[0]
                    if n_real < block:
                        chunk = torch.cat(
                            [chunk, chunk.new_zeros(block - n_real, H)], dim=0
                        )
                    outs.append(self.experts[eid](chunk)[:n_real])
                expert_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
                output.index_add_(
                    0, idx, expert_out * sorted_weights[start : start + cnt].unsqueeze(-1)
                )
            start += cnt

        # Shared experts (always active)
        for shared in self.shared_experts:
            output = output + shared(flat)

        return output.view(B, seq, H)

    @torch.no_grad()
    def update_bias(self, expert_load=None, step=None):
        """Aux-free bias update (DeepSeek-V3 style).

        Experts whose load is above the mean get their selection bias
        decreased by ``step``; below-mean experts get it increased. Pass an
        explicit ``expert_load`` tensor, or omit it to consume (and reset)
        the statistics accumulated during forward passes.

        Distributed (M-C3): when a process group is active the load is
        all-reduced (SUM) across ranks first, so the update is driven by
        GLOBAL routing statistics instead of this rank's local shard.
        ``route_bias`` needs no separate sync: the update rule is
        deterministic, so identical global loads on every rank produce
        identical biases (they start identical under DDP broadcast).
        """
        load = self.expert_load if expert_load is None else expert_load.clone()
        load = load.to(device=self.route_bias.device, dtype=self.route_bias.dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(load, op=torch.distributed.ReduceOp.SUM)
        u = self.bias_update_rate if step is None else step
        mean = load.mean()
        self.route_bias.add_(-u * torch.sign(load - mean))
        if expert_load is None:
            self.expert_load.zero_()
        return self.route_bias
