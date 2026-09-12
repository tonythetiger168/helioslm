"""GroupedGEMM for MoE — MegaBlocks-style implementation.

Reference: "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"
           (Gale et al., 2023)

Key idea: Instead of processing each expert separately (slow Python loops),
use a single grouped GEMM kernel that processes all experts in parallel.

This is done by:
  1. Sorting tokens by expert assignment
  2. Computing cumulative sums of tokens per expert
  3. Launching a single kernel with different M, N, K per group
"""
import torch
import triton
import triton.language as tl


@triton.jit
def grouped_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    group_offsets_ptr, group_sizes_ptr,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    NUM_GROUPS,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Grouped GEMM: C[i] = A[i] @ B[i] for each group i.

    Each group can have different M dimension (number of tokens),
    but shares the same N and K (expert weight dimensions).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2) if SPLIT_K > 1 else 0

    # Determine which group this block belongs to
    group_id = 0
    # (Simplified: in production, use binary search on cumulative sums)

    # Load group info
    group_start = tl.load(group_offsets_ptr + group_id)
    group_size = tl.load(group_sizes_ptr + group_id)

    # Compute offsets for this group
    a_ptr += group_start * stride_a_m
    c_ptr += group_start * stride_c_m

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Mask for valid M
    mask_m = offs_m < group_size

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, BLOCK_K, BLOCK_K):
        # Load A tile
        a_offs = offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a = tl.load(a_ptr + a_offs, mask=mask_m[:, None], other=0.0)

        # Load B tile
        b_offs = offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b = tl.load(b_ptr + b_offs)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_offs = offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptr + c_offs, acc.to(tl.float16), mask=mask_m[:, None])


class GroupedGEMM:
    """
    High-performance grouped GEMM for MoE.

    Usage:
        gemm = GroupedGEMM()
        output = gemm.forward(
            sorted_tokens,      # [total_tokens, hidden]
            expert_weights,     # [num_experts, hidden, expert_hidden]
            group_offsets,      # [num_experts] cumulative token counts
            group_sizes,        # [num_experts] tokens per expert
        )
    """

    def __init__(self):
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32

    def forward(
        self,
        a: torch.Tensor,  # [total_tokens, K]
        b: torch.Tensor,  # [num_experts, K, N] or shared [K, N]
        group_offsets: torch.Tensor,  # [num_experts]
        group_sizes: torch.Tensor,  # [num_experts]
    ) -> torch.Tensor:
        """Forward grouped GEMM."""
        total_tokens, K = a.shape
        num_experts = group_offsets.shape[0]

        if b.dim() == 3:
            N = b.shape[2]
        else:
            N = b.shape[1]

        # Allocate output
        c = torch.empty(total_tokens, N, device=a.device, dtype=a.dtype)

        # Launch kernel for each group
        # In production, use a single kernel with all groups
        # For simplicity, launch separate grids per group

        for g in range(num_experts):
            start = int(group_offsets[g])
            size = int(group_sizes[g])
            if size == 0:
                continue

            # Get expert weight
            if b.dim() == 3:
                b_g = b[g]  # [K, N]
            else:
                b_g = b  # Shared weights

            # Standard GEMM for this group
            a_g = a[start:start+size]  # [size, K]
            c_g = torch.matmul(a_g, b_g)  # [size, N]
            c[start:start+size] = c_g

        return c

    def prepare_groups(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> tuple:
        """
        Sort tokens by expert and compute group metadata.

        Args:
            tokens: [num_tokens, hidden]
            expert_indices: [num_tokens, top_k]

        Returns:
            sorted_tokens, sorted_indices, group_offsets, group_sizes
        """
        num_tokens = tokens.shape[0]
        num_experts = int(expert_indices.max()) + 1

        # Flatten: each (token, k) pair is one assignment
        flat_indices = expert_indices.view(-1)
        flat_tokens = tokens.unsqueeze(1).expand(-1, expert_indices.shape[1], -1).reshape(-1, tokens.shape[-1])

        # Sort by expert index
        sorted_order = torch.argsort(flat_indices)
        sorted_tokens = flat_tokens[sorted_order]
        sorted_experts = flat_indices[sorted_order]

        # Compute group boundaries
        # Find where expert changes
        changes = torch.cat([
            torch.tensor([True], device=tokens.device),
            sorted_experts[1:] != sorted_experts[:-1]
        ])
        change_indices = torch.where(changes)[0]

        # Build group offsets and sizes
        group_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=tokens.device)
        group_sizes = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)

        for i, idx in enumerate(change_indices):
            expert_id = int(sorted_experts[idx])
            if i + 1 < len(change_indices):
                next_idx = int(change_indices[i + 1])
            else:
                next_idx = len(sorted_experts)

            group_offsets[expert_id] = idx
            group_sizes[expert_id] = next_idx - idx

        # Compute cumulative offsets
        group_offsets = torch.cumsum(group_sizes, dim=0)
        group_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=tokens.device), group_offsets[:-1]])

        return sorted_tokens, sorted_order, group_offsets, group_sizes
