"""Ring Attention core implementation.

Implements blockwise attention computation across a ring of GPUs.
Each GPU holds a chunk of the query (Q) and iteratively receives
chunks of key (K) and value (V) from other GPUs in the ring.

Algorithm (per GPU):
  1. Split local Q, K, V into blocks
  2. Initialize local output O and running statistics (m, l)
  3. For each step in ring:
     a. Receive K_j, V_j from neighbor
     b. Compute attention scores: S = Q_i @ K_j^T
     c. Update running max and softmax denominator
     d. Accumulate output: O += exp(S - m) @ V_j / l
     e. Send local K, V to next neighbor
  4. Return final O

This avoids materializing the full N x N attention matrix.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .context_parallel_group import get_cp_group, get_cp_world_size, get_cp_rank


def ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Ring Attention forward pass.

    Args:
        q: [batch, local_seq, num_heads, head_dim] — local query chunk
        k: [batch, local_seq, num_heads, head_dim] — local key chunk
        v: [batch, local_seq, num_heads, head_dim] — local value chunk
        causal: Whether to apply causal masking
        softmax_scale: 1/sqrt(head_dim), computed if None
        block_size: Block size for chunking within a GPU

    Returns:
        output: [batch, local_seq, num_heads, head_dim] — local output chunk
    """
    cp_group = get_cp_group()
    cp_size = get_cp_world_size()
    cp_rank = get_cp_rank()

    batch_size, local_seq, num_heads, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Initialize output and running statistics
    output = torch.zeros_like(q)
    running_max = torch.full((batch_size, local_seq, num_heads, 1), float('-inf'), device=q.device)
    running_sum = torch.zeros((batch_size, local_seq, num_heads, 1), device=q.device)

    # Determine which KV blocks this GPU needs to process
    # For causal attention, GPU i only needs KV from GPUs 0..i
    kv_source_ranks = list(range(cp_size)) if not causal else list(range(cp_rank + 1))

    # Current KV to send (starts with local KV)
    current_k = k.clone()
    current_v = v.clone()

    for step in range(cp_size):
        # Determine which rank's KV we're processing this step
        # In a ring, step 0 = local, step 1 = from rank-1, etc.
        kv_rank = (cp_rank - step) % cp_size

        if kv_rank in kv_source_ranks:
            # Process current KV block
            _process_kv_block(
                q, current_k, current_v,
                output, running_max, running_sum,
                softmax_scale, causal, cp_rank, kv_rank, cp_size, local_seq
            )

        # Ring communication: send KV to next rank, receive from prev rank
        if step < cp_size - 1:
            send_to = (cp_rank + 1) % cp_size
            recv_from = (cp_rank - 1) % cp_size

            # Prepare send/recv buffers
            k_recv = torch.empty_like(current_k)
            v_recv = torch.empty_like(current_v)

            # Isend/Irecv for overlap
            send_k = dist.isend(current_k, dst=send_to, group=cp_group)
            send_v = dist.isend(current_v, dst=send_to, group=cp_group)
            recv_k = dist.irecv(k_recv, src=recv_from, group=cp_group)
            recv_v = dist.irecv(v_recv, src=recv_from, group=cp_group)

            # Wait for completion
            send_k.wait()
            send_v.wait()
            recv_k.wait()
            recv_v.wait()

            current_k = k_recv
            current_v = v_recv

    # Normalize output
    output = output / running_sum

    return output


def _process_kv_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    q_rank: int,
    kv_rank: int,
    cp_size: int,
    local_seq: int,
):
    """Process a single KV block against local Q."""
    batch_size, _, num_heads, head_dim = q.shape

    # Compute attention scores
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) * softmax_scale

    # Causal mask: if q is from a later rank than kv, all positions are valid
    # If q is from an earlier rank, mask out future positions
    if causal:
        if q_rank == kv_rank:
            # Same rank: standard causal mask within the block
            causal_mask = torch.triu(torch.ones(local_seq, local_seq, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        elif q_rank < kv_rank:
            # Q is earlier in sequence: all KV positions are "future", mask all
            scores = scores.masked_fill(torch.ones_like(scores).bool(), float('-inf'))
        # else: q_rank > kv_rank, all positions valid (no mask)

    # Blockwise online softmax
    # For each query position, track max and sum across all KV processed so far
    block_max = scores.max(dim=-1, keepdim=True).values
    block_exp = torch.exp(scores - block_max)
    block_sum = block_exp.sum(dim=-1, keepdim=True)

    # Update running statistics
    new_max = torch.max(running_max, block_max)

    # Rescale previous output and sum
    output_scale = torch.exp(running_max - new_max)
    block_scale = torch.exp(block_max - new_max)

    running_sum = running_sum * output_scale + block_sum * block_scale

    # Update output: O_new = (O_old * scale_old + block_output * scale_block) / sum_new
    block_output = torch.einsum('bhqk,bkhd->bqhd', block_exp, v)
    output = output * output_scale + block_output * block_scale

    running_max = new_max


class RingAttention(nn.Module):
    """Ring Attention module that can replace standard attention."""

    def __init__(self, config, block_size: int = 1024):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.num_kv_heads = config.attention.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        """Forward with context parallelism."""
        from .sequence_parallel import split_sequence

        B, seq, h = hidden_states.shape

        # Split sequence across CP group
        local_hidden = split_sequence(hidden_states, dim=1)
        local_seq = local_hidden.shape[1]

        # Project
        q = self.q_proj(local_hidden).view(B, local_seq, self.num_heads, self.head_dim)
        k = self.k_proj(local_hidden).view(B, local_seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(local_hidden).view(B, local_seq, self.num_kv_heads, self.head_dim)

        # GQA repeat
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # Ring attention
        output = ring_attention_forward(q, k, v, causal=True, block_size=self.block_size)

        # Output projection
        output = self.o_proj(output.reshape(B, local_seq, -1))

        return output
