"""Ring Attention for 1M+ context length

Based on "Ring Attention with Blockwise Transformers for Near-Infinite Context"
(Liu et al., 2023) and FlashAttention-style blockwise computation.

Key idea:
  - Split sequence into chunks, each GPU processes one chunk
  - KV blocks are passed around a ring of GPUs
  - Each GPU computes attention with its local Q and all KV blocks
  - Final outputs are gathered back
"""
from .ring_attention import RingAttention, ring_attention_forward
from .context_parallel_group import ContextParallelGroup, get_cp_group, get_cp_world_size, get_cp_rank
from .sequence_parallel import split_sequence, gather_sequence

__all__ = ["RingAttention","ring_attention_forward","ContextParallelGroup","get_cp_group","get_cp_world_size","get_cp_rank","split_sequence","gather_sequence"]
