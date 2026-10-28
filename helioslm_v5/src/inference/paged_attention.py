"""PagedAttention - vLLM Style KV-Cache Management

Key innovations:
- Block-based KV-Cache allocation (fixed-size pages)
- Non-contiguous storage via block tables
- Reference-counted copy-on-write (CoW) for shared prefixes
- Efficient batching with different sequence lengths

Write convention (IMPORTANT):
- New K/V for a sequence are *appended* at logical index == current
  ``context_len`` (BlockTable.num_tokens). ``PagedAttention.forward``
  calls ``BlockManager.append_tokens`` itself before writing, so block
  allocation on the write path (including block-boundary crossings) is
  automatic. Callers must NOT pre-append.
- Cache dtype is chosen at construction (``dtype`` argument); ``None``
  means "follow the dtype seen at first allocation/write". Reads cast
  cached K/V to the query dtype before matmul when they differ.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class BlockTable:
    """Maps logical token positions to physical block indices."""
    logical_blocks: List[int]  # logical block id -> physical block id
    num_tokens: int  # total tokens in sequence


class BlockManager:
    """Manages physical blocks for PagedAttention.

    Blocks are reference counted: ``allocate``/append take a fresh block with
    refcount 1, ``fork`` shares blocks (refcount += 1), ``free`` decrements
    and only returns a block to the free pool when its refcount hits 0.
    Writes to a shared block (refcount > 1) trigger a true copy-on-write via
    ``ensure_writable``.
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 10000,
                 device: str = "cpu", dtype: Optional[torch.dtype] = None):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device
        # None => dtype follows the first allocation/write (C11)
        self.dtype = dtype

        # Free block pool (deque: O(1) popleft)
        self.free_blocks = deque(range(num_blocks))

        # Block tables for each sequence
        self.block_tables: Dict[int, BlockTable] = {}

        # Per-physical-block reference counts (M21)
        self.refcounts: Dict[int, int] = {}

        # Physical blocks: [num_blocks, block_size, num_heads, head_dim]
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        # Cache shape is fixed by the first allocation (m5)
        self._cache_shape: Optional[tuple] = None  # (num_heads, head_dim)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _init_cache(self, num_heads: int, head_dim: int, dtype: Optional[torch.dtype]):
        if self._cache_shape is None:
            self._cache_shape = (num_heads, head_dim)
            cache_dtype = dtype or self.dtype
            if cache_dtype is not None:
                self._create_cache(cache_dtype)
            # else: dtype stays undecided; cache tensors are created lazily
            # at the first write, following the incoming q/k/v dtype (C11).
        elif self._cache_shape != (num_heads, head_dim):
            raise ValueError(
                f"Cache shape mismatch: cache was created for "
                f"(num_heads, head_dim)={self._cache_shape}, got "
                f"({num_heads}, {head_dim})"
            )

    def _create_cache(self, dtype: torch.dtype):
        num_heads, head_dim = self._cache_shape
        self.dtype = dtype
        self.k_cache = torch.zeros(
            self.num_blocks, self.block_size, num_heads, head_dim,
            device=self.device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            self.num_blocks, self.block_size, num_heads, head_dim,
            device=self.device, dtype=dtype,
        )

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of memory: no free blocks available")
        block = self.free_blocks.popleft()
        self.refcounts[block] = 1
        return block

    def _release_block(self, block: int):
        self.refcounts[block] -= 1
        if self.refcounts[block] == 0:
            del self.refcounts[block]
            self.free_blocks.append(block)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def allocate(self, seq_id: int, num_tokens: int, num_heads: int, head_dim: int,
                 dtype: Optional[torch.dtype] = None):
        """Allocate blocks for a new sequence (initial refcount 1 per block)."""
        if seq_id in self.block_tables:
            raise ValueError(f"seq_id {seq_id} already allocated")
        self._init_cache(num_heads, head_dim, dtype)

        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of memory: no free blocks available")

        blocks = [self._alloc_block() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = BlockTable(logical_blocks=blocks, num_tokens=num_tokens)

    def append_tokens(self, seq_id: int, num_new_tokens: int):
        """Grow a sequence by ``num_new_tokens``, allocating new physical
        blocks when the write crosses a block boundary (M19/M20)."""
        table = self.block_tables[seq_id]
        new_total = table.num_tokens + num_new_tokens
        new_blocks_needed = (new_total + self.block_size - 1) // self.block_size

        while len(table.logical_blocks) < new_blocks_needed:
            table.logical_blocks.append(self._alloc_block())

        table.num_tokens = new_total

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get physical block indices for a sequence."""
        return self.block_tables[seq_id].logical_blocks

    def get_context_length(self, seq_id: int) -> int:
        """Get number of tokens in sequence."""
        return self.block_tables[seq_id].num_tokens

    def free(self, seq_id: int):
        """Free blocks for a sequence (refcount--; block returns to the pool
        only when no sequence references it anymore)."""
        if seq_id in self.block_tables:
            for block in self.block_tables[seq_id].logical_blocks:
                self._release_block(block)
            del self.block_tables[seq_id]

    def fork(self, parent_seq_id: int, child_seq_id: int):
        """Copy-on-write fork for beam search / parallel sampling.

        Parent and child share physical blocks (refcount += 1). Any later
        write into a shared block copies it first (see ``ensure_writable``),
        so partial-block appends never mutate the sibling.
        """
        if child_seq_id in self.block_tables:
            raise ValueError(f"seq_id {child_seq_id} already allocated")
        parent_table = self.block_tables[parent_seq_id]
        for block in parent_table.logical_blocks:
            self.refcounts[block] += 1
        self.block_tables[child_seq_id] = BlockTable(
            logical_blocks=parent_table.logical_blocks.copy(),
            num_tokens=parent_table.num_tokens,
        )

    def ensure_writable(self, seq_id: int, logical_block_idx: int) -> int:
        """Return a physical block that is safe to write for this sequence,
        performing a true copy-on-write if the block is shared (M21)."""
        table = self.block_tables[seq_id]
        physical = table.logical_blocks[logical_block_idx]
        if self.refcounts[physical] == 1:
            return physical
        # Shared block: copy contents into a fresh private block.
        new_block = self._alloc_block()
        if self.k_cache is not None:
            self.k_cache[new_block] = self.k_cache[physical]
            self.v_cache[new_block] = self.v_cache[physical]
        self._release_block(physical)
        table.logical_blocks[logical_block_idx] = new_block
        return new_block

    def write_kv(self, seq_id: int, position: int, k: torch.Tensor, v: torch.Tensor):
        """Write a single token's K/V at logical ``position`` (CoW-safe,
        casts to cache dtype). Callers must have grown the table via
        ``append_tokens`` first."""
        table = self.block_tables[seq_id]
        if position >= table.num_tokens:
            raise IndexError(
                f"write position {position} beyond context length {table.num_tokens}"
            )
        if self.k_cache is None:
            # Lazy cache creation: dtype follows the first written k/v (C11)
            self._create_cache(k.dtype)
        logical_block = position // self.block_size
        slot = position % self.block_size
        physical = self.ensure_writable(seq_id, logical_block)
        self.k_cache[physical, slot] = k.to(self.k_cache.dtype)
        self.v_cache[physical, slot] = v.to(self.v_cache.dtype)

    def gather_kv(self, seq_id: int):
        """Vectorized gather of the full K/V history for a sequence.

        Returns (k, v) each [context_len, num_heads, head_dim]. Uses tensor
        block-table indexing instead of per-token Python indexing (m4)."""
        table = self.block_tables[seq_id]
        ctx = table.num_tokens
        if self.k_cache is None:
            raise RuntimeError("KV cache not initialized: no values written yet")
        device = self.k_cache.device
        positions = torch.arange(ctx, device=device)
        block_ids = torch.tensor(
            table.logical_blocks, device=device, dtype=torch.long
        )[positions // self.block_size]
        slots = positions % self.block_size
        return self.k_cache[block_ids, slots], self.v_cache[block_ids, slots]

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding. x: [B, S, num_heads, head_dim];
    cos/sin: [S, head_dim], [B, S, head_dim], or already-4-D broadcastable
    to x (e.g. [1, S, 1, head_dim]). Anything else raises ValueError
    (M-I2: a 2-D [S, D] table must map to [1, S, 1, D] — broadcasting it
    as-is would align S with the BATCH axis, crashing for B != S and
    silently misaligning positions when B == S)."""
    def _prep(t: torch.Tensor, name: str) -> torch.Tensor:
        if t.dim() == 2:        # [S, D]        -> [1, S, 1, D]
            return t.unsqueeze(0).unsqueeze(2)
        if t.dim() == 3:        # [B, S, D]     -> [B, S, 1, D]
            return t.unsqueeze(2)
        if t.dim() == 4:        # already broadcastable to x
            return t
        raise ValueError(
            f"_apply_rope: {name} must be [S, D], [B, S, D] or 4-D "
            f"broadcastable to x; got shape {tuple(t.shape)}"
        )

    cos = _prep(cos, "cos")
    sin = _prep(sin, "sin")
    return (x * cos) + (_rotate_half(x) * sin)


class PagedAttention(nn.Module):
    """Attention with paged KV-cache.

    Write convention: new K/V are appended at logical index == context_len;
    this module grows the block table itself (append_tokens) before writing,
    so callers simply forward the newest hidden states. Optional RoPE can be
    applied to q/k via the ``cos``/``sin`` arguments; when None, RoPE is
    skipped (assumed unused or already applied upstream).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.attention.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, block_manager: BlockManager,
                seq_ids: List[int], cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, hidden] — the S newest tokens per sequence
                (S=1 for single-token decode).
            block_manager: BlockManager instance.
            seq_ids: list of sequence IDs, one per batch row.
            cos, sin: optional RoPE tables for the new tokens' absolute
                positions; if None, no RoPE is applied (see class docstring).

        Returns:
            [B, S, hidden] attention output for the new tokens.
        """
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)

        if cos is not None and sin is not None:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        # else: RoPE skipped by design (documented in class docstring)

        # --- write path: grow block table, then append at context_len -------
        start_positions = []
        for i, seq_id in enumerate(seq_ids):
            start = block_manager.get_context_length(seq_id)
            # M19/M20: allocation-on-write; safe across block boundaries.
            block_manager.append_tokens(seq_id, S)
            start_positions.append(start)
            for s in range(S):
                block_manager.write_kv(seq_id, start + s, k[i, s], v[i, s])

        # --- read path: vectorized gather + masked attention ----------------
        outputs = []
        for i, seq_id in enumerate(seq_ids):
            start = start_positions[i]
            cached_k, cached_v = block_manager.gather_kv(seq_id)  # [ctx, H, D]
            # C11: match cache to query dtype before matmul
            if cached_k.dtype != q.dtype:
                cached_k = cached_k.to(q.dtype)
                cached_v = cached_v.to(q.dtype)

            qi = q[i].transpose(0, 1)               # [H, S, D]
            ki = cached_k.transpose(0, 1)           # [H, ctx, D]
            vi = cached_v.transpose(0, 1)           # [H, ctx, D]

            scores = torch.matmul(qi, ki.transpose(-2, -1)) / (self.head_dim ** 0.5)
            # Causal mask within the chunk: query s (absolute pos start+s) may
            # attend to keys 0..start+s only.
            ctx = ki.shape[1]
            key_pos = torch.arange(ctx, device=scores.device).unsqueeze(0)      # [1, ctx]
            query_pos = (start + torch.arange(S, device=scores.device)).unsqueeze(1)  # [S, 1]
            causal_mask = key_pos > query_pos                                     # [S, ctx]
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, vi)            # [H, S, D]
            out = out.transpose(0, 1).reshape(S, self.num_heads * self.head_dim)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)        # [B, S, H*D]
        return self.o_proj(output)
