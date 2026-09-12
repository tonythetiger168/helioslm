"""FlashAttention Backward Pass — Complete Triton Implementation

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
           (Dao et al., 2022)

The backward pass recomputes attention matrices S and P on-the-fly
(instead of storing them) to save memory. This is the key insight of FlashAttention.

Algorithm (per block):
  1. Load Q, K, V, dO, L, M tiles
  2. Recompute S = Q @ K^T, P = softmax(S)
  3. Compute dV = P^T @ dO
  4. Compute dP = dO @ V^T
  5. Compute dS = P * (dP - sum(dO * O, axis=-1))
  6. Compute dQ = dS @ K, dK = dS^T @ Q
"""
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_bwd_kernel_dk_dv(
    Q, K, V, dO, dQ, dK, dV,
    L, M,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_do_b, stride_do_h, stride_do_m, stride_do_k,
    stride_dq_b, stride_dq_h, stride_dq_m, stride_dq_k,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_k,
    stride_dv_b, stride_dv_h, stride_dv_n, stride_dv_k,
    BATCH, N_HEADS, SEQ_LEN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Backward kernel for computing dK and dV."""
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % N_HEADS
    off_b = off_hz // N_HEADS

    # Initialize offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute K, V tile pointers
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    # Load K, V tiles
    k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
    k = k.to(tl.float32)
    v = v.to(tl.float32)

    # Initialize accumulators for dK and dV
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Load L and M for this block
    l_ptrs = L + off_hz * SEQ_LEN + offs_n
    m_ptrs = M + off_hz * SEQ_LEN + offs_n
    l_curr = tl.load(l_ptrs, mask=offs_n < SEQ_LEN, other=1.0)
    m_curr = tl.load(m_ptrs, mask=offs_n < SEQ_LEN, other=0.0)

    # Iterate over Q blocks
    lo = 0
    hi = (start_n + 1) * BLOCK_N if IS_CAUSAL else SEQ_LEN
    hi = tl.cdiv(hi, BLOCK_M) * BLOCK_M

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        offs_m = start_m + tl.arange(0, BLOCK_M)

        # Load Q, dO tiles
        q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        do_ptrs = dO + (off_b * stride_do_b + off_h * stride_do_h) + (offs_m[:, None] * stride_do_m + offs_d[None, :] * stride_do_k)

        q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
        q = q.to(tl.float32)
        do = do.to(tl.float32)

        # Recompute S = Q @ K^T
        qk = tl.dot(q, tl.trans(k))

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Scale
        qk *= 1.0 / tl.sqrt(float(BLOCK_DMODEL))

        # Recompute P = softmax(S)
        p = tl.exp(qk - m_curr[None, :]) / l_curr[None, :]

        # Compute dV = P^T @ dO
        dv += tl.dot(tl.trans(p.to(tl.float32)), do)

        # Compute dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # Compute Di = rowsum(dO * O) — loaded from L (which stores logsumexp)
        # For simplicity, we recompute: Di = rowsum(dO * (P @ V))
        # Actually, Di = rowsum(dO * O) where O = P @ V / l
        # We can compute this as: Di = rowsum(P * (dO @ V^T)) = rowsum(P * dp)
        di = tl.sum(p * dp, axis=1)

        # Compute dS = P * (dP - Di)
        ds = p * (dp - di[:, None])

        # Compute dK = dS^T @ Q
        dk += tl.dot(tl.trans(ds), q)

    # Store dK, dV
    dk_ptrs = dK + (off_b * stride_dk_b + off_h * stride_dk_h) + (offs_n[:, None] * stride_dk_n + offs_d[None, :] * stride_dk_k)
    dv_ptrs = dV + (off_b * stride_dv_b + off_h * stride_dv_h) + (offs_n[:, None] * stride_dv_n + offs_d[None, :] * stride_dv_k)

    tl.store(dk_ptrs, dk.to(tl.float16), mask=offs_n[:, None] < SEQ_LEN)
    tl.store(dv_ptrs, dv.to(tl.float16), mask=offs_n[:, None] < SEQ_LEN)


@triton.jit
def flash_attn_bwd_kernel_dq(
    Q, K, V, dO, dQ,
    L, M,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_do_b, stride_do_h, stride_do_m, stride_do_k,
    stride_dq_b, stride_dq_h, stride_dq_m, stride_dq_k,
    BATCH, N_HEADS, SEQ_LEN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Backward kernel for computing dQ."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % N_HEADS
    off_b = off_hz // N_HEADS

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q, dO tiles
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    do_ptrs = dO + (off_b * stride_do_b + off_h * stride_do_h) + (offs_m[:, None] * stride_do_m + offs_d[None, :] * stride_do_k)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    q = q.to(tl.float32)
    do = do.to(tl.float32)

    # Initialize accumulator for dQ
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load L and M for this block
    l_ptrs = L + off_hz * SEQ_LEN + offs_m
    m_ptrs = M + off_hz * SEQ_LEN + offs_m
    l_curr = tl.load(l_ptrs, mask=offs_m < SEQ_LEN, other=1.0)
    m_curr = tl.load(m_ptrs, mask=offs_m < SEQ_LEN, other=0.0)

    # Iterate over K,V blocks
    lo = start_m * BLOCK_M if IS_CAUSAL else 0
    hi = SEQ_LEN

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K, V tiles
        k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
        k = k.to(tl.float32)
        v = v.to(tl.float32)

        # Recompute S = Q @ K^T
        qk = tl.dot(q, tl.trans(k))

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Scale
        qk *= 1.0 / tl.sqrt(float(BLOCK_DMODEL))

        # Recompute P = softmax(S)
        p = tl.exp(qk - m_curr[:, None]) / l_curr[:, None]

        # Compute dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # Compute Di = rowsum(P * dP)
        di = tl.sum(p * dp, axis=1)

        # Compute dS = P * (dP - Di)
        ds = p * (dp - di[:, None])

        # Compute dQ = dS @ K
        dq += tl.dot(ds, k)

    # Store dQ
    dq_ptrs = dQ + (off_b * stride_dq_b + off_h * stride_dq_h) + (offs_m[:, None] * stride_dq_m + offs_d[None, :] * stride_dq_k)
    tl.store(dq_ptrs, dq.to(tl.float16), mask=offs_m[:, None] < SEQ_LEN)


def flash_attn_backward(q, k, v, o, do, lse, m_max, causal=True):
    """
    Complete FlashAttention backward pass.

    Args:
        q, k, v: Forward inputs [batch, n_heads, seq_len, head_dim]
        o: Forward output
        do: Gradient w.r.t. output
        lse, m_max: Logsumexp and max from forward pass
        causal: Whether forward used causal mask

    Returns:
        dq, dk, dv: Gradients w.r.t. inputs
    """
    batch, n_heads, seq_len, head_dim = q.shape

    # Allocate gradient tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Kernel config
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim

    # Launch dK, dV kernel
    grid_dk_dv = (triton.cdiv(seq_len, BLOCK_N), batch * n_heads)
    flash_attn_bwd_kernel_dk_dv[grid_dk_dv](
        q, k, v, do, dq, dk, dv,
        lse, m_max,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        batch, n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=causal,
        num_stages=2, num_warps=4,
    )

    # Launch dQ kernel
    grid_dq = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
    flash_attn_bwd_kernel_dq[grid_dq](
        q, k, v, do, dq,
        lse, m_max,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        batch, n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=causal,
        num_stages=2, num_warps=4,
    )

    return dq, dk, dv
