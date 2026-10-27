"""Grouped expert matmul with a fallback for torch without
``F.grouped_mm`` (v5.161).

``safe_grouped_matmul(inputs, weights, expert_indices)`` returns
``inputs[i] @ weights[expert_indices[i]]`` for every row, in the
original row order. With ``F.grouped_mm`` (torch >= 2.10) the rows are
sorted by expert, run as one grouped GEMM and scattered back; without it
each expert's rows run as one matmul and are written back to their own
positions (a plain concatenation of per-expert outputs would reorder the
tokens). ``DeviceLimitedMoE`` has its own fused dispatch (v5.149 /
v5.156); this helper is for callers that hold an [E, D, F] weight stack.
"""
import torch
import torch.nn.functional as F

from .sigmoid_moe import HAS_GROUPED_MM


def safe_grouped_matmul(inputs, weights, expert_indices, use_grouped_mm=None):
    """``inputs`` [N, D], ``weights`` [E, D, F], ``expert_indices`` [N]
    (ints in [0, E)) -> [N, F], row i = ``inputs[i] @ weights[e_i]``.

    ``use_grouped_mm`` None = when ``F.grouped_mm`` exists. Gradients flow
    to ``inputs`` and ``weights`` on both paths.
    """
    N, D = inputs.shape
    E, Dw, Fo = weights.shape
    if Dw != D or expert_indices.shape != (N,):
        raise ValueError(f"shapes: inputs {tuple(inputs.shape)}, weights "
                         f"{tuple(weights.shape)}, expert_indices "
                         f"{tuple(expert_indices.shape)}")
    if use_grouped_mm is None:
        use_grouped_mm = HAS_GROUPED_MM
    idx = expert_indices.long()
    if use_grouped_mm:
        order = idx.argsort(stable=True)
        offs = torch.bincount(idx, minlength=E).cumsum(0).to(torch.int32)
        y = F.grouped_mm(inputs[order], weights, offs=offs)
        return torch.empty_like(y).index_copy(0, order, y)
    out = inputs.new_zeros(N, Fo, dtype=torch.result_type(inputs, weights))
    for e in idx.unique().tolist():
        rows = (idx == e).nonzero(as_tuple=True)[0]
        out = out.index_copy(0, rows, inputs[rows] @ weights[e])
    return out
