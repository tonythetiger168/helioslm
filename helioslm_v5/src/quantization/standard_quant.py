"""Standard Quantization - AWQ / GPTQ / FP8

Deterministic 4-bit weight quantization with GPTQ/AWQ-compatible storage
layouts, plus an FP8 (float8_e4m3fn) path when supported by the installed
torch.

Honesty notes:
  - GPTQLinear implements the real GPTQ algorithm (Frantar et al. 2022):
    with calibration activations it builds the Hessian H = 2/N * X^T X
    (percdamp-damped), inverts it via Cholesky, and quantizes column by
    column (OBS order) while propagating each column's quantization error
    into the not-yet-quantized columns. Without calibration data it falls
    back to plain per-group RTN (documented on ``from_linear``).
    GPTQ minimizes the *activation-weighted* (output) reconstruction
    error; its raw weight-space error is typically comparable to or
    slightly higher than RTN — that trade-off is inherent to the
    algorithm, not a bug.
  - AWQLinear performs plain per-group RTN unless calibration
    activations are passed to ``from_linear``, in which case a
    conservative activation-aware scaling is applied: the per-channel
    scale exponent alpha is grid-searched over {0, 0.25, 0.5} (alpha=0
    is exactly RTN), scales are clamped to [1, 2] (salient channels are
    only scaled UP, moderately), and each group additionally runs a
    clip grid-search over [0.5, 1.0] x (min, max) minimizing the
    activation-weighted output error. Because the grids include the
    RTN configuration, the calibrated result is never significantly
    worse than RTN and is typically better under peaky/outlier
    activations (v5.3 fix; the v5.2 full-magnitude alpha=1 scaling
    degraded badly under peaky activations).
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# AWQ calibration-path search grids (see AWQLinear.from_linear).
# alpha=0.0 disables scaling (pure RTN), so the search can never land
# worse than RTN on the calibration objective; the clip grid includes
# the full [min, max] range (clip=1.0) for the same reason.
_AWQ_ALPHA_GRID = (0.0, 0.25, 0.5)
_AWQ_CLIP_GRID = tuple(1.0 - 0.05 * i for i in range(11))  # 1.00 .. 0.50
_AWQ_SCALE_CLAMP = (1.0, 2.0)


def _quantize_groups(W, group_size, bits, clip_grid=(1.0,), col_weight=None):
    """Per-group asymmetric min/max quantization of W [out, in].

    Args:
        W: float weight matrix [out_features, in_features].
        group_size: quantization group size along the input dim.
        bits: quantization bits.
        clip_grid: candidate fractions of the [min, max] group range to
            try; must include 1.0 (plain min/max RTN). The clip
            minimizing the (optionally weighted) quantization error is
            kept per group.
        col_weight: optional [in_features] per-column error weights
            (AWQ calibration path: activation power / scale**2, i.e. the
            activation-weighted output-error objective).

    Returns (q uint8 [out, in], scales [out, G], zeros [out, G],
    total weighted error). All tensors live on W.device.
    """
    qmax = 2 ** bits - 1
    out_f, in_f = W.shape
    num_groups = (in_f + group_size - 1) // group_size
    dev = W.device
    q = torch.zeros(out_f, in_f, dtype=torch.uint8, device=dev)
    scales = torch.zeros(out_f, num_groups, device=dev)
    zeros = torch.zeros(out_f, num_groups, device=dev)
    total_err = 0.0
    for gi in range(num_groups):
        g0, g1 = gi * group_size, min((gi + 1) * group_size, in_f)
        wg = W[:, g0:g1]
        wmin = wg.min(dim=1).values
        wmax = wg.max(dim=1).values
        cw = (col_weight[g0:g1].clamp(min=1e-12).unsqueeze(0)
              if col_weight is not None else None)
        best = None
        for c in clip_grid:
            lo, hi = wmin * c, wmax * c
            scale = ((hi - lo) / qmax).clamp(min=1e-8)
            zp = torch.round(-lo / scale).clamp(0, qmax)
            qg = (torch.round(wg / scale.unsqueeze(1)) + zp.unsqueeze(1)
                  ).clamp(0, qmax)
            err = 0.0
            if cw is not None or len(clip_grid) > 1:
                dq = (qg - zp.unsqueeze(1)) * scale.unsqueeze(1)
                d = (dq - wg) ** 2
                err = (d * cw).sum().item() if cw is not None else d.sum().item()
            if best is None or err < best[0]:
                best = (err, qg, scale, zp)
        err, qg, scale, zp = best
        q[:, g0:g1] = qg.to(torch.uint8)
        scales[:, gi] = scale
        zeros[:, gi] = zp
        total_err += err
    return q, scales, zeros, total_err


class AWQLinear(nn.Module):
    """4-bit weight quantization (AWQ-style storage).

    Weights are quantized per-group along the input dimension with
    asymmetric affine quantization (scale = (max - min) / (2**bits - 1),
    integer zero point), packed two 4-bit values per uint8.

    If calibration activations are provided to ``from_linear``, a
    conservative activation-aware scaling is applied before
    quantization (alpha grid-searched over {0, 0.25, 0.5}, scales
    clamped to [1, 2], per-group clip grid-search over [0.5, 1.0] of
    the group range, objective = activation-weighted output error).
    Without calibration data this is plain RTN.
    """

    def __init__(self, in_features, out_features, group_size=128, bits=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        # Fused group-wise dequant x matmul (v5.11): never materializes the
        # dense [out, in] fp32 weight. Set to False to use the reference path.
        self.fused = True

        num_groups = (in_features + group_size - 1) // group_size
        # Packed weights: ceil(in/2) bytes per output row (supports odd in_features)
        self.register_buffer("qweight", torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, num_groups))
        self.register_buffer("zeros", torch.zeros(out_features, num_groups))
        # Optional activation-aware per-input-channel scale
        self.register_buffer("act_scale", None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size=128, bits=4, activations=None):
        """Quantize an nn.Linear's real weights via per-group RTN packing.

        Args:
            linear: source layer.
            group_size: quantization group size along input dim.
            bits: quantization bits (only 4 supported).
            activations: optional calibration activations [..., in_features]
                enabling conservative activation-aware scaling: the
                per-input-channel scale exponent alpha is grid-searched
                over {0, 0.25, 0.5} (alpha=0 == plain RTN), scales are
                clamped to [1, 2], and every group is clip-searched over
                [0.5, 1.0] x (min, max). The objective is the
                activation-weighted output error. Since the grids include
                the plain-RTN configuration, the calibrated path is never
                significantly worse than RTN (the v5.2 alpha=1
                full-magnitude scaling was, by up to ~10x on peaky
                activations).
        """
        if bits != 4:
            raise ValueError(
                f"only 4-bit packing is implemented, got bits={bits}")
        mod = cls(linear.in_features, linear.out_features,
                  group_size=group_size, bits=bits, bias=linear.bias is not None)
        W = linear.weight.detach().float()  # [out, in]

        if activations is not None:
            # Aligns with GPTQLinear.from_linear: a wrong last dim must
            # fail loudly instead of silently reshape-reinterpreting the
            # calibration data (v5.4 LOW fix).
            if activations.shape[-1] != linear.in_features:
                raise ValueError(
                    f"activations last dim {activations.shape[-1]} "
                    f"!= linear.in_features {linear.in_features}")
            act = (activations.detach().float()
                   .reshape(-1, linear.in_features).to(W.device))
            if act.shape[0] == 0:
                raise ValueError("calibration activations must be non-empty")
            s_raw = act.abs().mean(dim=0).clamp(min=1e-5)
            s_raw = s_raw / s_raw.mean()
            act_pow = act.pow(2).mean(dim=0)
            # Grid-search the scaling exponent; alpha=0 is plain RTN, so
            # the search falls back to RTN whenever scaling does not help
            # the activation-weighted error objective.
            best = None
            for alpha in _AWQ_ALPHA_GRID:
                s = s_raw.pow(alpha).clamp(*_AWQ_SCALE_CLAMP)
                q_c, sc_c, zp_c, err = _quantize_groups(
                    W * s.unsqueeze(0), group_size, bits,
                    clip_grid=_AWQ_CLIP_GRID,
                    col_weight=act_pow / s.pow(2).clamp(min=1e-12))
                if best is None or err < best[0]:
                    best = (err, q_c, sc_c, zp_c, s)
            _, q, scales, zeros, s = best
            mod.act_scale = s
        else:
            mod.act_scale = None
            q, scales, zeros, _ = _quantize_groups(W, group_size, bits)

        # Pack two 4-bit values per uint8: byte i = q[2i] | q[2i+1] << 4
        if mod.in_features % 2 == 1:
            q = F.pad(q, (0, 1))
        packed = q[:, 0::2] | (q[:, 1::2] << 4)
        mod.qweight = packed.contiguous()
        mod.scales = scales
        mod.zeros = zeros
        if linear.bias is not None:
            mod.bias = nn.Parameter(linear.bias.detach().clone())
        return mod.to(linear.weight.device)

    def _unpack(self) -> torch.Tensor:
        """Unpack 4-bit weights from uint8 -> [out_features, in_features]."""
        low = (self.qweight & 0x0F)
        high = (self.qweight >> 4) & 0x0F
        # Interleave back: even input positions from low nibble, odd from high
        weight = torch.stack([low, high], dim=2).flatten(1)
        weight = weight[:, :self.in_features]
        return weight.to(device=self.scales.device, dtype=self.scales.dtype)

    def _dequantize(self) -> torch.Tensor:
        """Deterministically reconstruct the weight matrix."""
        w = self._unpack()  # quantized integers as float
        group_idx = torch.arange(self.in_features, device=w.device) // self.group_size
        w = (w - self.zeros[:, group_idx]) * self.scales[:, group_idx]
        return w

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility accessor: the reconstructed [out_features, in_features]
        weight matrix.

        This is the SLOW path — every access dequantizes the packed buffers.
        It exists so weight-absorbing code (e.g. MLA's kv_b_proj folding) and
        other consumers that need the dense weight matrix keep working after
        quantization. When an activation-aware scale was applied at quantize
        time, it is undone here so the result is the *effective* linear
        weight (forward applies x/act_scale before the matmul).
        """
        w = self._dequantize()
        if self.act_scale is not None:
            w = w / self.act_scale.to(device=w.device, dtype=w.dtype).unsqueeze(0)
        return w

    def _fused_forward(self, x, bias):
        """Group-wise dequant x matmul (v5.11).

        Dequantizes only one group slice at a time and accumulates
        ``x[:, g] @ w_g.T`` — the dense [out, in] fp32 weight is never
        materialized (the old path allocates it on EVERY call, plus the
        full unpacked-int copy). Arithmetic per group is identical to
        ``_dequantize``; only the matmul reduction order differs, so the
        result matches the reference path to fp32 rounding (~1e-6).
        """
        acc = None
        for g0 in range(0, self.in_features, self.group_size):
            g1 = min(g0 + self.group_size, self.in_features)
            # Nibble interleave: byte j holds cols 2j (low) and 2j+1 (high)
            b0, b1 = g0 // 2, (g1 + 1) // 2
            lo = self.qweight[:, b0:b1] & 0x0F
            hi = (self.qweight[:, b0:b1] >> 4) & 0x0F
            w = torch.stack([lo, hi], dim=2).flatten(1)
            w = w[:, (g0 - 2 * b0):(g0 - 2 * b0) + (g1 - g0)]
            g = g0 // self.group_size
            w = (w.to(self.scales.dtype) - self.zeros[:, [g]]) * self.scales[:, [g]]
            part = F.linear(x[..., g0:g1], w.to(dtype=x.dtype))
            acc = part if acc is None else acc + part
        if bias is not None:
            acc = acc + bias
        return acc

    def forward(self, x):
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if getattr(self, "fused", True):
            if self.act_scale is not None:
                x = x / self.act_scale.to(device=x.device, dtype=x.dtype)
            return self._fused_forward(x, bias)
        # Reference path: materialize the dense weight (kept for A/B checks)
        weight = self._dequantize()
        if self.act_scale is not None:
            x = x / self.act_scale.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight.to(dtype=x.dtype), bias)


class GPTQLinear(nn.Module):
    """GPTQ: Accurate Post-Training Quantization (Frantar et al. 2022).

    When calibration activations are passed to ``from_linear``, this runs
    the real GPTQ algorithm: the Hessian H = 2/N * X^T X of the layer
    output reconstruction error is damped (``percdamp``), inverted via a
    Cholesky factorization, and the weight columns are quantized one by
    one; after each column is rounded, the remaining quantization error
    is propagated into the not-yet-quantized columns through the
    corresponding row of the inverse-Hessian Cholesky factor (Optimal
    Brain Surgeon update). Group scales/zero-points are recomputed from
    the *compensated* weights at every group boundary (standard GPTQ
    group handling). Without calibration data it falls back to plain
    per-group RTN packing.

    Note on error metrics: GPTQ minimizes the activation-weighted output
    error ||X(W - W_hat)||^2. Its raw weight-space error is usually on
    par with (or slightly above) RTN — GPTQ's win shows up in the layer
    output, not in the weight matrix itself.

    Layout (AutoGPTQ-compatible):
      - qweight: int32 [ceil(in/8), out], 8x 4-bit values packed per int32
        along the input dimension (nibble k = input row 8*row_block + k).
      - qzeros:  int32 [num_groups, ceil(out/8)], 8x 4-bit zero points.
      - scales:  float [num_groups, out].
      - g_idx:   int32 [in], group index per input row.
    """

    def __init__(self, in_features, out_features, bits=4, group_size=128, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        # Fused group-wise dequant x matmul (v5.11, see AWQLinear.fused).
        self.fused = True

        num_groups = (in_features + group_size - 1) // group_size
        pack_rows = (in_features + 7) // 8
        pack_cols = (out_features + 7) // 8
        self.register_buffer("qweight", torch.zeros(pack_rows, out_features, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(num_groups, pack_cols, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(num_groups, out_features))
        self.register_buffer("g_idx",
                             torch.arange(in_features, dtype=torch.int32) // group_size)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _rtn_quantize(W, group_size, bits):
        """Plain per-group min/max RTN. W: [in, out].

        Returns (q int32 [in, out], scales [G, out], zeros int32 [G, out]).
        """
        in_f, out_f = W.shape
        qmax = 2 ** bits - 1
        dev = W.device
        num_groups = (in_f + group_size - 1) // group_size
        g_idx = torch.arange(in_f, device=dev) // group_size
        q = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
        zeros = torch.zeros(num_groups, out_f, dtype=torch.int32, device=dev)
        scales = torch.zeros(num_groups, out_f, device=dev)
        for gi in range(num_groups):
            rows = (g_idx == gi).nonzero(as_tuple=True)[0]
            wg = W[rows]  # [g, out]
            wmin = wg.min(dim=0).values
            wmax = wg.max(dim=0).values
            scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
            zp = torch.round(-wmin / scale).clamp(0, qmax)
            qg = (torch.round(wg / scale.unsqueeze(0)) + zp.unsqueeze(0)).clamp(0, qmax)
            q[rows] = qg.to(torch.int32)
            zeros[gi] = zp.to(torch.int32)
            scales[gi] = scale
        return q, scales, zeros

    @staticmethod
    def _gptq_quantize(W, X, group_size, bits, percdamp, blocksize=128,
                       act_order=False):
        """True GPTQ quantization with Hessian error compensation.

        Args:
            W: [in, out] float weight (input-major).
            X: [N, in] calibration activations (the layer inputs the
                quantized weights will see).
            group_size: quantization group size along the input dim.
            bits: quantization bits (4).
            percdamp: damping as a fraction of mean(diag(H)).
            blocksize: OBS block width (AutoGPTQ uses 128): error is
                propagated densely inside a block and via one rank-`block`
                matmul to the remaining columns after each block.
            act_order (v5.8): quantize columns in DESCENDING diag(H) order
                (AutoGPTQ's act-order / "desc" heuristic). High-activation
                columns are quantized FIRST, while the still-dense later
                columns can absorb their error; with heterogeneous column
                scales this measurably lowers the output error. The packed
                layout is unchanged: columns are un-permuted afterwards and
                ``g_idx`` records each ORIGINAL column's group.

        Returns (q int32 [in, out], scales [G, out], zeros int32 [G, out],
        g_idx int64 [in]) with the same conventions as ``_rtn_quantize``;
        ``g_idx`` is the plain arange//group_size when act_order=False.
        """
        in_f, out_f = W.shape
        qmax = 2 ** bits - 1
        dev = W.device
        W = W.clone()

        # Hessian of the output reconstruction error (scale factor is
        # arbitrary; kept at the paper's 2/N).
        H = (2.0 / X.shape[0]) * (X.t() @ X)
        # Dead input columns (never activated): fix them so H stays PD.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[dead, :] = 0

        if act_order:
            # Permute columns so quantization walks diag(H) descending.
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[perm]
            H = H[perm][:, perm]

        # Damped Cholesky -> inverse -> upper Cholesky factor of H^{-1}
        # (the GPTQ update coefficients are rows of this factor).
        damp = percdamp * torch.diag(H).mean().clamp(min=1e-8)
        diag_idx = torch.arange(in_f, device=dev)
        Hinv = None
        for _ in range(6):  # retry with stronger damping if not PD
            try:
                Hd = H.clone()
                Hd[diag_idx, diag_idx] += damp
                L = torch.linalg.cholesky(Hd)
                Hinv = torch.cholesky_inverse(L)
                Hinv = torch.linalg.cholesky(Hinv, upper=True)
                break
            except Exception:
                damp = damp * 10.0
        if Hinv is None:
            raise RuntimeError(
                "GPTQ: Hessian is not positive definite even after damping; "
                "check the calibration data")

        num_groups = (in_f + group_size - 1) // group_size
        q = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
        zeros = torch.zeros(num_groups, out_f, dtype=torch.int32, device=dev)
        scales = torch.zeros(num_groups, out_f, device=dev)
        err_blk = torch.zeros(blocksize, out_f, device=dev)
        cur_scale = cur_zero = None
        for i1 in range(0, in_f, blocksize):
            i2 = min(i1 + blocksize, in_f)
            count = i2 - i1
            for i in range(count):
                col = i1 + i
                # Standard GPTQ group handling: at every group boundary,
                # recompute scale/zero from the *compensated* weights.
                if col % group_size == 0:
                    gi = col // group_size
                    wg = W[col:min(col + group_size, in_f)]
                    wmin = wg.min(dim=0).values
                    wmax = wg.max(dim=0).values
                    cur_scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
                    cur_zero = torch.round(-wmin / cur_scale).clamp(0, qmax)
                    scales[gi] = cur_scale
                    zeros[gi] = cur_zero.to(torch.int32)
                w = W[col]
                qc = (torch.round(w / cur_scale) + cur_zero).clamp(0, qmax)
                q[col] = qc.to(torch.int32)
                # OBS update: propagate the normalized quantization error
                # into the not-yet-quantized columns of this block.
                err = (w - (qc - cur_zero) * cur_scale) / Hinv[col, col]
                err_blk[i] = err
                if i + 1 < count:
                    W[col + 1:i2] -= (err.unsqueeze(0)
                                      * Hinv[col, col + 1:i2].unsqueeze(1))
            # Propagate this block's error to all remaining columns.
            if i2 < in_f:
                W[i2:] -= (err_blk[:count].t() @ Hinv[i1:i2, i2:]).t()

        g_idx = torch.arange(in_f, device=dev) // group_size
        if act_order:
            # Un-permute: q rows back to the original column order, and
            # map each original column to the group it was quantized in
            # (groups are contiguous in PERMUTED space).
            inv_perm = torch.argsort(perm)
            q = q[inv_perm]
            g_idx = g_idx[inv_perm]
        return q, scales, zeros, g_idx

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size=128, bits=4,
                    calibration_data=None, percdamp=0.01, act_order=False):
        """Quantize an nn.Linear into the GPTQ-compatible layout.

        Args:
            linear: source layer.
            group_size: quantization group size along the input dim.
            bits: quantization bits (only 4 supported).
            calibration_data: optional calibration activations
                [..., in_features]. When given, the true GPTQ algorithm
                (Hessian-based error compensation) is used; when None,
                this is a plain per-group RTN packing fallback.
            percdamp: Hessian damping fraction (only used with
                calibration data).
            act_order (v5.8): quantize columns in descending diag(H)
                order (only meaningful with calibration data; ignored on
                the RTN fallback path since no Hessian exists).
        """
        if bits != 4:
            raise ValueError(
                f"only 4-bit packing is implemented, got bits={bits}")
        in_f, out_f = linear.in_features, linear.out_features
        mod = cls(in_f, out_f, bits=bits, group_size=group_size,
                  bias=linear.bias is not None)
        W = linear.weight.detach().float().t().contiguous()  # [in, out]

        if calibration_data is None:
            q, scales, zeros = cls._rtn_quantize(W, group_size, bits)
            num_groups = (in_f + group_size - 1) // group_size
            g_idx = torch.arange(in_f, device=q.device) // group_size
        else:
            if calibration_data.shape[-1] != in_f:
                raise ValueError(
                    f"calibration_data last dim {calibration_data.shape[-1]} "
                    f"!= linear.in_features {in_f}")
            X = calibration_data.detach().float().reshape(-1, in_f).to(W.device)
            q, scales, zeros, g_idx = cls._gptq_quantize(
                W, X, group_size, bits, percdamp, act_order=act_order)
            num_groups = (in_f + group_size - 1) // group_size

        # Pack qweight: int32 per 8 input rows
        in_pad = ((in_f + 7) // 8) * 8
        if in_pad != in_f:
            q = F.pad(q, (0, 0, 0, in_pad - in_f))
        q = q.reshape(in_pad // 8, 8, out_f)
        shifts = (torch.arange(8, dtype=torch.int32, device=q.device) * 4
                  ).view(1, 8, 1)
        mod.qweight = (q << shifts).sum(dim=1, dtype=torch.int32).contiguous()

        # Pack qzeros: int32 per 8 output cols
        out_pad = ((out_f + 7) // 8) * 8
        z = zeros
        if out_pad != out_f:
            z = F.pad(z, (0, out_pad - out_f))
        z = z.reshape(num_groups, out_pad // 8, 8)
        shifts = (torch.arange(8, dtype=torch.int32, device=z.device) * 4
                  ).view(1, 1, 8)
        mod.qzeros = (z << shifts).sum(dim=2, dtype=torch.int32).contiguous()

        mod.scales = scales
        mod.g_idx = g_idx.to(torch.int32)
        if linear.bias is not None:
            mod.bias = nn.Parameter(linear.bias.detach().clone())
        return mod.to(linear.weight.device)

    def _dequantize(self) -> torch.Tensor:
        """Deterministically unpack and dequantize -> [out_features, in_features]."""
        dev = self.qweight.device
        shifts8 = (torch.arange(8, device=dev, dtype=torch.int32) * 4)
        # Unpack qweight -> q [in_pad, out]
        q = (self.qweight.unsqueeze(1) >> shifts8.view(1, 8, 1)) & 0xF  # [R, 8, out]
        q = q.reshape(-1, self.out_features)[:self.in_features]
        # Unpack qzeros -> z [num_groups, out]
        z = (self.qzeros.unsqueeze(-1) >> shifts8.view(1, 1, 8)) & 0xF  # [G, OB, 8]
        z = z.reshape(self.qzeros.shape[0], -1)[:, :self.out_features]
        # Dequantize per group
        scales = self.scales.to(q.device)
        w = (q.float() - z[self.g_idx.long()].float()) * scales[self.g_idx.long()]
        return w.t().contiguous()  # [out, in]

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility accessor: the reconstructed [out_features, in_features]
        weight matrix.

        This is the SLOW path — every access unpacks and dequantizes the
        packed buffers. It exists so weight-absorbing code (e.g. MLA's
        kv_b_proj folding) and other consumers that need the dense weight
        matrix keep working after quantization.
        """
        return self._dequantize()

    def _fused_forward(self, x, bias):
        """Run-wise dequant x matmul (v5.11).

        GPTQ's ``g_idx`` maps each input column to a group (act-order makes
        it monotonic; plain grouping makes it piecewise-constant). Slicing by
        maximal runs of constant ``g_idx`` gives the largest fused tiles; the
        dense [out, in] weight is never materialized. Per-run arithmetic is
        identical to ``_dequantize``.
        """
        dev = self.qweight.device
        shifts8 = torch.arange(8, device=dev, dtype=torch.int32) * 4
        g_idx = self.g_idx.long()
        # Maximal runs of constant group id
        boundaries = (g_idx[1:] != g_idx[:-1]).nonzero().flatten() + 1
        starts = [0] + boundaries.tolist()
        ends = boundaries.tolist() + [g_idx.numel()]
        acc = None
        for c0, c1 in zip(starts, ends):
            g = int(g_idx[c0])
            # int32 element r holds input cols 8r..8r+8 (4 bits each)
            r0, r1 = c0 // 8, (c1 - 1) // 8 + 1
            q = (self.qweight[r0:r1].unsqueeze(1) >> shifts8.view(1, 8, 1)) & 0x0F
            # input columns flatten into DIM 0 after reshape (dim 1 is
            # out_features) — the dense reference slices [:, :in] on dim 0
            q = q.reshape(-1, self.out_features)
            q = q[(c0 - 8 * r0):(c0 - 8 * r0) + (c1 - c0), :]
            z = (self.qzeros[g].unsqueeze(-1) >> shifts8.view(1, 8)) & 0x0F
            z = z.reshape(-1)[:self.out_features]
            w = (q.to(self.scales.dtype) - z) * self.scales[g]
            part = F.linear(x[..., c0:c1], w.t().to(dtype=x.dtype))
            acc = part if acc is None else acc + part
        if bias is not None:
            acc = acc + bias
        return acc

    def forward(self, x):
        # Deterministic: rebuild W from the packed buffers every call.
        # In production, use auto-gptq / exllama fused kernels instead.
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if getattr(self, "fused", True):
            return self._fused_forward(x, bias)
        weight = self._dequantize()
        return F.linear(x, weight.to(dtype=x.dtype), bias)


if _FP8_DTYPE is not None:

    class FP8Linear(nn.Module):
        """FP8 (float8_e4m3fn) weight-only quantization.

        Weights are stored as float8 with a per-tensor scale; forward
        dequantizes to the input dtype for the matmul (no FP8 matmul
        kernel is used).
        """

        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.register_buffer("weight_fp8", torch.zeros(out_features, in_features, dtype=_FP8_DTYPE))
            self.register_buffer("scale", torch.ones(()))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

        @classmethod
        def from_linear(cls, linear: nn.Linear):
            mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
            W = linear.weight.detach().float()
            amax = W.abs().max().clamp(min=1e-8)
            scale = amax / 448.0  # e4m3 max finite ~448
            mod.weight_fp8 = (W / scale).to(_FP8_DTYPE)
            mod.scale = scale
            if linear.bias is not None:
                mod.bias = nn.Parameter(linear.bias.detach().clone())
            return mod.to(linear.weight.device)

        @property
        def weight(self) -> torch.Tensor:
            """Compatibility accessor: the reconstructed
            [out_features, in_features] weight matrix.

            This is the SLOW path — every access dequantizes the FP8
            buffer. It exists so weight-absorbing code (e.g. MLA's
            kv_b_proj folding) and other consumers that need the dense
            weight matrix keep working after quantization.
            """
            return self.weight_fp8.to(self.scale.dtype) * self.scale

        def forward(self, x):
            weight = self.weight_fp8.to(x.dtype) * self.scale.to(x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
            return F.linear(x, weight, bias)

else:
    FP8Linear = None


class MXFP4Linear(nn.Module):
    """MXFP4 microscaling FP4 weight-only quantization (v5.6).

    Aligns with the Kimi-K3 / OCP MX recipe: weights are quantized to the
    FP4 E2M1 format in fixed blocks (default 32 elements, the MX block
    size) with one E8M0 (power-of-two) scale per block:

        scale = 2^ceil(log2(amax / 6.0))   # 6.0 = max finite E2M1 value
        q     = round_to_E2M1(w / scale)   # per element, sign + 3-bit code

    E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}; rounding is to the
    nearest magnitude with round-half-away-from-zero via searchsorted.
    Codes are packed two-per-uint8 along the input dim (padded with zero
    codes when in_features is odd). The E8M0 scale is stored in fp32 but
    is a power of two by construction.

    This is a numerical simulation of the MXFP4 format (dequantize to the
    input dtype for the matmul, like FP8Linear) — no MX hardware kernel is
    used. No calibration data: MX is a fixed-format recipe, not an
    activation-aware one. Biases are preserved.
    """

    E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def __init__(self, in_features, out_features, block_size=32, bias=True):
        super().__init__()
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        # Fused block-wise dequant x matmul (v5.11, see AWQLinear.fused).
        self.fused = True
        n_blocks = (in_features + block_size - 1) // block_size
        n_packed = (in_features + 1) // 2
        # qweight: two 4-bit codes per uint8; a pad code is 0 (magnitude 0,
        # positive sign) so odd in_features dequantize to exact zeros.
        self.register_buffer("qweight", torch.zeros(out_features, n_packed,
                                                    dtype=torch.uint8))
        # E8M0 scales are powers of two; stored fp32 for device flexibility.
        self.register_buffer("scales", torch.ones(out_features, n_blocks))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size=32):
        mod = cls(linear.in_features, linear.out_features,
                  block_size=block_size, bias=linear.bias is not None)
        W = linear.weight.detach().float()
        out_f, in_f = W.shape
        bs = block_size
        pad = (-in_f) % bs
        if pad:
            W = torch.cat([W, W.new_zeros(out_f, pad)], dim=1)
        blocks = W.view(out_f, (in_f + pad) // bs, bs)
        amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)
        # E8M0: smallest power of two >= amax / 6 (max finite E2M1).
        scales = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
        q = blocks / scales.unsqueeze(-1)
        mags = q.new_tensor(cls.E2M1_MAGNITUDES)          # [8]
        aq = q.abs()
        # Nearest magnitude: searchsorted right, compare both neighbors.
        idx = torch.searchsorted(mags, aq.contiguous())
        lo = (idx - 1).clamp(min=0)
        hi = idx.clamp(max=len(cls.E2M1_MAGNITUDES) - 1)
        pick_hi = (aq - mags[lo]) >= (mags[hi] - aq)
        mag_idx = torch.where(pick_hi, hi, lo)
        codes = mag_idx.to(torch.uint8)                     # magnitude code
        sign = (torch.where(q < 0, 1, 0).to(torch.uint8) << 3)
        codes = codes | sign
        codes = codes.view(out_f, (in_f + pad) // 2, 2)
        packed = codes[:, :, 0] | (codes[:, :, 1] << 4)
        mod.qweight = packed.contiguous()
        mod.scales = scales.contiguous()
        if linear.bias is not None:
            mod.bias = nn.Parameter(linear.bias.detach().clone())
        return mod.to(linear.weight.device)

    def _dequantize(self) -> torch.Tensor:
        """Rebuild [out_features, in_features] (slow path, deterministic)."""
        lo = self.qweight & 0xF
        hi = (self.qweight >> 4) & 0xF
        codes = torch.stack([lo, hi], dim=-1).view(self.out_features, -1)
        codes = codes[:, :self.in_features].long()
        # v5.11 fix: new_tensor on the LONG codes tensor inherited long
        # dtype, silently truncating the table to (0,0,1,1,2,3,4,6) — .5
        # and 1.5 magnitudes were lost. Build the fp32 table explicitly so
        # decode is the exact inverse of from_linear's float-table encode.
        mag = torch.tensor(self.E2M1_MAGNITUDES, dtype=self.scales.dtype,
                           device=codes.device)
        mags = mag[codes & 0x7]
        signs = torch.where((codes & 0x8) != 0, -1.0, 1.0).to(self.scales.dtype)
        w = mags * signs
        pad = (-self.in_features) % self.block_size
        if pad:
            w = torch.cat([w, w.new_zeros(self.out_features, pad)], dim=1)
        blocks = w.view(self.out_features, -1, self.block_size)
        return (blocks * self.scales.unsqueeze(-1)).view(
            self.out_features, -1)[:, :self.in_features]

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility accessor: reconstructed dense weight (slow path)."""
        return self._dequantize()

    def _fused_forward(self, x, bias):
        """Block-wise dequant x matmul (v5.11).

        Decodes only one E2M1 block slice at a time (magnitude lookup +
        sign + E8M0 power-of-two scale) and accumulates; the dense
        [out, in] weight is never materialized. Pad codes decode to exact
        zero magnitude, so a ragged final block needs no special casing.
        """
        mag = x.new_tensor(self.E2M1_MAGNITUDES).to(self.scales.dtype)
        acc = None
        for b0 in range(0, self.in_features, self.block_size):
            b1 = min(b0 + self.block_size, self.in_features)
            bb0, bb1 = b0 // 2, (b1 + 1) // 2
            lo = self.qweight[:, bb0:bb1] & 0x0F
            hi = (self.qweight[:, bb0:bb1] >> 4) & 0x0F
            codes = torch.stack([lo, hi], dim=-1).view(self.out_features, -1)
            codes = codes[:, (b0 - 2 * bb0):(b0 - 2 * bb0) + (b1 - b0)].long()
            mags = mag[codes & 0x7]
            signs = torch.where((codes & 0x8) != 0, -1.0, 1.0).to(self.scales.dtype)
            w = mags * signs * self.scales[:, [b0 // self.block_size]]
            part = F.linear(x[..., b0:b1], w.to(dtype=x.dtype))
            acc = part if acc is None else acc + part
        if bias is not None:
            acc = acc + bias
        return acc

    def forward(self, x):
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if getattr(self, "fused", True):
            return self._fused_forward(x, bias)
        weight = self._dequantize().to(x.dtype)
        return F.linear(x, weight, bias)


class QuantizationManager:
    """Manage model quantization by replacing nn.Linear layers with
    quantized equivalents (real weights are quantized via from_linear;
    biases are preserved)."""

    METHODS = {
        "awq": AWQLinear,
        "gptq": GPTQLinear,
        "fp8": FP8Linear,  # None if torch lacks float8_e4m3fn
        "mxfp4": MXFP4Linear,
    }

    def __init__(self, method="awq"):
        self.method = method

    def quantize_model(self, model, group_size=None, calibration_data=None,
                       act_order=False):
        """Quantize all Linear layers in model, in place.

        Args:
            model: module tree to quantize.
            group_size: quantization group size (AWQ/GPTQ) or MX block size
                (MXFP4). None selects the method default: 128 for AWQ/GPTQ,
                32 for MXFP4 (the OCP MX block size).
            calibration_data: optional calibration activations. Either a
                single tensor [..., in_features] (then every target
                Linear must share that in_features) or a dict mapping
                module name -> activation tensor for per-layer
                calibration (layers missing from the dict fall back to
                the no-calibration path). For method="gptq" it enables
                the true GPTQ Hessian error compensation; for
                method="awq" it enables the activation-aware per-channel
                scaling (AWQLinear's ``activations`` argument). It is
                ignored for fp8 and mxfp4.
            act_order (v5.8, gptq only): quantize columns in descending
                diag(H) order; ignored by the other methods.

        Raises:
            ValueError: unknown quantization method.
            NotImplementedError: method="fp8" on a torch build without
                float8_e4m3fn support.
        """
        if self.method not in self.METHODS:
            raise ValueError(
                f"Unknown quantization method {self.method!r}; "
                f"available: {sorted(self.METHODS)}"
            )
        qcls = self.METHODS[self.method]
        if qcls is None:
            raise NotImplementedError(
                "FP8 quantization requires a torch build with float8_e4m3fn support"
            )
        if group_size is None:
            group_size = 32 if self.method == "mxfp4" else 128

        # Build the parent map once; snapshot targets before mutating.
        modules = dict(model.named_modules())
        targets = [(name, m) for name, m in modules.items() if isinstance(m, nn.Linear)]

        for name, module in targets:
            if not name:
                # `model` itself is a bare nn.Linear; a module cannot be
                # replaced in place — skip it (quantize its parent instead).
                # v5.4: warn instead of skipping silently, since the caller
                # asked for quantization and gets none for this layer.
                warnings.warn(
                    "quantize_model: `model` itself is a bare nn.Linear and "
                    "cannot be replaced in place; it is left UNQUANTIZED. "
                    "Wrap it in a container module (or quantize its parent) "
                    "to include it.",
                    stacklevel=2,
                )
                continue
            if self.method == "fp8":
                qmodule = qcls.from_linear(module)
            elif self.method == "mxfp4":
                # group_size doubles as the MX block size (default 32).
                qmodule = qcls.from_linear(module, block_size=group_size)
            else:
                if isinstance(calibration_data, dict):
                    calib = calibration_data.get(name)
                else:
                    calib = calibration_data
                if self.method == "gptq":
                    qmodule = qcls.from_linear(module, group_size=group_size,
                                               calibration_data=calib,
                                               act_order=act_order)
                else:  # awq
                    qmodule = qcls.from_linear(module, group_size=group_size,
                                               activations=calib)

            parent_name, _, child_name = name.rpartition(".")
            parent = modules[parent_name] if parent_name else model
            setattr(parent, child_name, qmodule)

        # Re-bind MTP shared modules (v5.1 leftover): quantizing replaces
        # model.lm_head with a quantized module, which leaves
        # mtp_modules[i].lm_head pointing at the OLD nn.Linear and silently
        # breaks the weight tying. Point every MTP module's lm_head /
        # embed_tokens back at the main model's CURRENT modules, whatever
        # type they now are. Defensive getattr: model may not have MTP.
        mtp_modules = getattr(model, "mtp_modules", None)
        if mtp_modules is not None:
            lm_head = getattr(model, "lm_head", None)
            embed_tokens = getattr(model, "embed_tokens", None)
            for mtp in mtp_modules:
                if lm_head is not None and hasattr(mtp, "lm_head"):
                    mtp.lm_head = lm_head
                if embed_tokens is not None and hasattr(mtp, "embed_tokens"):
                    mtp.embed_tokens = embed_tokens
        return model
