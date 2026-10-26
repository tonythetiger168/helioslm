"""Sigmoid MoE with Device-Limited Routing - DeepSeek-V3 Style

Key properties:
- Sigmoid routing (independent per-expert scores, no softmax competition).
- Aux-free load balancing (DeepSeek-V3 style): ``route_bias`` is added ONLY
  for top-k expert selection; the gating weights that scale expert outputs
  are the bias-free sigmoid scores. ``route_bias`` is not gradient-trained
  (requires_grad=False); call ``update_bias(expert_load)`` periodically.
  Two strategies (``config.moe.balance_strategy``):
    "heuristic" (default, v5.4): nudge the bias by a fixed step toward
        balance (sign of deviation from mean load).
    "quantile" (v5.5): simplified quantile estimator — each expert keeps a
        sliding window of routing margins (biased selection score minus the
        token's (K+1)-th largest score, i.e. the first NON-selected score,
        so margin > 0 iff the expert is selected); update_bias sets
        ``bias_e -= quantile(margins_e, 1 - target_frac)`` with
        ``target_frac = top_k / num_experts``, so each expert's selection
        probability is driven toward the uniform-load target in one step.
        This is a SIMPLIFIED estimator: the boundary tau is treated as
        fixed across the update and margins are pooled per expert without
        modelling token distribution shift; it is not the full K3
        quantile controller.
    "smebu" (v5.72, Arcee Trinity): soft-clamped momentum update,
        ``v = (mean - load) / mean``,
        ``m <- beta m + (1 - beta) tanh(kappa v)``,
        ``bias += bias_update_rate * m``.
    "quantile_balancing" (v5.77, Kimi K3 Quantile Balancing, after the
        K3 report as implemented in torchtitan): every TRAINING token
        records, per expert, the margin ``cutoff - score`` (cutoff = the
        token's (K+1)-th largest biased score, score = the bias-free
        sigmoid) into a per-expert histogram over
        ``[min(bias) - 1, max(bias) + 1]``. ``update_bias`` solves the bias
        instead of stepping it: expert e is selected iff
        ``score + bias_e > cutoff``, i.e. iff ``margin < bias_e``, so
        ``bias_e = quantile(margins_e, K / E)`` would give it selection
        rate K / E over every token since the last update if the cutoffs
        stayed put (they move with the biases, so repeated solves are a
        fixed-point iteration; on a fixed batch the first one halves the
        largest imbalance, see T116); the result is then
        mean-centred (a common shift never changes a top-k ranking, so
        centring only stops drift). Histograms are SUMMED across ranks, so
        the quantile is the global one, exact up to the bin width.
        Compared with "quantile": same target (b - q_{1-f}(s + b - c) =
        q_f(c - s)), but all tokens instead of the last 512, centred, and
        a global instead of a rank-averaged quantile.
- Device-limited routing: experts are partitioned across
  ``config.moe.device_group_size`` devices; under torch.distributed each rank
  only routes to and computes its local experts.
- Shared experts are always active (always full-width, even under LatentMoE).
- LatentMoE (v5.5, ``config.moe.latent_dim``): when set, a shared
  ``down_proj`` maps hidden -> latent once, the router and the routed
  experts operate in latent space (experts are latent->expert_hidden->latent),
  and a shared ``up_proj`` maps the combined latent output back to hidden.
  Load-balancing statistics are unchanged (still router-selection based).
- Expert activation (v5.5, ``config.moe.activation``): "swiglu" (SiLU-gated
  GLU) or "situ" — a simplified K3-style SiTU-GLU:
      t(x) = softcap * tanh(x / softcap)
      situ_glu(a, b) = silu(t(a)) * t(b)
  with an RMSNorm on the gated product before the output projection. The
  soft cap bounds every pre-activation (|t(x)| <= softcap), which damps
  outlier activations in the expert MLP.

The transformer layer is responsible for normalizing the input exactly once
before calling this module (there is intentionally no input norm here).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# F.grouped_mm is new in torch 2.10; older builds run "grouped" as
# "batched" (same experts, padded bmm) instead of failing (v5.156).
HAS_GROUPED_MM = hasattr(F, "grouped_mm")

try:
    from evastar.src.attention.mla import RMSNorm
except ImportError:  # pragma: no cover - package-relative fallback
    from ..attention.mla import RMSNorm


def _situ_cap(x, softcap):
    """K3-style tanh soft-cap: t(x) = softcap * tanh(x / softcap).

    Bounded: |t(x)| < softcap for all finite x.
    """
    return softcap * torch.tanh(x / softcap)


def k3_situ(a, b, beta, linear_beta):
    """Kimi K3 situ GLU (v5.142), as llama.cpp's kimi-k3 graph computes it:
    beta * tanh(a / beta) * sigmoid(a) * b', with b' = linear_beta *
    tanh(b / linear_beta) when linear_beta > 0, else b."""
    h = beta * torch.tanh(a / beta) * torch.sigmoid(a)
    if linear_beta > 0:
        b = linear_beta * torch.tanh(b / linear_beta)
    return h * b


class SwiGLUExpert(nn.Module):
    """SiLU-gated GLU expert MLP: w2(silu(a) * b) with w13(x) = (a, b).

    v5.68 (roadmap U5): optional gpt-oss clamping. With ``limit`` set or
    non-default ``alpha`` / ``up_offset`` the activation becomes
    a = min(a, limit), b = clamp(b, -limit, limit),
    w2(a * sigmoid(alpha * a) * (b + up_offset)) — gpt-oss uses limit 7,
    alpha 1.702, up_offset 1. The pre-w2 activation is then bounded:
    a * sigmoid(alpha * a) lies in (-0.2785 / alpha, limit] and
    b + up_offset in [up_offset - limit, up_offset + limit]. Defaults keep
    the original ``F.silu`` path bit for bit.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, limit=None, alpha=1.0,
                 up_offset=0.0, situ=None):
        super().__init__()
        # v5.142: situ = (beta, linear_beta) swaps in k3_situ
        self.situ = situ
        self.w13 = nn.Linear(in_dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.limit = None if limit is None else float(limit)
        self.alpha = float(alpha)
        self.up_offset = float(up_offset)
        self.clamped = not (self.limit is None and self.alpha == 1.0
                            and self.up_offset == 0.0)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        if self.situ is not None:
            return self.w2(k3_situ(a, b, *self.situ))
        if not self.clamped:
            return self.w2(F.silu(a) * b)
        if self.limit is not None:
            a = a.clamp(max=self.limit)
            b = b.clamp(-self.limit, self.limit)
        return self.w2(a * torch.sigmoid(self.alpha * a) * (b + self.up_offset))


class SiTUExpert(nn.Module):
    """Simplified SiTU-GLU expert (K3-style tanh soft-cap, v5.5).

    situ_glu(a, b) = silu(t(a)) * t(b) with t the soft-cap above; an
    RMSNorm is applied to the gated product before the output projection
    (w2). Not the full K3 SiTU block (no per-channel learnable caps).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, softcap, eps=1e-6):
        super().__init__()
        if softcap <= 0:
            raise ValueError(f"situ_softcap must be positive, got {softcap}")
        self.softcap = float(softcap)
        self.w13 = nn.Linear(in_dim, 2 * hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim, eps=eps)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        h = F.silu(_situ_cap(a, self.softcap)) * _situ_cap(b, self.softcap)
        return self.w2(self.norm(h))


def build_expert(config, in_dim, out_dim):
    """Expert MLP selected by config.moe.activation ("swiglu" | "situ")."""
    hidden_dim = config.moe.expert_hidden_size
    if config.moe.activation == "situ":
        return SiTUExpert(in_dim, hidden_dim, out_dim,
                          softcap=config.moe.situ_softcap,
                          eps=config.rms_norm_eps)
    m = config.moe
    return SwiGLUExpert(in_dim, hidden_dim, out_dim,
                        limit=getattr(m, "swiglu_limit", None),
                        alpha=getattr(m, "swiglu_alpha", 1.0),
                        up_offset=getattr(m, "swiglu_up_offset", 0.0),
                        situ=situ_params(config))


def situ_params(config):
    """(beta, linear_beta) for moe.activation "situ_k3", else None."""
    m = config.moe
    if m.activation != "situ_k3":
        return None
    return float(m.situ_beta), float(m.situ_linear_beta)


class DeviceLimitedMoE(nn.Module):
    """MoE with sigmoid routing, aux-free bias balancing, device-limited experts."""

    # Fixed expert-call block size; see the dispatch comment in forward().
    _EXPERT_BLOCK = 64
    # Sliding-window length for the per-expert routing-margin buffer used by
    # the "quantile" balance strategy.
    _MARGIN_BUF = 512

    def __init__(self, config, device_map=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.num_shared = config.moe.num_shared_experts
        self.top_k = config.moe.num_activated_experts
        self.expert_hidden = config.moe.expert_hidden_size
        self.bias_update_rate = config.moe.bias_update_rate
        # v5.5: LatentMoE / balance strategy. getattr for robustness against
        # configs built before the fields existed.
        self.latent_dim = config.moe.latent_dim
        self.balance_strategy = config.moe.balance_strategy
        if self.balance_strategy not in ("heuristic", "quantile", "smebu",
                                         "quantile_balancing"):
            raise ValueError(
                f"moe.balance_strategy must be 'heuristic', 'quantile', "
                f"'smebu' or 'quantile_balancing', got "
                f"{self.balance_strategy!r}"
            )

        # Device mapping: expert_id -> device_id (uses device_group_size).
        self.device_map = device_map or self._default_device_map()
        self.num_devices = max(self.device_map.values()) + 1

        # LatentMoE: router + routed experts live in the latent space.
        self.router_dim = self.latent_dim or self.hidden_size
        if self.latent_dim is not None:
            self.down_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
            self.up_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        else:
            self.down_proj = None
            self.up_proj = None

        # Router: sigmoid activation (NOT softmax). v5.142
        # moe.router_input="full" scores the full-width input instead of the
        # latent (Kimi K3 / llama.cpp kimi-k3); experts stay in the latent.
        self.route_full = config.moe.router_input == "full"
        self.router = nn.Linear(self.hidden_size if self.route_full else self.router_dim,
                                self.num_experts, bias=False)

        # Aux-free load-balancing bias: used ONLY for top-k selection, never
        # in the gating weights, and never updated by gradient descent.
        self.route_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        # Accumulated selection counts since the last update_bias() call.
        self.register_buffer("expert_load", torch.zeros(self.num_experts))
        if self.balance_strategy == "quantile":
            # Sliding window of per-expert routing margins (selection score
            # minus the token's top-k boundary). Uniform write count across
            # experts, so a single scalar fill counter is exact.
            self.register_buffer(
                "margin_buffer",
                torch.zeros(self.num_experts, self._MARGIN_BUF),
            )
            self.register_buffer("margin_count", torch.zeros((), dtype=torch.long))
        if self.balance_strategy == "quantile_balancing":
            # Per-expert margin histogram since the last update_bias(), and
            # its range (fixed between updates because the bias is).
            self.quantile_bins = int(config.moe.quantile_bins)
            self.register_buffer(
                "margin_hist",
                torch.zeros(self.num_experts, self.quantile_bins,
                            dtype=torch.float64))
            self.register_buffer("hist_range", torch.tensor([-1.0, 2.0]))
        if self.balance_strategy == "smebu":
            # SMEBU momentum of the soft-clamped load violation (one per
            # expert); only registered for "smebu", so other strategies'
            # state dicts are unchanged.
            self.smebu_beta = float(config.moe.smebu_beta)
            self.smebu_kappa = float(config.moe.smebu_kappa)
            self.register_buffer("bias_momentum", torch.zeros(self.num_experts))
        # v5.63 (roadmap Q3, Qwen3 MoE): optional load-balancing loss
        # records. None = off (no records, forward unchanged). When set,
        # every TRAINING forward appends (counts, sum of per-token
        # normalized gate probs, token count); model_v5.balance_loss()
        # turns them into the loss with micro- or global-batch f_i and
        # clears them.
        self.balance_loss_scope = config.moe.balance_loss_scope
        self._balance_records = []
        # Lazy cache for the distributed local-expert id list (O3): the
        # expert->device map is fixed at construction and a rank's partition
        # never changes within a process, so the list is built at most once
        # per (rank, device) instead of on every forward.
        self._local_experts_cache = None
        # v5.106: "blocked" (bit-stable per token) or "batched" (bmm);
        # v5.149: "grouped" (F.grouped_mm, no padding).
        self.dispatch = config.moe.dispatch

        # Routed experts (latent<->expert_hidden under LatentMoE).
        self.experts = nn.ModuleList([
            build_expert(config, self.router_dim, self.router_dim)
            for _ in range(self.num_experts)
        ])

        # Shared experts (always active, always FULL-WIDTH even under
        # LatentMoE).
        self.shared_experts = nn.ModuleList([
            build_expert(config, self.hidden_size, self.hidden_size)
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

        ``flat`` is the routing input ([N, router_dim] — latent rows under
        LatentMoE). Selection scores include route_bias; gating weights
        do not.
        """
        router_logits = self.router(flat)
        scores = torch.sigmoid(router_logits)                    # bias-free gates
        self._last_scores = scores if self.balance_loss_scope else None
        scores_for_select = scores + self.route_bias             # selection only

        if self.balance_strategy == "quantile" and self.training \
                and torch.is_grad_enabled():
            # Training-only margin statistics (same guard as expert_load).
            # margin_e = biased selection score minus the (K+1)-th largest
            # score (the first NON-selected score), so
            #     margin_e > 0  <=>  expert e is in the token's top-k.
            # (v5.5 M1 fix: the boundary must be the (K+1)-th largest, not
            # the K-th — using the top-k boundary itself makes margin > 0
            # equivalent to ranking top-(K-1), the K/E target becomes
            # unreachable, and the bias drifts without bound.)
            if self.top_k < self.num_experts:
                kth = scores_for_select.kthvalue(
                    self.num_experts - self.top_k, dim=-1, keepdim=True
                ).values
            else:
                # K == E: every expert is selected for every token, so the
                # selection boundary is -inf (all margins positive).
                kth = torch.full_like(scores_for_select[..., :1], float("-inf"))
            self._record_margins(scores_for_select - kth)
        if self.balance_strategy == "quantile_balancing" and self.training \
                and torch.is_grad_enabled() and self.top_k < self.num_experts:
            cutoff = scores_for_select.kthvalue(
                self.num_experts - self.top_k, dim=-1, keepdim=True).values
            self._record_histogram(cutoff - scores)

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

    @torch.no_grad()
    def _record_margins(self, margins):
        """Append [N, E] margins to the per-expert sliding window.

        Every expert receives the same number of samples per call, so the
        fill level is uniform and tracked by a single scalar counter.
        """
        m = margins.detach().t().to(self.margin_buffer.dtype)  # [E, N]
        N = m.shape[1]
        buf = self._MARGIN_BUF
        if N >= buf:
            self.margin_buffer.copy_(m[:, -buf:])
            self.margin_count.fill_(buf)
        else:
            self.margin_buffer.copy_(
                torch.cat([self.margin_buffer[:, N:], m], dim=1)
            )
            self.margin_count.fill_(min(int(self.margin_count) + N, buf))

    @torch.no_grad()
    def _record_histogram(self, margins):
        """Add [N, E] margins (cutoff - score) to the per-expert histogram.
        The range is set from the current bias when the histogram is empty;
        margins lie inside it because cutoff and score are both a sigmoid
        (0..1) plus a bias."""
        if self.margin_hist.sum() == 0:
            b = self.route_bias.detach().float()
            self.hist_range.copy_(torch.stack([b.min() - 1.0, b.max() + 1.0]))
        lo, hi = self.hist_range.tolist()
        nb = self.quantile_bins
        idx = ((margins.detach().float() - lo) / (hi - lo) * nb).floor() \
            .clamp_(0, nb - 1).long().t()                       # [E, N]
        self.margin_hist.scatter_add_(
            1, idx, torch.ones_like(idx, dtype=self.margin_hist.dtype))

    @staticmethod
    def histogram_quantile(hist, lo, hi, q):
        """Per-row q-quantile of a [E, bins] histogram over [lo, hi],
        linearly interpolated inside the bin where the CDF crosses q."""
        nb = hist.shape[1]
        width = (hi - lo) / nb
        total = hist.sum(1, keepdim=True).clamp(min=1e-12)
        cdf = hist.cumsum(1) / total
        i = torch.searchsorted(cdf, torch.full_like(cdf[:, :1], q)) \
            .clamp(max=nb - 1)
        prev = torch.where(i > 0, cdf.gather(1, (i - 1).clamp(min=0)),
                           torch.zeros_like(cdf[:, :1]))
        mass = (cdf.gather(1, i) - prev).clamp(min=1e-12)
        frac = ((q - prev) / mass).clamp(0.0, 1.0)
        return (lo + width * (i.double() + frac)).squeeze(1)

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

        # LatentMoE: one shared projection into the latent space; routing and
        # routed experts all operate there.
        z = self.down_proj(flat) if self.down_proj is not None else flat

        topk_indices, topk_gates = self._route(flat if self.route_full else z)

        # Accumulate routing-load statistics for aux-free bias updates.
        # Training-only (M-C2): eval / inference forwards (e.g. RL rollouts)
        # must not pollute the statistics that drive update_bias().
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                counts_all = torch.bincount(
                    topk_indices.reshape(-1), minlength=self.num_experts
                )
                self.expert_load += counts_all.to(self.expert_load.dtype)
            if self.balance_loss_scope:
                sc = self._last_scores
                probs = sc / sc.sum(-1, keepdim=True).clamp(min=1e-9)
                self._balance_records.append(
                    (counts_all.detach().float(), probs.sum(0), N))
        self._last_scores = None

        if self.dispatch in ("batched", "grouped") and self._batchable():
            output_z = self._dispatch_batched(z, topk_indices, topk_gates)
        else:
            output_z = self._dispatch_blocked(z, topk_indices, topk_gates)

        # Back to hidden width (LatentMoE), then shared experts (always
        # active, always full-width). v5.162: each shared output is cast to
        # the routed output's dtype before the add (the same type promotion
        # eager does); torch 2.8's inductor otherwise fused "fp32 routed +
        # bf16 shared down-projection" into one addmm with mixed dtypes,
        # which failed on CUDA under --compile.
        output = self.up_proj(output_z) if self.up_proj is not None else output_z
        for shared in self.shared_experts:
            s = shared(flat)
            output = output + s.to(torch.promote_types(output.dtype, s.dtype))

        return output.view(B, seq, H)

    # v5.158: both dispatch paths (this and _dispatch_batched) are
    # host-synced with data-dependent shapes (per-expert counts via
    # .tolist()); torch.compile runs it eagerly (torch 2.8's dynamo failed
    # with GuardOnDataDependentSymNode on the count loop) and compiles the
    # rest of the layer.
    @torch.compiler.disable
    def _dispatch_blocked(self, z, topk_indices, topk_gates):
        """Routed-expert output [N, D] (default "blocked" dispatch)."""
        N, D = z.shape
        K = topk_indices.shape[1]
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
        token_idx = torch.arange(N, device=z.device).repeat_interleave(K)
        order = flat_expert.argsort()
        sorted_tokens = token_idx[order]
        sorted_weights = flat_weight[order]
        counts = torch.bincount(flat_expert, minlength=self.num_experts).tolist()

        output_z = torch.zeros_like(z)
        start = 0
        block = self._EXPERT_BLOCK
        for eid, cnt in enumerate(counts):
            if cnt:
                idx = sorted_tokens[start : start + cnt]
                rows = z[idx]
                outs = []
                for b0 in range(0, cnt, block):
                    chunk = rows[b0 : b0 + block]
                    n_real = chunk.shape[0]
                    if n_real < block:
                        chunk = torch.cat(
                            [chunk, chunk.new_zeros(block - n_real, D)], dim=0
                        )
                    outs.append(self.experts[eid](chunk)[:n_real])
                expert_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
                # fp32 accumulation regardless of the expert compute dtype:
                # under bf16 autocast expert outputs are bf16 while z (a
                # norm output) stays fp32 -- index_add_ demands matching
                # dtypes. First caught on a real GPU (v5.33 mid run); CPU
                # fp32 never exercised this path.
                output_z.index_add_(
                    0, idx,
                    (expert_out * sorted_weights[start : start + cnt].unsqueeze(-1))
                    .to(output_z.dtype)
                )
            start += cnt

        return output_z

    def _batchable(self):
        """True when every routed expert is a plain SwiGLU / SiTU expert
        with nn.Linear projections (not quantized or FP8-converted)."""
        kinds = {type(e) for e in self.experts}
        if len(kinds) != 1 or kinds.pop() not in (SwiGLUExpert, SiTUExpert):
            return False
        return all(type(e.w13) is nn.Linear and type(e.w2) is nn.Linear
                   for e in self.experts)

    @torch.compiler.disable
    def _dispatch_batched(self, z, topk_indices, topk_gates):
        """Routed-expert output [N, D] with all experts in two bmm calls.

        Token slots are sorted by expert and scattered into a zero-padded
        [A, C, D] tensor (A = experts that received tokens, C = the largest
        expert load this forward; one host sync, as the blocked path's
        ``.tolist()``); the stacked weights [A, 2F, D] and [A, D, F] then
        run every active expert at once.
        Padding rows are never read back. Each expert's output equals the
        blocked path to float tolerance; it is not bit-identical, because
        the GEMM shape depends on C.

        ``moe.dispatch="grouped"`` (v5.149) runs the same two GEMMs as
        ``F.grouped_mm`` over the expert-sorted rows, with no padding: the
        [A, C, D] scatter and the A * C - N * K padded rows are gone.
        A torch without ``F.grouped_mm`` (before 2.10) runs the batched
        GEMMs instead (v5.156).
        """
        N, D = z.shape
        K = topk_indices.shape[1]
        E = self.num_experts
        flat_expert = topk_indices.reshape(-1)                       # [N*K]
        token_idx = torch.arange(N, device=z.device).repeat_interleave(K)
        order = flat_expert.argsort(stable=True)
        sorted_expert = flat_expert[order]
        sorted_tokens = token_idx[order]
        sorted_weights = topk_gates.reshape(-1)[order]
        counts = torch.bincount(flat_expert, minlength=E)
        # Only experts that received tokens are stacked, so an idle
        # expert's weights get no gradient (None, as in the blocked path)
        # rather than a zero one that AdamW would still step and decay.
        host_counts = counts.tolist()
        active = [i for i, c in enumerate(host_counts) if c]
        compact = torch.full((E,), -1, dtype=torch.long, device=z.device)
        compact[active] = torch.arange(len(active), device=z.device)
        row = compact[sorted_expert]
        experts = [self.experts[i] for i in active]
        w13 = torch.stack([e.w13.weight for e in experts])           # [A, 2F, D]
        w2 = torch.stack([e.w2.weight for e in experts])             # [A, D, F]
        grouped = self.dispatch == "grouped" and HAS_GROUPED_MM
        if grouped:
            offs = counts[active].cumsum(0).to(torch.int32)
            x = z[sorted_tokens]                                     # [N*K, D]
            a, b = F.grouped_mm(x, w13.transpose(1, 2), offs=offs).chunk(2, dim=-1)
        else:
            starts = counts.cumsum(0) - counts
            slot = torch.arange(N * K, device=z.device) - starts[sorted_expert]
            x = z.new_zeros(len(active), max(host_counts), D)
            x[row, slot] = z[sorted_tokens]
            a, b = torch.bmm(x, w13.transpose(1, 2)).chunk(2, dim=-1)   # [A, C, F]
        e0 = self.experts[0]
        if isinstance(e0, SiTUExpert):
            h = F.silu(_situ_cap(a, e0.softcap)) * _situ_cap(b, e0.softcap)
            g = torch.stack([e.norm.weight for e in experts])        # [A, F]
            if e0.norm.zero_centered:
                g = 1.0 + g
            g = g[row] if grouped else g[:, None, :]
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True)
                                + e0.norm.eps) * g
        elif e0.situ is not None:
            h = k3_situ(a, b, *e0.situ)
        elif not e0.clamped:
            h = F.silu(a) * b
        else:
            if e0.limit is not None:
                a = a.clamp(max=e0.limit)
                b = b.clamp(-e0.limit, e0.limit)
            h = a * torch.sigmoid(e0.alpha * a) * (b + e0.up_offset)
        if grouped:
            y = F.grouped_mm(h, w2.transpose(1, 2), offs=offs)        # [N*K, D]
        else:
            y = torch.bmm(h, w2.transpose(1, 2))[row, slot]          # [N*K, D]

        out = y * sorted_weights.unsqueeze(-1)
        output_z = torch.zeros_like(z)
        # fp32 accumulation, as in the blocked path (bf16 autocast).
        output_z.index_add_(0, sorted_tokens, out.to(output_z.dtype))
        return output_z

    @torch.no_grad()
    def update_bias(self, expert_load=None, step=None):
        """Aux-free bias update.

        "heuristic" (DeepSeek-V3 style, v5.4): experts whose load is above
        the mean get their selection bias decreased by ``step``; below-mean
        experts get it increased. Pass an explicit ``expert_load`` tensor,
        or omit it to consume (and reset) the statistics accumulated during
        forward passes.

        "quantile" (v5.5, simplified estimator): each expert's bias is
        shifted to the point that would have made its recent selection
        frequency equal the uniform-load target::

            bias_e -= quantile(margins_e, 1 - target_frac)
            target_frac = top_k / num_experts

        where margins_e is the sliding window of (biased selection score -
        top-k boundary) recorded by ``_record_margins``. Simplifications:
        the top-k boundary tau is treated as fixed across the bias shift
        (it actually moves when biases move), and the window is a plain
        FIFO rather than a proper quantile sketch. Converges in a few
        updates on stationary distributions. The accumulated ``expert_load``
        is consumed in both strategies. ``expert_load``/``step`` arguments
        are accepted for interface compatibility and ignored by "quantile".

        Distributed (M-C3): when a process group is active, BOTH strategies
        first turn their local statistics into a global one so every rank
        applies the SAME update — ``route_bias`` has requires_grad=False, so
        DDP never syncs it and divergent per-rank updates would silently
        diverge the replicas. "heuristic" all-reduces the load (SUM);
        "quantile" all-reduces the per-rank margin quantiles (SUM / world
        size = mean over ranks' quantile estimates). ``route_bias`` needs
        no separate sync: the update rule is deterministic, so identical
        updates on every rank produce identical biases (they start
        identical under DDP broadcast). Single-process behaviour is
        unchanged.
        """
        if self.balance_strategy == "quantile_balancing":
            hist = self.margin_hist
            if torch.distributed.is_available() \
                    and torch.distributed.is_initialized():
                # exact global histogram: the quantile of all ranks' tokens
                torch.distributed.all_reduce(
                    hist, op=torch.distributed.ReduceOp.SUM)
            if hist.sum() > 0:
                lo, hi = self.hist_range.tolist()
                q = self.histogram_quantile(
                    hist, lo, hi, self.top_k / self.num_experts)
                self.route_bias.copy_((q - q.mean()).to(self.route_bias.dtype))
            hist.zero_()
            self.expert_load.zero_()
            return self.route_bias

        if self.balance_strategy == "quantile":
            n = int(self.margin_count)
            if n > 0:
                target_frac = self.top_k / self.num_experts
                # _record_margins left-shifts and appends, so the valid
                # samples are the LAST n columns of the buffer.
                window = self.margin_buffer[:, self._MARGIN_BUF - n:].float()
                if self.top_k >= self.num_experts \
                        or not torch.isfinite(window).any():
                    # K == E (explicitly allowed by config validation)
                    # selects every expert for every token: _route records
                    # margins against a -inf boundary, so the window is all
                    # +inf and torch.quantile over it is NaN — which would
                    # poison route_bias forever. Selection bias is also
                    # meaningless when all experts are always chosen (the
                    # gates are bias-free). Skip the update and drop the
                    # poisoned window; the finiteness guard covers the same
                    # hazard for any other non-finite window.
                    self.margin_count.zero_()
                else:
                    q = torch.quantile(window, 1.0 - target_frac, dim=1)
                    if torch.distributed.is_available() \
                            and torch.distributed.is_initialized():
                        # Each rank's window holds only its LOCAL tokens; average
                        # the per-rank quantile estimates so the bias update is
                        # driven by global statistics and stays identical across
                        # ranks (mirrors the heuristic path's load all-reduce).
                        torch.distributed.all_reduce(
                            q, op=torch.distributed.ReduceOp.SUM)
                        q = q / torch.distributed.get_world_size()
                    self.route_bias.add_(-q.to(self.route_bias.dtype))
                    self.margin_count.zero_()
            self.expert_load.zero_()
            return self.route_bias

        load = self.expert_load if expert_load is None else expert_load.clone()
        load = load.to(device=self.route_bias.device, dtype=self.route_bias.dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(load, op=torch.distributed.ReduceOp.SUM)
        u = self.bias_update_rate if step is None else step
        mean = load.mean()
        if self.balance_strategy == "smebu":
            # v5.72: magnitude-aware, soft-clamped, momentum-smoothed. No
            # update when nothing was routed (mean 0 has no violation).
            if mean > 0:
                v = (mean - load) / mean
                self.bias_momentum.mul_(self.smebu_beta).add_(
                    (1.0 - self.smebu_beta) * torch.tanh(self.smebu_kappa * v))
                self.route_bias.add_(u * self.bias_momentum)
        else:
            self.route_bias.add_(-u * torch.sign(load - mean))
        if expert_load is None:
            self.expert_load.zero_()
        return self.route_bias
