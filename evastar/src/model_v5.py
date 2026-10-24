"""HeliosLM v5.5 - Unified Model

Model core: token embedding + N x (pre-norm attention + pre-norm Sigmoid
MoE) + final RMSNorm + LM head, with optional MTP modules and multimodal
encoders. Attention is MLA on every layer by default; with
``config.hybrid_attention.enabled`` (v5.5) layers are interleaved: layer i
is MLA iff ``i == 0 or i % every == every - 1``, GatedDeltaAttention
(fixed-size recurrent state) otherwise.

Forward contract (relied upon by MTP / GRPO / training modules):
    logits, hidden_states, past_key_values = model(
        input_ids, attention_mask=None, position_ids=None,
        past_key_values=None, use_cache=False,
        images=None, audio_features=None,
    )
  - logits: [B, L, vocab]
  - hidden_states: [B, L, hidden] after the final norm
  - past_key_values: per-layer cache tuples, or None. MLA layers follow
    config.attention.use_absorption: (c_kv, k_rope) when absorbed
    (default), (k_nope, k_rope, v) otherwise; dim 2 is the sequence
    length for every tensor in both layouts. GatedDeltaAttention layers
    return (state,) where state is a StateTensor [B, H, Dk, Dv] marked
    with ``_is_recurrent_state = True`` — NOT sliceable along dim 2;
    rollback = restore a pre-speculation clone (see inference/mtp.py).

Packed sequences (per-document ``position_ids`` restarts, v5.4 B1) are
supported on pure-MLA models only. In a hybrid model GatedDeltaAttention
cannot segment its recurrent state, so a position restart raises a loud
ValueError (v5.5 M3) instead of leaking across documents.

PagedAttention / block management is intentionally NOT part of the model
layer; it lives in the inference engine.
"""
import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm is defined once in attention.mla (O14); re-exported here so the
# historical public name ``evastar.src.model_v5.RMSNorm`` keeps working.
from evastar.src.attention.gated_conv import GatedShortConv
from evastar.src.attention.mla import MLA, RMSNorm
from evastar.src.attention.linear_attention import (
    GatedDeltaAttention, past_seq_len)
from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE
from evastar.src.inference.mtp import MTPModule, MTPDecoder
from evastar.src.vision.navit import NaViTEncoder
from evastar.src.audio.streaming_encoder import StreamingAudioEncoder


def _is_full_attention_layer(config, layer_idx):
    """Hybrid interleave rule (v5.5): MLA iff layer 0 or
    ``idx % every == every - 1``; GatedDeltaAttention otherwise. With
    ``first_layer_full=False`` (v5.57, K3 block order) the layer-0
    exception is dropped. Layers in ``gated_conv_layers`` (v5.126, LB9)
    are never MLA."""
    if layer_idx in (config.gated_conv_layers or ()):
        return False                      # LB9: a gated short-conv layer
    hcfg = config.hybrid_attention
    if hcfg is None or not getattr(hcfg, "enabled", False):
        return True
    if layer_idx == 0 and getattr(hcfg, "first_layer_full", True):
        return True
    every = hcfg.full_attention_every
    return layer_idx % every == every - 1


class DenseFFN(nn.Module):
    """Dense SwiGLU FFN for the ``moe.first_k_dense`` layers (B2-3, v5.65):
    w2(silu(a) * b) with w13(x) = (a, b), width ``config.intermediate_size``.
    ``is_dense`` lets tooling tell it from a routed MoE layer."""

    is_dense = True

    def __init__(self, config):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.w13 = nn.Linear(h, 2 * i, bias=False)
        self.w2 = nn.Linear(i, h, bias=False)
        from evastar.src.moe.sigmoid_moe import situ_params
        self.situ = situ_params(config)     # v5.142 "situ_k3" (else SwiGLU)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        if self.situ is not None:
            from evastar.src.moe.sigmoid_moe import k3_situ
            return self.w2(k3_situ(a, b, *self.situ))
        return self.w2(F.silu(a) * b)


def decay_param_groups(model, weight_decay):
    """AdamW param groups (v5.63, roadmap Q2).

    Matrices (dim >= 2) and ZERO-CENTERED norm weights are decayed;
    standard norm gains, biases and other 1-D parameters are not. With
    standard RMSNorm, decay would pull the gain toward 0 (a shrinking
    layer); with ``zero_centered_norm`` the stored weight is ``gain - 1``,
    so decay pulls the gain toward 1 — the reason Qwen3-Next centers it.
    Every trainable parameter lands in exactly one group.
    """
    from evastar.src.attention.mla import RMSNorm
    zc = {id(m.weight) for m in model.modules()
          if isinstance(m, RMSNorm) and m.zero_centered}
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 or id(p) in zc else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def balance_loss(model):
    """Load-balancing loss E * sum_i f_i * P_i (v5.63, roadmap Q3).

    Averaged over MoE layers; 1.0 at perfect balance. f_i is the share of
    top-k slots routed to expert i (no gradient), P_i the mean per-token
    normalized gate probability (differentiable). Scope comes from
    ``moe.balance_loss_scope``:
      - "micro": the loss of each recorded forward, averaged;
      - "global": f_i and P_i over ALL forwards recorded since the last
        call — with gradient accumulation this is exactly the loss of the
        concatenated global batch (Qwen's global-batch balancing), which
        lets experts specialize per domain inside the batch.
    Records are cleared. Returns None when no layer recorded anything.
    Keep the micro-batch graphs alive until this loss is backpropagated
    (sum the micro-batch LM losses with it and call backward once).
    """
    from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE
    losses = []
    for m in model.modules():
        if not isinstance(m, DeviceLimitedMoE) or not m._balance_records:
            continue
        recs, m._balance_records = m._balance_records, []
        E, K = m.num_experts, m.top_k
        if m.balance_loss_scope == "global":
            counts = sum(r[0] for r in recs)
            p_sum = sum(r[1] for r in recs)
            n = sum(r[2] for r in recs)
            losses.append(E * ((counts / (n * K)) * (p_sum / n)).sum())
        else:
            losses.append(sum(E * ((c / (n * K)) * (p / n)).sum()
                              for c, p, n in recs) / len(recs))
    if not losses:
        return None
    return sum(losses) / len(losses)


def _shared_kv_kw(shared_kv):
    """v5.71 U6: only KV-sharing MLA layers receive ``shared_kv``, so every
    other attention call keeps its exact pre-U6 signature."""
    return {} if shared_kv is None else {"shared_kv": shared_kv}


class HeliosLMv5Layer(nn.Module):
    """Single transformer layer: pre-norm attention + pre-norm Sigmoid MoE.

    Each sublayer input is normalized exactly once:
      h = h + attn(pre_attn_norm(h))
      h = h + moe(pre_moe_norm(h))

    Attention residuals (v5.5, config.use_attention_residuals): the input
    to attention additionally receives ``attn_res_gate * attn_res`` (the
    accumulated sum of previous layers' attention outputs). The accumulator
    is threaded EXPLICITLY (v5.5 M4/M5 fix): the caller passes ``attn_res``
    and the layer returns the updated accumulator as the third element of
    its return tuple — no ``self._last_attn_out`` attribute side channel
    (a side channel silently truncates cross-layer residual gradients
    under reentrant checkpointing and is invisible to the DualPipe
    recompute scheme). The gate is a raw scalar Parameter (init 0.1) so
    the model-wide Linear init does not touch it.

    Returns ``(hidden_states, present_kv, attn_res_new)`` where
    ``attn_res_new`` is the updated accumulator (``attn_res + attn_out``
    when the gate exists; the input ``attn_res`` passed through unchanged
    otherwise). With ``use_attention_residuals=False`` the numerics are
    bit-identical to v5.4. With ``use_hyper_connections=True`` (v5.7) the
    first element is the widened [B, L, n, d] stream and the third is
    always None (mutually exclusive with attention residuals).
    """

    def __init__(self, config, layer_idx, shared_indexer=None):
        super().__init__()
        self.layer_idx = layer_idx  # kept for external cache plumbing/debugging
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=bool(config.zero_centered_norm))
        if layer_idx in (config.gated_conv_layers or ()):
            self.attention = GatedShortConv(config)     # v5.126 LB9
        elif _is_full_attention_layer(config, layer_idx):
            self.attention = MLA(config, shared_indexer=shared_indexer,
                                 layer_idx=layer_idx)
        else:
            self.attention = GatedDeltaAttention(config)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=bool(config.zero_centered_norm))
        # B2-3 (v5.65): the first ``moe.first_k_dense`` layers use a dense
        # FFN. The attribute keeps the name ``moe`` so every forward path
        # (plain, hyper-connections) is unchanged; routing statistics and
        # balance losses only ever look at DeviceLimitedMoE instances.
        if layer_idx < int(config.moe.first_k_dense or 0):
            self.moe = DenseFFN(config)
        else:
            self.moe = DeviceLimitedMoE(config)
        # v5.72 (roadmap U7): depth-scaled sandwich norm — one RMSNorm on
        # each sublayer output, gain initialized to 1/sqrt(num_layers)
        # (the model-wide init only touches Linear / Embedding).
        if config.sandwich_norm:
            g = config.num_hidden_layers ** -0.5
            zc = bool(config.zero_centered_norm)
            self.post_attn_norm = RMSNorm(config.hidden_size,
                                          eps=config.rms_norm_eps, zero_centered=zc)
            self.post_moe_norm = RMSNorm(config.hidden_size,
                                         eps=config.rms_norm_eps, zero_centered=zc)
            with torch.no_grad():
                for n in (self.post_attn_norm, self.post_moe_norm):
                    n.weight.fill_(g - 1.0 if zc else g)
        else:
            self.post_attn_norm = self.post_moe_norm = None
        self.res_block = (bool(config.use_attention_residuals)
                          and config.attention_residual_style == "block")
        self.attn_res_gate = None
        if self.res_block:
            # v5.142 block attention residuals (see block_res_mix)
            self.res_bank = layer_idx % config.attn_res_block_size == 0
            self.res_eps = config.rms_norm_eps
            self.attn_res_score = nn.Parameter(torch.zeros(config.hidden_size))
            self.ffn_res_score = nn.Parameter(torch.zeros(config.hidden_size))
        elif bool(config.use_attention_residuals):
            self.attn_res_gate = nn.Parameter(torch.tensor(0.1))

        # Hyper-Connections (v5.7, simplified HC/mHC): the residual stream
        # is widened to n virtual branches. Each sublayer reads through a
        # STATIC mixing matrix A (identity init; unit-norm columns are the
        # manifold constraint — A never trains, so the constraint holds by
        # construction) and writes back through a LEARNABLE matrix B
        # (zero init), giving h_l = A h_{l-1} + B f(A h_{l-1}) per branch.
        # At init B = 0, so the streams pass through unchanged and the
        # network is exactly a vanilla transformer (the sublayer still runs
        # so the KV cache populates). Simplifications vs the paper: single
        # static A per sublayer (not per-layer-index dynamic), mean readout,
        # no width/FLOP expansion is avoided — branches share the sublayer
        # weights and are folded into the batch dim.
        self.use_hyper_connections = bool(
            config.use_hyper_connections)
        if self.use_hyper_connections:
            n = config.hyper_connection_branches
            self.hc_num_branches = n
            self.register_buffer("hc_attn_A", torch.eye(n))
            self.hc_attn_B = nn.Parameter(torch.zeros(n, n))
            self.register_buffer("hc_moe_A", torch.eye(n))
            self.hc_moe_B = nn.Parameter(torch.zeros(n, n))
            # mHC (v5.66, B2-4): learnable doubly stochastic read matrices.
            self.hc_sinkhorn = bool(
                config.hyper_connection_sinkhorn)
            if self.hc_sinkhorn:
                self.hc_sinkhorn_iters = int(
                    config.hyper_connection_sinkhorn_iters)
                self.hc_attn_logits = nn.Parameter(torch.zeros(n, n))
                self.hc_attn_alpha = nn.Parameter(torch.zeros(()))
                self.hc_moe_logits = nn.Parameter(torch.zeros(n, n))
                self.hc_moe_alpha = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------
    # Hyper-Connection stream step (v5.7)
    # ------------------------------------------------------------------
    @staticmethod
    def sinkhorn(logits, iters):
        """Sinkhorn-Knopp in the log domain: alternate row and column
        normalization of exp(tanh(logits)); ends on a column step, so
        columns sum to 1 exactly and rows to the convergence tolerance.

        The logits are squashed to [-1, 1] first: Sinkhorn's convergence
        rate degrades with the spread of the entries, and unbounded logits
        left rows up to 9% off after 20 iterations (T104), i.e. the
        "doubly stochastic" claim would not hold. With entries in
        [e^-1, e^1] 20 iterations reach float precision for any logits.
        The identity end stays reachable through the alpha mix."""
        log_p = torch.tanh(logits)
        for _ in range(iters):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        return log_p.exp()

    def hc_read_matrix(self, which):
        """The HC read matrix for ``which`` in {"attn", "moe"}: the static
        identity buffer, or with mHC (1 - a) I + a Sinkhorn(exp(L))."""
        A = getattr(self, f"hc_{which}_A")
        if not getattr(self, "hc_sinkhorn", False):
            return A
        # Straight-through clamp: the VALUE is clamp(raw, 0, 1) (keeps A in
        # the Birkhoff polytope; raw == 0 gives exactly 0, so A == I at
        # init), the GRADIENT is 1. A plain clamp has zero gradient at the
        # boundary in torch, which pinned alpha at its init 0 forever (T104).
        raw = getattr(self, f"hc_{which}_alpha")
        alpha = raw - (raw - raw.clamp(0.0, 1.0)).detach()
        S = self.sinkhorn(getattr(self, f"hc_{which}_logits"),
                          self.hc_sinkhorn_iters)
        return (1.0 - alpha) * A + alpha * S

    def _hc_step(self, streams, A, B, sublayer, norm, attention_mask,
                 past_key_value, use_cache, position_ids, returns_kv,
                 shared_kv=None, post_norm=None):
        """One HC sublayer: read A, apply f(x) = x + sublayer(norm(x)) per
        branch, write back through B.

        streams: [B, L, n, d]. Branches are folded into the batch dim for
        the sublayer call, so the KV cache (when returns_kv) holds one
        entry per (batch row, branch) — the widened layout is self-
        consistent across prefill/decode because every forward expands the
        same way. Returns (streams_new, present_kv_or_None).
        """
        x = torch.einsum("blnd,mn->blmd", streams, A)  # read: h̃ = A S
        Bb, L, n, d = x.shape
        # Fold branches into the batch dim as (batch, branch) rows. The
        # branch axis must be moved next to the batch axis first: a plain
        # reshape of [B, L, n, d] to [B*n, L, d] interleaves sequence
        # positions with branches, so every branch saw a scrambled
        # sequence (cached decode != full forward as soon as B != 0;
        # fixed in v5.66, found by T104).
        xf = x.permute(0, 2, 1, 3).reshape(Bb * n, L, d)
        mask = attention_mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(Bb, n, -1).reshape(Bb * n, -1)
        pos = position_ids
        if pos is not None:
            pos = pos.unsqueeze(1).expand(Bb, n, L).reshape(Bb * n, L)
        if returns_kv:
            sub_out, present_kv = sublayer(
                norm(xf), attention_mask=mask, past_key_value=past_key_value,
                use_cache=use_cache, position_ids=pos,
                **_shared_kv_kw(shared_kv))
        else:
            sub_out, present_kv = sublayer(norm(xf)), None
        if post_norm is not None:
            sub_out = post_norm(sub_out)        # v5.72 sandwich norm
        f = (xf + sub_out).reshape(Bb, n, L, d).permute(0, 2, 1, 3)  # f(h̃)
        streams_new = x + torch.einsum("blmd,km->blkd", f, B)
        return streams_new, present_kv

    def _forward_hc(self, streams, attention_mask=None, past_key_value=None,
                    use_cache=False, position_ids=None, shared_kv=None):
        """Hyper-Connection forward. streams in/out: [B, L, n, d]."""
        if streams.dim() != 4:
            raise NotImplementedError(
                "Hyper-Connection layers expect the widened [B, L, n, d] "
                "stream; callers that pass a plain [B, L, d] hidden state "
                "(the DualPipe schedule, hand-rolled layer loops) do not "
                "support hyper connections yet"
            )
        attn_mask = attention_mask
        streams, present_kv = self._hc_step(
            streams, self.hc_read_matrix("attn"), self.hc_attn_B, self.attention,
            self.pre_attn_norm, attn_mask, past_key_value, use_cache,
            position_ids, returns_kv=True, shared_kv=shared_kv,
            post_norm=self.post_attn_norm)
        # MoE sublayer: no cache, so the stream step runs without KV args.
        moe_mask = attention_mask
        if moe_mask is not None and moe_mask.dim() == 2:
            # MoE does not consume the mask; pass None so the branch-fold
            # reshape in _hc_step is skipped (keeps the call honest).
            moe_mask = None
        streams, _ = self._hc_step(
            streams, self.hc_read_matrix("moe"), self.hc_moe_B, self.moe,
            self.pre_moe_norm, moe_mask, None, False, position_ids,
            returns_kv=False, post_norm=self.post_moe_norm)
        return streams, present_kv, None

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None, attn_res=None,
                shared_kv=None):
        if self.use_hyper_connections:
            streams, present_kv, _ = self._forward_hc(
                hidden_states, attention_mask=attention_mask,
                past_key_value=past_key_value, use_cache=use_cache,
                position_ids=position_ids, shared_kv=shared_kv)
            return streams, present_kv, None
        if self.res_block:
            return self._forward_block_res(
                hidden_states, attn_res, attention_mask=attention_mask,
                past_key_value=past_key_value, use_cache=use_cache,
                position_ids=position_ids, shared_kv=shared_kv)
        attn_in = self.pre_attn_norm(hidden_states)
        if self.attn_res_gate is not None and attn_res is not None:
            attn_in = attn_in + self.attn_res_gate * attn_res
        attn_out, present_kv = self.attention(
            attn_in,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
            **_shared_kv_kw(shared_kv),
        )
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)   # v5.72 sandwich norm
        # Explicit accumulator threading: the updated accumulator is a
        # return value (part of the autograd graph), not an attribute.
        if self.attn_res_gate is not None:
            attn_res_new = attn_out if attn_res is None else attn_res + attn_out
        else:
            attn_res_new = attn_res  # pass-through (normally None)
        hidden_states = hidden_states + attn_out
        moe_out = self.moe(self.pre_moe_norm(hidden_states))
        if self.post_moe_norm is not None:
            moe_out = self.post_moe_norm(moe_out)      # v5.72 sandwich norm
        hidden_states = hidden_states + moe_out
        return hidden_states, present_kv, attn_res_new

    def _forward_block_res(self, prefix, bank, attention_mask=None,
                           past_key_value=None, use_cache=False,
                           position_ids=None, shared_kv=None):
        """Block attention residuals (v5.142): ``bank`` is the list of
        banked layer inputs, shared by all layers of one forward and
        returned as the third output."""
        cur = block_res_mix(bank, prefix, self.attn_res_score, self.res_eps)
        if self.res_bank:
            bank = bank + [prefix]          # the RAW layer input
        attn_out, present_kv = self.attention(
            self.pre_attn_norm(cur), attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
            position_ids=position_ids, **_shared_kv_kw(shared_kv))
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        prefix = attn_out if self.res_bank else prefix + attn_out
        cur = block_res_mix(bank, prefix, self.ffn_res_score, self.res_eps)
        moe_out = self.moe(self.pre_moe_norm(cur))
        if self.post_moe_norm is not None:
            moe_out = self.post_moe_norm(moe_out)
        return prefix + moe_out, present_kv, bank


def block_res_mix(bank, cur, score, eps):
    """Softmax mix of the banked states and ``cur`` (v5.142).

    Each candidate is scored as ``sum(rms_norm(x) * score)`` (no learned
    norm gain: the gain and the projection fold into ``score``); the mix
    sums the RAW candidates. An empty bank returns ``cur``.
    """
    if not bank:
        return cur
    xs = torch.stack(bank + [cur], dim=-2)                  # [B, T, n+1, d]
    xf = xs.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    probs = (normed * score.float()).sum(-1).softmax(-1)    # [B, T, n+1]
    return (probs[..., None] * xf).sum(-2).to(cur.dtype)


class PerLayerEmbedding(nn.Module):
    """Per-layer embeddings (LB8, Gemma 3n / Gemma 4 E2B).

    inputs(ids, h0) -> [B, L, N, p]: a per-layer token embedding (one
    [vocab, N * p] table, scaled by sqrt(p)) plus a normalised projection
    of the input embedding h0, averaged (* 1/sqrt 2). For layer i,
    ``apply(i, h, x_i)`` = h + post_norm_i(up_i(gelu(gate_i(h)) * x_i)).
    ``up_i`` starts at zero, so at init the residual stream is unchanged.
    Positions without a token (multimodal prefix) get the projection
    term only."""

    def __init__(self, config):
        super().__init__()
        d, n, p = (config.hidden_size, config.num_hidden_layers,
                   config.per_layer_embed_dim)
        self.n, self.p = n, p
        zc = bool(config.zero_centered_norm)
        self.embed = nn.Embedding(config.vocab_size, n * p)
        self.model_proj = nn.Linear(d, n * p, bias=False)
        self.proj_norm = RMSNorm(p, eps=config.rms_norm_eps, zero_centered=zc)
        self.gate = nn.ModuleList(nn.Linear(d, p, bias=False)
                                  for _ in range(n))
        self.up = nn.ModuleList(nn.Linear(p, d, bias=False) for _ in range(n))
        self.post_norm = nn.ModuleList(
            RMSNorm(d, eps=config.rms_norm_eps, zero_centered=zc)
            for _ in range(n))

    def zero_init(self):
        for up in self.up:
            nn.init.zeros_(up.weight)

    def inputs(self, input_ids, h0, prefix_len=0):
        B, L = h0.shape[:2]
        m = self.model_proj(h0).view(B, L, self.n, self.p) * h0.shape[-1] ** -0.5
        m = self.proj_norm(m)
        e = self.embed(input_ids).view(B, -1, self.n, self.p) * self.p ** 0.5
        if prefix_len:
            e = F.pad(e, (0, 0, 0, 0, prefix_len, 0))
        return (m + e.to(m.dtype)) * 2 ** -0.5

    def apply_layer(self, i, h, x):
        g = F.gelu(self.gate[i](h), approximate="tanh") * x.to(h.dtype)
        return h + self.post_norm[i](self.up[i](g))


class EvastarV5(nn.Module):
    """HeliosLM v5.5 model. Sizing is driven by EvastarV5Config(size=...)."""

    def __init__(self, config, size=None):
        super().__init__()
        if size is not None and size != config.size:
            raise ValueError(
                f"size={size!r} conflicts with config.size={config.size!r}; "
                "sizing now lives in EvastarV5Config(size=...)"
            )
        self.config = config
        self.size = config.size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers. IndexShare (B1-1): when
        # config.attention.indexer_share_every > 1, MLA layers (counted in
        # model order, hybrid interleaving included) form groups of
        # ``indexer_share_every``; the group leader builds and REGISTERS
        # the learned indexer, followers score with the leader's indexer
        # via an unregistered reference (MLA._effective_indexer), so
        # parameters and state-dict keys are not duplicated. The default
        # 1 keeps one indexer per MLA layer (bit-identical to before).
        share = int(config.attention.indexer_share_every)
        layers = []
        mla_ordinal = 0
        group_leader_indexer = None
        for i in range(config.num_hidden_layers):
            shared = None
            if _is_full_attention_layer(config, i):
                if share > 1 and mla_ordinal % share != 0:
                    shared = group_leader_indexer
                mla_ordinal += 1
            layer = HeliosLMv5Layer(config, i, shared_indexer=shared)
            if (_is_full_attention_layer(config, i) and share > 1
                    and (mla_ordinal - 1) % share == 0):
                group_leader_indexer = layer.attention.indexer
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=bool(config.zero_centered_norm))
        if (bool(config.use_attention_residuals)
                and config.attention_residual_style == "block"):
            self.output_res_score = nn.Parameter(torch.zeros(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # P0: MTP modules — share the main model's embedding and LM head
        # (weight tying, M24) so an MTP loss trains shared representations.
        if config.mtp.enabled:
            self.mtp_modules = nn.ModuleList([
                MTPModule(config, i,
                          embed_tokens=self.embed_tokens,
                          lm_head=self.lm_head)
                for i in range(config.mtp.num_modules)
            ])
        else:
            self.mtp_modules = None

        # P2: Multimodal encoders (skipped for size="lite")
        if config.multimodal.enabled:
            self.vision_encoder = NaViTEncoder(config)
            self.audio_encoder = StreamingAudioEncoder(config)
            self.vision_proj = nn.Linear(config.multimodal.vision_hidden_size, config.hidden_size)
            self.audio_proj = nn.Linear(config.multimodal.audio_hidden_size, config.hidden_size)
        else:
            self.vision_encoder = None
            self.audio_encoder = None

        self.ple = (PerLayerEmbedding(config)
                    if config.per_layer_embed_dim > 0 else None)

        self.apply(self._init_weights)
        if self.ple is not None:
            self.ple.zero_init()

        # v5.71 U6 (Gemma 3n/4 cross-layer KV sharing): the last
        # ``kv_share_last_n`` MLA layers reuse the K/V of the most recent
        # non-shared MLA layer of the same window type. Done after init so
        # every remaining weight matches the unshared model's init.
        self._kv_share_src = {}
        n_share = int(config.attention.kv_share_last_n or 0)
        if n_share > 0:
            mla_idx = [i for i, l in enumerate(self.layers)
                       if isinstance(l.attention, MLA)]
            if n_share >= len(mla_idx):
                raise ValueError(
                    f"attention.kv_share_last_n={n_share} must be smaller than "
                    f"the number of MLA layers ({len(mla_idx)})")
            owners = mla_idx[:-n_share]
            for j in mla_idx[-n_share:]:
                glob = self.layers[j].attention.sliding_window is None
                src = [i for i in owners
                       if (self.layers[i].attention.sliding_window is None) == glob]
                if not src:
                    raise ValueError(
                        f"attention.kv_share_last_n: shared MLA layer {j} has "
                        "no earlier non-shared MLA layer of the same window "
                        "type to take its KV from")
                self.layers[j].attention.share_kv_from(
                    self.layers[src[-1]].attention)
                self._kv_share_src[j] = src[-1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, images=None,
                audio_features=None):
        """See module docstring for the (logits, hidden_states, past) contract.

        attention_mask: [B, L] with 1 = real token, 0 = pad. When decoding it
        must cover past + current tokens; a mask covering only the current
        tokens is left-padded with ones for the cached prefix.
        """
        B, seq = input_ids.shape
        if seq == 0:
            raise ValueError(
                "input_ids has sequence length 0; forward requires at least "
                "one token (an empty sequence has no defined positions or "
                "causal structure)"
            )
        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    f"past_key_values has {len(past_key_values)} entries, "
                    f"expected {len(self.layers)}"
                )
            # Hybrid models (v5.5): recurrent-state tensors have no sequence
            # axis — take the length from the first dim-2-sequence (MLA)
            # cache tensor via the shared helper (m11). It scans every
            # layer, so a non-MLA layer 0 (first_layer_full=False, v5.57)
            # is fine; config validation guarantees an MLA layer exists.
            past_len = past_seq_len(past_key_values)

        hidden_states = self.embed_tokens(input_ids)
        prefix_len = 0

        # P2: Multimodal prefix tokens. Vision/audio features are prepended to
        # the token sequence, so they occupy absolute positions
        # [past_len, past_len + prefix_len) and text tokens continue from
        # there — positions stay continuous across the modality boundary.
        if images is not None and self.vision_encoder is not None:
            vision_feat = self.vision_proj(self.vision_encoder(images))
            hidden_states = torch.cat([vision_feat, hidden_states], dim=1)
            prefix_len += vision_feat.shape[1]

        if audio_features is not None and self.audio_encoder is not None:
            # StreamingAudioEncoder is a stateful STREAMING encoder: it
            # carries conv tails and attention memory across calls. Prefix
            # encoding here is an independent one-shot operation (a fresh
            # utterance per forward), so reset that carried state first —
            # otherwise a previous forward's stream state would leak in and
            # the same audio_features would encode differently twice.
            self.audio_encoder.reset_state()
            audio_feat = self.audio_proj(self.audio_encoder(audio_features))
            hidden_states = torch.cat([audio_feat, hidden_states], dim=1)
            prefix_len += audio_feat.shape[1]

        total_len = hidden_states.shape[1]
        ple_in = (self.ple.inputs(input_ids, hidden_states, prefix_len)
                  if self.ple is not None else None)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + total_len, device=hidden_states.device
            ).unsqueeze(0).expand(B, total_len)
        elif prefix_len > 0 and position_ids.shape[-1] != total_len:
            # Multimodal prefixes prepend tokens the caller's text-length
            # position_ids cannot cover — without this guard the layers
            # later fail with a cryptic broadcast RuntimeError.
            raise ValueError(
                f"position_ids length {position_ids.shape[-1]} does not "
                f"match the full sequence length {total_len} "
                f"({prefix_len} multimodal prefix tokens + {seq} text "
                "tokens); pass position_ids covering the whole sequence "
                "(prefix included) or None for the default layout"
            )

        if attention_mask is not None and prefix_len > 0:
            # Extend the caller's text mask with ones for the prefix tokens.
            if attention_mask.shape[1] == seq:
                prefix_mask = torch.ones(
                    B, prefix_len, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Transformer layers. Attention residuals (v5.5): attn_res carries
        # the running sum of attention outputs across layers; layer i's
        # attention input is injected with gate_i * attn_res, and each layer
        # returns the updated accumulator explicitly (M4/M5 — no attribute
        # side channel). The accumulator starts at zeros, so layer 0's
        # injection is a no-op.
        #
        # Hyper-Connections (v5.7): instead of a single [B, L, d] stream,
        # the layers operate on n widened virtual branches [B, L, n, d]
        # (initialized as copies of the embedding, read out by mean at the
        # end). attn_res stays unused: the config validator rejects enabling
        # both mechanisms.
        use_hc = bool(self.config.use_hyper_connections)
        present_key_values = [] if use_cache else None
        # v5.71 U6: source layers always build their cache tuple (it is
        # what their KV consumers attend over); consumers get it as
        # shared_kv and return it as their present (aliased storage).
        kv_src = getattr(self, "_kv_share_src", {})
        kv_needed = set(kv_src.values())
        kv_saved = {}
        if use_hc:
            n_br = self.config.hyper_connection_branches
            streams = hidden_states.unsqueeze(2).expand(
                B, total_len, n_br, self.config.hidden_size)
            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                streams, present_kv, _ = layer(
                    streams,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache or i in kv_needed,
                    position_ids=position_ids,
                    shared_kv=kv_saved.get(kv_src.get(i)),
                )
                if i in kv_needed:
                    kv_saved[i] = present_kv
                if use_cache:
                    present_key_values.append(present_kv)
            # Readout: mean over branches, accumulated in float64. The
            # wider accumulator is not optional: at init the n branches are
            # IDENTICAL copies, and summing n copies of x in float32 rounds
            # (n*x needs up to 2 extra mantissa bits), which would break
            # the exact identity-at-init property; in float64 n*x and the
            # division by n are both exact, so the copy is restored
            # bit-for-bit. (This also sidesteps .mean's reciprocal
            # multiply, inexact for non-power-of-two n.)
            hidden_states = (streams.sum(dim=2, dtype=torch.float64) / n_br).to(
                streams.dtype)
        else:
            use_attn_res = bool(self.config.use_attention_residuals)
            res_block = use_attn_res and self.config.attention_residual_style == "block"
            if res_block:
                attn_res = []                  # the bank of block residuals
            else:
                attn_res = torch.zeros_like(hidden_states) if use_attn_res else None
            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                hidden_states, present_kv, attn_res = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache or i in kv_needed,
                    position_ids=position_ids,
                    attn_res=attn_res,
                    shared_kv=kv_saved.get(kv_src.get(i)),
                )
                if ple_in is not None:
                    hidden_states = self.ple.apply_layer(
                        i, hidden_states, ple_in[:, :, i])
                if i in kv_needed:
                    kv_saved[i] = present_kv
                if use_cache:
                    present_key_values.append(present_kv)
            if res_block:
                hidden_states = block_res_mix(
                    attn_res, hidden_states, self.output_res_score,
                    self.config.rms_norm_eps)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # v5.9: Gemma-2 style final logit soft-capping (None = uncapped,
        # v5.8 behaviour). Bounds every logit to (-cap, cap); generate()
        # inherits it because it consumes these logits.
        cap = self.config.final_logit_soft_cap
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)

        return logits, hidden_states, present_key_values

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9,
                 attention_mask=None, use_mtp=False, thinking_budget=None):
        """Batch-safe generation.

        - temperature == 0 -> greedy argmax; temperature > 0 -> sampling
          (with optional top-p nucleus filtering).
        - Rows that emit EOS are frozen (subsequent positions are filled with
          pad_token_id); the loop stops when every row is finished.
        - Multimodal limitation: this is a text-only path — it does NOT take
          ``images`` / ``audio_features``. To generate conditioned on
          multimodal inputs, run :meth:`forward` once with those arguments to
          obtain ``past_key_values`` covering the modality prefix, then
          continue with your own decode loop (the per-layer cache tuples are
          directly usable).
        - thinking_budget (v5.73, roadmap U2): an optional
          ``inference.thinking_budget.ThinkingBudget``; each row's thinking
          span is forced closed after ``budget`` tokens. None (default) =
          unchanged decoding. Plain path only.
        """
        if use_mtp and thinking_budget is not None:
            raise ValueError("thinking_budget is not supported with use_mtp=True")
        if use_mtp:
            if self.mtp_modules is None:
                raise ValueError(
                    "use_mtp=True but config.mtp.enabled=False: no MTP modules"
                )
            # MTPDecoder supports batched generation (v5.2): rows are
            # grouped by cache length and verified with batched main-model
            # forwards; early-stopped rows are right-padded with
            # pad_token_id, matching this method's freezing convention.
            decoder = MTPDecoder(self, self.mtp_modules, self.config)
            # Pass top_p and attention_mask through when the installed
            # MTPDecoder supports them (F5: silently dropping top_p changes
            # the sampling distribution; the mask keeps pad tokens out of
            # the MTP path exactly like the plain path below).
            mtp_params = inspect.signature(MTPDecoder.generate).parameters
            if "top_p" in mtp_params:
                result = decoder.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p,
                    attention_mask=attention_mask,
                )
            else:
                if top_p is not None and temperature is not None and temperature > 0:
                    warnings.warn(
                        "MTPDecoder.generate does not accept top_p; the "
                        "nucleus filter is not applied on the MTP path",
                        stacklevel=2,
                    )
                result = decoder.generate(
                    input_ids, max_new_tokens, temperature,
                )
            return result.sequences  # MTPGenerateResult -> token tensor

        # Snapshot the caller's training mode and restore it on every exit
        # path (including exceptions): generate must not leave the model in
        # eval mode as a side effect.
        was_training = self.training
        self.eval()
        try:
            return self._generate_impl(
                input_ids, max_new_tokens, temperature, top_p, attention_mask,
                thinking_budget,
            )
        finally:
            self.train(was_training)

    def _generate_impl(self, input_ids, max_new_tokens, temperature, top_p,
                       attention_mask, thinking_budget=None):
        device = input_ids.device
        B = input_ids.shape[0]
        generated = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values = None
        if thinking_budget is not None:
            thinking_budget.start(input_ids)

        for _ in range(max_new_tokens):
            if past_key_values is None:
                logits, _, past_key_values = self.forward(
                    generated, attention_mask=attention_mask, use_cache=True
                )
            else:
                logits, _, past_key_values = self.forward(
                    generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            next_logits = logits[:, -1, :].float()
            if thinking_budget is not None:
                # Masks with -inf before temperature / top-p, so a forced
                # close survives sampling (it is the only finite logit).
                next_logits = thinking_budget.process(next_logits)

            if temperature is not None and temperature > 0:
                next_logits = next_logits / temperature
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            else:
                # Greedy decoding (temperature == 0)
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Freeze finished rows: keep appending pad so shapes stay aligned.
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            if thinking_budget is not None:
                thinking_budget.update(next_token, active=~finished)
            finished = finished | (next_token.squeeze(1) == self.config.eos_token_id)
            if bool(finished.all()):
                break

        return generated


# Pre-v5.134 name, kept so existing imports keep working.
HeliosLMv5 = EvastarV5
