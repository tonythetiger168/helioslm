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
# historical public name ``helioslm_v5.src.model_v5.RMSNorm`` keeps working.
from helioslm_v5.src.attention.mla import MLA, RMSNorm
from helioslm_v5.src.attention.linear_attention import (
    GatedDeltaAttention, past_seq_len)
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
from helioslm_v5.src.inference.mtp import MTPModule, MTPDecoder
from helioslm_v5.src.vision.navit import NaViTEncoder
from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder


def _is_full_attention_layer(config, layer_idx):
    """Hybrid interleave rule (v5.5): MLA iff layer 0 or
    ``idx % every == every - 1``; GatedDeltaAttention otherwise."""
    hcfg = getattr(config, "hybrid_attention", None)
    if hcfg is None or not getattr(hcfg, "enabled", False):
        return True
    if layer_idx == 0:
        return True
    every = hcfg.full_attention_every
    return layer_idx % every == every - 1


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

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # kept for external cache plumbing/debugging
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = (MLA(config) if _is_full_attention_layer(config, layer_idx)
                          else GatedDeltaAttention(config))
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = DeviceLimitedMoE(config)
        if bool(getattr(config, "use_attention_residuals", False)):
            self.attn_res_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.attn_res_gate = None

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
            getattr(config, "use_hyper_connections", False))
        if self.use_hyper_connections:
            n = config.hyper_connection_branches
            self.hc_num_branches = n
            self.register_buffer("hc_attn_A", torch.eye(n))
            self.hc_attn_B = nn.Parameter(torch.zeros(n, n))
            self.register_buffer("hc_moe_A", torch.eye(n))
            self.hc_moe_B = nn.Parameter(torch.zeros(n, n))

    # ------------------------------------------------------------------
    # Hyper-Connection stream step (v5.7)
    # ------------------------------------------------------------------
    def _hc_step(self, streams, A, B, sublayer, norm, attention_mask,
                 past_key_value, use_cache, position_ids, returns_kv):
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
        xf = x.reshape(Bb * n, L, d)
        mask = attention_mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(Bb, n, -1).reshape(Bb * n, -1)
        pos = position_ids
        if pos is not None:
            pos = pos.unsqueeze(1).expand(Bb, n, L).reshape(Bb * n, L)
        if returns_kv:
            sub_out, present_kv = sublayer(
                norm(xf), attention_mask=mask, past_key_value=past_key_value,
                use_cache=use_cache, position_ids=pos)
        else:
            sub_out, present_kv = sublayer(norm(xf)), None
        f = (xf + sub_out).reshape(Bb, L, n, d)  # f(h̃) = h̃ + F(h̃)
        streams_new = x + torch.einsum("blmd,km->blkd", f, B)
        return streams_new, present_kv

    def _forward_hc(self, streams, attention_mask=None, past_key_value=None,
                    use_cache=False, position_ids=None):
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
            streams, self.hc_attn_A, self.hc_attn_B, self.attention,
            self.pre_attn_norm, attn_mask, past_key_value, use_cache,
            position_ids, returns_kv=True)
        # MoE sublayer: no cache, so the stream step runs without KV args.
        moe_mask = attention_mask
        if moe_mask is not None and moe_mask.dim() == 2:
            # MoE does not consume the mask; pass None so the branch-fold
            # reshape in _hc_step is skipped (keeps the call honest).
            moe_mask = None
        streams, _ = self._hc_step(
            streams, self.hc_moe_A, self.hc_moe_B, self.moe,
            self.pre_moe_norm, moe_mask, None, False, position_ids,
            returns_kv=False)
        return streams, present_kv, None

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None, attn_res=None):
        if self.use_hyper_connections:
            streams, present_kv, _ = self._forward_hc(
                hidden_states, attention_mask=attention_mask,
                past_key_value=past_key_value, use_cache=use_cache,
                position_ids=position_ids)
            return streams, present_kv, None
        attn_in = self.pre_attn_norm(hidden_states)
        if self.attn_res_gate is not None and attn_res is not None:
            attn_in = attn_in + self.attn_res_gate * attn_res
        attn_out, present_kv = self.attention(
            attn_in,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        # Explicit accumulator threading: the updated accumulator is a
        # return value (part of the autograd graph), not an attribute.
        if self.attn_res_gate is not None:
            attn_res_new = attn_out if attn_res is None else attn_res + attn_out
        else:
            attn_res_new = attn_res  # pass-through (normally None)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.moe(self.pre_moe_norm(hidden_states))
        return hidden_states, present_kv, attn_res_new


class HeliosLMv5(nn.Module):
    """HeliosLM v5.5 model. Sizing is driven by HeliosLMv5Config(size=...)."""

    def __init__(self, config, size=None):
        super().__init__()
        if size is not None and size != config.size:
            raise ValueError(
                f"size={size!r} conflicts with config.size={config.size!r}; "
                "sizing now lives in HeliosLMv5Config(size=...)"
            )
        self.config = config
        self.size = config.size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            HeliosLMv5Layer(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        self.apply(self._init_weights)

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
            # cache tensor via the shared helper (m11; layer 0 is always
            # MLA under the interleave rule).
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
        use_hc = bool(getattr(self.config, "use_hyper_connections", False))
        present_key_values = [] if use_cache else None
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
                    use_cache=use_cache,
                    position_ids=position_ids,
                )
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
            use_attn_res = bool(getattr(self.config, "use_attention_residuals", False))
            attn_res = torch.zeros_like(hidden_states) if use_attn_res else None
            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                hidden_states, present_kv, attn_res = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    position_ids=position_ids,
                    attn_res=attn_res,
                )
                if use_cache:
                    present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # v5.9: Gemma-2 style final logit soft-capping (None = uncapped,
        # v5.8 behaviour). Bounds every logit to (-cap, cap); generate()
        # inherits it because it consumes these logits.
        cap = getattr(self.config, "final_logit_soft_cap", None)
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)

        return logits, hidden_states, present_key_values

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_p=0.9,
                 attention_mask=None, use_mtp=False):
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
        """
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
                input_ids, max_new_tokens, temperature, top_p, attention_mask
            )
        finally:
            self.train(was_training)

    def _generate_impl(self, input_ids, max_new_tokens, temperature, top_p,
                       attention_mask):
        device = input_ids.device
        B = input_ids.shape[0]
        generated = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values = None

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
            finished = finished | (next_token.squeeze(1) == self.config.eos_token_id)
            if bool(finished.all()):
                break

        return generated
