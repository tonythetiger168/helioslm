"""HeliosLM v5.4 - Unified Model

Model core: token embedding + N x (pre-norm MLA + pre-norm Sigmoid MoE) +
final RMSNorm + LM head, with optional MTP modules and multimodal encoders.

Forward contract (relied upon by MTP / GRPO / training modules):
    logits, hidden_states, past_key_values = model(
        input_ids, attention_mask=None, position_ids=None,
        past_key_values=None, use_cache=False,
        images=None, audio_features=None,
    )
  - logits: [B, L, vocab]
  - hidden_states: [B, L, hidden] after the final norm
  - past_key_values: per-layer cache tuples, or None. Layout follows
    config.attention.use_absorption: (c_kv, k_rope) when absorbed
    (default), (k_nope, k_rope, v) otherwise; dim 2 is the sequence
    length for every tensor in both layouts.

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
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
from helioslm_v5.src.inference.mtp import MTPModule, MTPDecoder
from helioslm_v5.src.vision.navit import NaViTEncoder
from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder


class HeliosLMv5Layer(nn.Module):
    """Single transformer layer: pre-norm MLA + pre-norm Sigmoid MoE.

    Each sublayer input is normalized exactly once:
      h = h + attn(pre_attn_norm(h))
      h = h + moe(pre_moe_norm(h))
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # kept for external cache plumbing/debugging
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MLA(config)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = DeviceLimitedMoE(config)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                use_cache=False, position_ids=None):
        attn_out, present_kv = self.attention(
            self.pre_attn_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.moe(self.pre_moe_norm(hidden_states))
        return hidden_states, present_kv


class HeliosLMv5(nn.Module):
    """HeliosLM v5.4 model. Sizing is driven by HeliosLMv5Config(size=...)."""

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
            past_len = past_key_values[0][0].shape[2]

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

        if attention_mask is not None and prefix_len > 0:
            # Extend the caller's text mask with ones for the prefix tokens.
            if attention_mask.shape[1] == seq:
                prefix_mask = torch.ones(
                    B, prefix_len, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Transformer layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                position_ids=position_ids,
            )
            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

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
            # Pass top_p through when the installed MTPDecoder supports it
            # (F5: silently dropping it changes the sampling distribution).
            mtp_params = inspect.signature(MTPDecoder.generate).parameters
            if "top_p" in mtp_params:
                result = decoder.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p,
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
