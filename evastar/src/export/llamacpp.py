"""Export HeliosLM checkpoints as llama.cpp ``deepseek2`` GGUF files.

v5.141: hybrid KDA + MLA configs (``hybrid_attention.style="k3"``, nope
MLA) export to llama.cpp's ``kimi-k3`` architecture instead; see
:func:`llamacpp_arch` and :func:`to_kimi_k3_tensors` at the end of this
module.

``gguf.export_gguf`` writes a self-describing ``helioslm`` container that no
GGML executor runs. This module maps the same weights onto llama.cpp's
existing ``deepseek2`` architecture (MLA + sigmoid-routed MoE with shared
experts and an aux-free selection bias), so llama.cpp, and the runtimes
built on it such as Ollama and LM Studio, can load the file with no
custom code. The mapping follows llama.cpp's own DeepSeek converter
(``conversion/deepseek.py``) and loader (``src/models/deepseek2.cpp``):

  - ``attn_kv_a_mqa`` = rows of ``kv_a_proj`` then ``k_rope_proj``:
    llama.cpp computes the latent and the shared RoPE key in one matmul.
  - ``attn_k_b`` / ``attn_v_b``: ``kv_b_proj`` split per head into the
    K_nope and V halves, K transposed (llama.cpp's absorbed MLA path).
    Metadata declares the MQA view (key_length = kv_lora_rank + rope,
    value_length = kv_lora_rank, one KV head) plus the real per-head sizes
    in ``key_length_mla`` / ``value_length_mla``.
  - **RoPE layout.** HeliosLM rotates dim i with dim i + d/2 (``rotate_half``,
    GPT-NeoX order); deepseek2 is a NORM-rope architecture that rotates
    adjacent pairs (2i, 2i+1). The RoPE rows of ``q_b_proj`` and
    ``k_rope_proj`` are permuted to interleaved order. q·k is unchanged
    because both sides get the same permutation.
  - Routing: ``expert_gating_func`` = sigmoid, ``route_bias`` becomes
    ``exp_probs_b`` (selection only, as here), no top-k renormalisation
    and no routed scaling, matching ``DeviceLimitedMoE`` (gates are the
    raw sigmoid scores of the selected experts).
  - Experts: each ``w13`` splits into gate (first half, SiLU side) and up;
    routed experts stack into ``ffn_{gate,up,down}_exps``; shared experts
    concatenate along the FFN width (a sum of SwiGLUs is one wider SwiGLU).
  - ``zero_centered_norm`` gains (1 + w) are folded into plain RMSNorm
    weights.
  - MTP modules are dropped: llama.cpp's deepseek2 does not run HeliosLM's
    MTP block format, and plain decoding does not need it.

Anything the deepseek2 graph does not compute is refused, not
approximated: :func:`llamacpp_incompatibilities` lists the config
features (KDA hybrid layers, NoPE, LatentMoE, ...), and the exporter also
refuses any trainable parameter it does not map, so a new module cannot
be dropped silently.

Tokenizer: without one the file carries no vocabulary
(``tokenizer.ggml.model = "none"``) and llama.cpp runs on token ids,
which is what the equivalence checks use. Passing the pretraining
``tokenizer.json`` (a Hugging Face byte-level BPE, as
``examples/pilot_prepare_data.py`` trains it) writes it as llama.cpp's
``gpt2`` vocabulary with the ``gpt-2`` pre-tokenizer: the same GPT-2
split regex, byte map and merge ranks, so text in gives the same ids
(:func:`gpt2_tokenizer_metadata`). Only that exact shape is accepted;
anything else (a normalizer, another pre-tokenizer, byte fallback) is
refused rather than approximated. HeliosBPE pre-tokenizes on spaces
only, which no llama.cpp pre-tokenizer reproduces, so it stays id-only.
"""
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .gguf import (
    _VAL_ARRAY, _VAL_BOOL, _VAL_FLOAT32, _VAL_INT32, _VAL_STRING, _VAL_UINT32,
    DEFAULT_ALIGNMENT,
    _FILE_TYPE_ENUM, write_gguf)

ARCHITECTURE = "deepseek2"
GATING_SIGMOID = 2      # llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
ROPE_FREQ_BASE = 10000.0  # RotaryEmbedding default; HeliosLM has no rope_theta field
# The absorbed-MLA matmuls take a permuted (non-contiguous) activation. With
# these two weights in F16, llama.cpp's CPU weight repacking (on by default)
# either has no kernel for that op (abort at graph build, DeepSeek-shaped
# heads) or returned wrong logits (cosine 0.03 on the lite shapes); in F32
# they run correctly everywhere tested. They are about 1-2% of the weights.
KEEP_F32 = ("attn_k_b.weight", "attn_v_b.weight")


def _get(obj, path, default=None):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


# (config path, offending-value test, reason). Each entry is a feature the
# deepseek2 graph in llama.cpp does not compute.
_CHECKS = (
    ("hybrid_attention.enabled", bool, "KDA / gated-delta linear attention layers"),
    ("attention.nope", bool, "NoPE (deepseek2 always applies RoPE)"),
    ("attention.nope_attn_temperature", bool, "NoPE attention temperature"),
    ("attention.rope_scaling", lambda v: v is not None, "RoPE scaling (YaRN/NTK not mapped yet)"),
    ("attention.global_rope_partial", lambda v: float(v) != 1.0, "partial RoPE on global layers (p-RoPE)"),
    ("attention.global_k_eq_v", bool, "K = V on global layers"),
    ("attention.kv_share_last_n", bool, "cross-layer KV sharing"),
    ("attention.output_gate", bool, "attention output gate"),
    ("attention.sparse_top_k", lambda v: v is not None, "DSA sparse attention"),
    ("attention.logit_soft_cap", lambda v: v is not None, "attention logit soft-capping"),
    ("attention.qk_norm", bool, "per-head QK-norm"),
    ("attention.sliding_window", lambda v: v is not None, "sliding-window attention"),
    ("attention.global_attention_every", lambda v: v is not None, "local/global attention mix"),
    ("attention.learnable_sink", bool, "learnable attention sink"),
    ("attention.q_lora_rank", lambda v: not v,
     "no Q low-rank path (llama.cpp reads q_lora_rank except on DeepSeek-V2-Lite shapes)"),
    ("moe.latent_dim", lambda v: v is not None, "LatentMoE down/up projections"),
    ("moe.activation", lambda v: v != "swiglu", "expert activation other than SwiGLU"),
    ("moe.swiglu_limit", lambda v: v is not None, "clamped SwiGLU"),
    ("moe.swiglu_alpha", lambda v: float(v) != 1.0, "SwiGLU alpha != 1"),
    ("moe.swiglu_up_offset", lambda v: float(v) != 0.0, "SwiGLU up offset"),
    ("moe.num_shared_experts", lambda v: int(v) < 1, "no shared expert (deepseek2 requires one)"),
    ("use_attention_residuals", bool, "attention residuals"),
    ("use_hyper_connections", bool, "hyper-connections"),
    ("final_logit_soft_cap", lambda v: v is not None, "final logit soft-capping"),
    ("sandwich_norm", bool, "sandwich (post-sublayer) norm"),
    ("gated_conv_layers", lambda v: v is not None, "gated short-conv layers (LB9)"),
    ("per_layer_embed_dim", bool, "per-layer embeddings (LB8)"),
    ("multimodal.enabled", bool, "vision / audio encoders"),
)


def llamacpp_incompatibilities(config):
    """Reasons ``config`` cannot run as llama.cpp deepseek2 ([] = exportable)."""
    out = []
    for path, bad, reason in _CHECKS:
        value = _get(config, path)
        if value is not None and bad(value):
            out.append(f"{path}={value!r}: {reason}")
    return out


def _interleave_rope_rows(w, rope_dim):
    """Reorder the last ``rope_dim`` rows of ``w`` from NeoX to adjacent pairs."""
    half = rope_dim // 2
    perm = torch.stack([torch.arange(half), torch.arange(half) + half], 1).reshape(-1)
    head, rope = w[:-rope_dim], w[-rope_dim:]
    return torch.cat([head, rope[perm]], 0)


def _norm_weight(w, zero_centered):
    w = w.detach().float()
    return 1.0 + w if zero_centered else w


def _split_w13(w13):
    gate, up = w13.detach().float().chunk(2, dim=0)
    return gate, up


def to_deepseek2_tensors(model, config):
    """Map a EvastarV5 model's weights to llama.cpp deepseek2 tensors.

    Returns an ordered list of ``(gguf_name, float32 tensor)``. Raises
    ValueError on an incompatible config or an unmapped parameter.
    """
    problems = llamacpp_incompatibilities(config)
    if problems:
        raise ValueError("config cannot run as llama.cpp deepseek2:\n  - "
                         + "\n  - ".join(problems))
    sd = model.state_dict() if not isinstance(model, Mapping) else dict(model)
    params = ({n for n, _ in model.named_parameters()}
              if hasattr(model, "named_parameters") else set(sd))
    used = set()

    def take(name):
        if name not in sd:
            raise ValueError(f"expected tensor {name!r} is missing")
        used.add(name)
        return sd[name].detach().float()

    a = config.attention
    H = a.num_attention_heads
    nope, rope, v_dim = a.no_rope_head_dim, a.rope_head_dim, a.v_head_dim
    zc = bool(getattr(config, "zero_centered_norm", False))
    n_dense = int(config.moe.first_k_dense or 0)
    out = [("token_embd.weight", take("embed_tokens.weight"))]

    for i in range(config.num_hidden_layers):
        p, b = f"layers.{i}.", f"blk.{i}."
        out.append((b + "attn_norm.weight", _norm_weight(take(p + "pre_attn_norm.weight"), zc)))
        out.append((b + "attn_q_a.weight", take(p + "attention.q_a_proj.weight")))
        out.append((b + "attn_q_a_norm.weight",
                    _norm_weight(take(p + "attention.norm_q.weight"), zc)))
        q = take(p + "attention.q_b_proj.weight").view(H, nope + rope, -1)
        q = torch.stack([_interleave_rope_rows(q[h], rope) for h in range(H)])
        out.append((b + "attn_q_b.weight", q.reshape(H * (nope + rope), -1)))
        kv_a = torch.cat([take(p + "attention.kv_a_proj.weight"),
                          _interleave_rope_rows(take(p + "attention.k_rope_proj.weight"), rope)], 0)
        out.append((b + "attn_kv_a_mqa.weight", kv_a))
        out.append((b + "attn_kv_a_norm.weight",
                    _norm_weight(take(p + "attention.norm_kv.weight"), zc)))
        kv_b = take(p + "attention.kv_b_proj.weight").view(H, nope + v_dim, -1)
        k_b, v_b = kv_b[:, :nope, :], kv_b[:, nope:, :]
        out.append((b + "attn_k_b.weight", k_b.transpose(1, 2).contiguous()))
        out.append((b + "attn_v_b.weight", v_b.contiguous()))
        out.append((b + "attn_output.weight", take(p + "attention.o_proj.weight")))
        out.append((b + "ffn_norm.weight", _norm_weight(take(p + "pre_moe_norm.weight"), zc)))

        if i < n_dense:
            gate, up = _split_w13(take(p + "moe.w13.weight"))
            out += [(b + "ffn_gate.weight", gate), (b + "ffn_up.weight", up),
                    (b + "ffn_down.weight", take(p + "moe.w2.weight"))]
            continue
        out.append((b + "ffn_gate_inp.weight", take(p + "moe.router.weight")))
        out.append((b + "exp_probs_b.bias", take(p + "moe.route_bias")))
        gates, ups, downs = [], [], []
        for e in range(config.moe.num_experts):
            gate, up = _split_w13(take(p + f"moe.experts.{e}.w13.weight"))
            gates.append(gate)
            ups.append(up)
            downs.append(take(p + f"moe.experts.{e}.w2.weight"))
        out += [(b + "ffn_gate_exps.weight", torch.stack(gates)),
                (b + "ffn_up_exps.weight", torch.stack(ups)),
                (b + "ffn_down_exps.weight", torch.stack(downs))]
        sg, su, sdn = [], [], []
        for s in range(config.moe.num_shared_experts):
            gate, up = _split_w13(take(p + f"moe.shared_experts.{s}.w13.weight"))
            sg.append(gate)
            su.append(up)
            sdn.append(take(p + f"moe.shared_experts.{s}.w2.weight"))
        out += [(b + "ffn_gate_shexp.weight", torch.cat(sg, 0)),
                (b + "ffn_up_shexp.weight", torch.cat(su, 0)),
                (b + "ffn_down_shexp.weight", torch.cat(sdn, 1))]

    out.append(("output_norm.weight", _norm_weight(take("norm.weight"), zc)))
    out.append(("output.weight", take("lm_head.weight")))

    unmapped = sorted(n for n in params - used if not n.startswith("mtp_modules."))
    if unmapped:
        raise ValueError(
            "parameters with no deepseek2 counterpart (refusing to drop them): "
            + ", ".join(unmapped[:8]) + (" ..." if len(unmapped) > 8 else ""))
    return out


def deepseek2_metadata(config, *, dtype="f16", name=None, tokenizer=None):
    """Typed GGUF keys for llama.cpp's deepseek2 loader (``tokenizer``: see
    :func:`gpt2_tokenizer_metadata`; None writes an id-only file)."""
    a, m = config.attention, config.moe
    arch = ARCHITECTURE
    kv = []

    def u32(key, value):
        kv.append((key, _VAL_UINT32, int(value)))

    def f32(key, value):
        kv.append((key, _VAL_FLOAT32, float(value)))

    kv.append(("general.architecture", _VAL_STRING, arch))
    u32("general.alignment", DEFAULT_ALIGNMENT)
    kv.append(("general.name", _VAL_STRING, name or getattr(config, "model_name", "HeliosLM")))
    kv.append(("general.license", _VAL_STRING, "Apache-2.0"))
    u32("general.file_type", _FILE_TYPE_ENUM[dtype])
    u32(f"{arch}.context_length", config.max_position_embeddings)
    u32(f"{arch}.embedding_length", config.hidden_size)
    u32(f"{arch}.block_count", config.num_hidden_layers)
    u32(f"{arch}.feed_forward_length", config.intermediate_size)
    u32(f"{arch}.vocab_size", config.vocab_size)
    u32(f"{arch}.attention.head_count", a.num_attention_heads)
    u32(f"{arch}.attention.head_count_kv", 1)          # MLA as MQA
    u32(f"{arch}.attention.key_length", a.kv_latent_dim + a.rope_head_dim)
    u32(f"{arch}.attention.value_length", a.kv_latent_dim)
    u32(f"{arch}.attention.key_length_mla", a.no_rope_head_dim + a.rope_head_dim)
    u32(f"{arch}.attention.value_length_mla", a.v_head_dim)
    u32(f"{arch}.attention.q_lora_rank", a.q_lora_rank)
    u32(f"{arch}.attention.kv_lora_rank", a.kv_latent_dim)
    f32(f"{arch}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    f32(f"{arch}.rope.freq_base", ROPE_FREQ_BASE)
    u32(f"{arch}.rope.dimension_count", a.rope_head_dim)
    u32(f"{arch}.leading_dense_block_count", int(m.first_k_dense or 0))
    u32(f"{arch}.expert_count", m.num_experts)
    u32(f"{arch}.expert_used_count", m.num_activated_experts)
    u32(f"{arch}.expert_shared_count", m.num_shared_experts)
    u32(f"{arch}.expert_feed_forward_length", m.expert_hidden_size)
    f32(f"{arch}.expert_weights_scale", 1.0)
    kv.append((f"{arch}.expert_weights_norm", _VAL_BOOL, False))
    u32(f"{arch}.expert_gating_func", GATING_SIGMOID)
    if tokenizer is not None:
        kv.extend(gpt2_tokenizer_metadata(tokenizer, config.vocab_size))
        return kv
    kv.append(("tokenizer.ggml.model", _VAL_STRING, "none"))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, key, None)
        if value is not None:
            u32(f"tokenizer.ggml.{key}", value)
    return kv


# llama.cpp llama_token_type
TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_CONTROL, TOKEN_USER_DEFINED, TOKEN_UNUSED = 1, 2, 3, 4, 5
EOS_NAMES = ("<|endoftext|>", "<eos>", "</s>")
UNK_NAMES = ("[UNK]", "<unk>")


def _tokenizer_json(tokenizer):
    if isinstance(tokenizer, Mapping):
        return tokenizer
    if isinstance(tokenizer, (str, Path)):
        return json.loads(Path(tokenizer).read_text(encoding="utf-8"))
    return json.loads(tokenizer.to_str())            # tokenizers.Tokenizer


def gpt2_tokenizer_metadata(tokenizer, vocab_size):
    """``tokenizer.ggml.*`` keys for a byte-level BPE ``tokenizer.json``.

    ``tokenizer`` is a ``tokenizers.Tokenizer``, a path to its JSON, or the
    parsed JSON. Ids above the tokenizer's size up to ``vocab_size`` (the
    embedding rows) become unused ``[PAD{id}]`` entries, as llama.cpp's own
    converter writes them. Raises ``ValueError`` for any tokenizer llama.cpp
    would split or merge differently."""
    tj = _tokenizer_json(tokenizer)
    model, pre = tj.get("model") or {}, tj.get("pre_tokenizer") or {}
    problems = []
    if model.get("type") != "BPE":
        problems.append(f"model type {model.get('type')!r} (need BPE)")
    for key in ("byte_fallback", "ignore_merges", "dropout",
                "continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(key):
            problems.append(f"model.{key}={model[key]!r}")
    if tj.get("normalizer"):
        problems.append(f"normalizer {tj['normalizer'].get('type')!r}")
    if pre.get("type") != "ByteLevel" or pre.get("add_prefix_space") \
            or not pre.get("use_regex", True):
        problems.append(f"pre_tokenizer {pre!r} (need ByteLevel, regex on, "
                        "no prefix space)")
    post = tj.get("post_processor")
    if post and post.get("type") != "ByteLevel":
        problems.append(f"post_processor {post.get('type')!r} adds tokens")
    if problems:
        raise ValueError("tokenizer cannot run as llama.cpp gpt2: "
                         + "; ".join(problems))

    tokens = {i: t for t, i in model["vocab"].items()}
    types = {i: TOKEN_NORMAL for i in tokens}
    for added in tj.get("added_tokens") or []:
        tokens[added["id"]] = added["content"]
        types[added["id"]] = TOKEN_CONTROL if added.get("special") else TOKEN_USER_DEFINED
    n = max(tokens) + 1
    if n > vocab_size:
        raise ValueError(f"tokenizer has {n} ids but the model only "
                         f"{vocab_size} embedding rows")
    names = [tokens.get(i, f"[PAD{i}]") for i in range(vocab_size)]
    kinds = [types.get(i, TOKEN_UNUSED) for i in range(vocab_size)]
    by_name = {t: i for i, t in tokens.items()}
    for name in UNK_NAMES:
        if name in by_name:
            kinds[by_name[name]] = TOKEN_UNKNOWN
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]

    kv = [
        ("tokenizer.ggml.model", _VAL_STRING, "gpt2"),
        ("tokenizer.ggml.pre", _VAL_STRING, "gpt-2"),
        ("tokenizer.ggml.tokens", _VAL_ARRAY, (_VAL_STRING, names)),
        ("tokenizer.ggml.token_type", _VAL_ARRAY, (_VAL_INT32, kinds)),
        ("tokenizer.ggml.merges", _VAL_ARRAY, (_VAL_STRING, merges)),
        ("tokenizer.ggml.add_bos_token", _VAL_BOOL, False),
        ("tokenizer.ggml.add_eos_token", _VAL_BOOL, False),
    ]
    eos = next((by_name[t] for t in EOS_NAMES if t in by_name), None)
    if eos is not None:   # GPT-2 convention: one <|endoftext|> for both
        kv += [("tokenizer.ggml.eos_token_id", _VAL_UINT32, eos),
               ("tokenizer.ggml.bos_token_id", _VAL_UINT32, eos)]
    unk = next((by_name[t] for t in UNK_NAMES if t in by_name), None)
    if unk is not None:
        kv.append(("tokenizer.ggml.unknown_token_id", _VAL_UINT32, unk))
    return kv


def export_llamacpp_gguf(model, path, config, *, dtype="f16", name=None,
                         tokenizer=None):
    """Write ``model`` as a llama.cpp-loadable deepseek2 GGUF file.

    ``dtype`` is ``"f32"``, ``"f16"`` (default), ``"q8_0"`` or ``"q4_k"``
    (see :func:`gguf.export_gguf`). 1-D tensors (norms, ``exp_probs_b``)
    and the router ``ffn_gate_inp`` stay F32 in every dtype, as llama.cpp's
    converter writes them, and so do ``attn_k_b`` / ``attn_v_b`` (see
    ``KEEP_F32``). ``tokenizer`` (a byte-level BPE ``tokenizer.json``)
    embeds the vocabulary so llama.cpp / Ollama take text; without it the
    file takes token ids. Returns the path written.
    """
    if llamacpp_arch(config) == KIMI_K3:
        tensors, kda = to_kimi_k3_tensors(model, config)
        kv = kimi_k3_metadata(config, kda, dtype=dtype, name=name, tokenizer=tokenizer)
        keep = KEEP_F32 + K3_KEEP_F32
    else:
        tensors = to_deepseek2_tensors(model, config)
        kv = deepseek2_metadata(config, dtype=dtype, name=name, tokenizer=tokenizer)
        keep = KEEP_F32
    return write_gguf(path, kv, tensors, dtype, keep_1d_f32=True, keep_f32=keep)


# ---------------------------------------------------------------------------
# kimi-k3: hybrid KDA + MLA models (v5.142)
# ---------------------------------------------------------------------------

KIMI_K3 = "kimi-k3"
# llama.cpp's kimi-k3 graph only has the situ activation,
# beta * tanh(g / beta) * sigmoid(g) * up (linear_beta <= 0 leaves up as is).
# With a huge beta, tanh(g / beta) * beta is g to float rounding, so the
# activation is SwiGLU's silu(g) * up.
SWIGLU_SITU_BETA = 1e30
# the short-conv taps and the low-rank decay gate stay F32 (small, and the
# decay is exponentiated)
K3_KEEP_F32 = ("ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
               "ssm_f_a.weight", "ssm_f_b.weight")

# features the deepseek2 list refuses that the kimi-k3 graph computes
_K3_OK = {"hybrid_attention.enabled", "attention.nope", "attention.output_gate",
          "moe.latent_dim", "moe.activation", "use_attention_residuals",
          "multimodal.enabled"}
# text only, as llama.cpp's own Kimi-K3 converter: the vision / audio
# encoders and their projections are left out of the GGUF file
_K3_TEXT_ONLY = ("mtp_modules.", "vision_encoder.", "vision_proj.",
                 "audio_encoder.", "audio_proj.")


def kimi_k3_incompatibilities(config):
    """Reasons ``config`` cannot run as llama.cpp kimi-k3 ([] = exportable)."""
    out = []
    for path, bad, reason in _CHECKS:
        if path in _K3_OK:
            continue
        value = _get(config, path)
        if value is not None and bad(value):
            out.append(f"{path}={value!r}: {reason}")
    a, h, m = config.attention, config.hybrid_attention, config.moe
    if m.latent_dim is not None and getattr(m, "router_input", "latent") != "full":
        out.append("moe.router_input='latent': kimi-k3 routes LatentMoE on the "
                   "full-width input (router_input='full')")
    if m.activation not in ("swiglu", "situ_k3"):
        out.append(f"moe.activation={m.activation!r}: kimi-k3 computes SwiGLU or situ_k3")
    if (config.use_attention_residuals
            and getattr(config, "attention_residual_style", "gate") != "block"):
        out.append("use_attention_residuals=True with attention_residual_style='gate': "
                   "kimi-k3 computes the 'block' style")
    if not a.nope:
        out.append("attention.nope=False: kimi-k3 MLA is nope-only (no RoPE)")
    if a.nope_attn_temperature:
        out.append("attention.nope_attn_temperature=True: NoPE attention temperature")
    if h.enabled:
        if getattr(h, "style", "default") != "k3":
            out.append(f"hybrid_attention.style={h.style!r}: only the 'k3' KDA layer "
                       "matches llama.cpp")
        if h.linear_num_heads != a.num_attention_heads:
            out.append(f"hybrid_attention.linear_num_heads={h.linear_num_heads}: "
                       "llama.cpp shares one head count between KDA and MLA layers "
                       f"(attention.num_attention_heads={a.num_attention_heads})")
        if h.gate_rank not in (None, h.linear_head_dim):
            out.append(f"hybrid_attention.gate_rank={h.gate_rank}: llama.cpp's "
                       "f_a/f_b rank is the KDA head dim")
    return out


def llamacpp_arch(config):
    """The llama.cpp architecture ``config`` exports to, or None."""
    if not llamacpp_incompatibilities(config):
        return ARCHITECTURE
    if not kimi_k3_incompatibilities(config):
        return KIMI_K3
    return None


def _kda_layers(model, config):
    if hasattr(model, "layers"):
        return [bool(getattr(layer.attention, "is_recurrent_attention", False))
                for layer in model.layers]
    sd = model
    return [f"layers.{i}.attention.beta_proj.weight" in sd
            for i in range(config.num_hidden_layers)]


def to_kimi_k3_tensors(model, config):
    """Map a hybrid EvastarV5 model to llama.cpp kimi-k3 tensors.

    Returns ``(tensors, kda_layers)``; raises ValueError on an incompatible
    config or an unmapped parameter.
    """
    problems = kimi_k3_incompatibilities(config)
    if problems:
        raise ValueError("config cannot run as llama.cpp kimi-k3:\n  - "
                         + "\n  - ".join(problems))
    kda = _kda_layers(model, config)
    sd = model.state_dict() if not isinstance(model, Mapping) else dict(model)
    params = ({n for n, _ in model.named_parameters()}
              if hasattr(model, "named_parameters") else set(sd))
    used = set()

    def take(name):
        if name not in sd:
            raise ValueError(f"expected tensor {name!r} is missing")
        used.add(name)
        return sd[name].detach().float()

    a = config.attention
    H = a.num_attention_heads
    nope, v_dim = a.no_rope_head_dim, a.v_head_dim
    zc = bool(getattr(config, "zero_centered_norm", False))
    n_dense = int(config.moe.first_k_dense or 0)
    res = bool(config.use_attention_residuals)
    out = [("token_embd.weight", take("embed_tokens.weight"))]
    if res:
        out.append(("output_res_score.weight", take("output_res_score")))

    for i in range(config.num_hidden_layers):
        p, b = f"layers.{i}.", f"blk.{i}."
        att = p + "attention."
        out.append((b + "attn_norm.weight", _norm_weight(take(p + "pre_attn_norm.weight"), zc)))
        if res:
            out += [(b + "attn_res_score.weight", take(p + "attn_res_score")),
                    (b + "ffn_res_score.weight", take(p + "ffn_res_score"))]
        if kda[i]:
            for x in "qkv":
                out.append((b + f"attn_{x}.weight", take(att + f"{x}_proj.weight")))
            conv = take(att + "conv_weight")                      # [3 * inner, K]
            for x, w in zip("qkv", conv.chunk(3, 0)):
                out.append((b + f"ssm_conv1d_{x}.weight", w[:, None, :].contiguous()))
            out += [(b + "ssm_f_a.weight", take(att + "f_a.weight")),
                    (b + "ssm_f_b.weight", take(att + "f_b.weight")),
                    (b + "ssm_beta.weight", take(att + "beta_proj.weight")),
                    (b + "ssm_a", -torch.exp(take(att + "A_log"))),
                    (b + "ssm_dt.bias", take(att + "dt_bias")),
                    (b + "ssm_g.weight", take(att + "out_gate.weight")),
                    (b + "ssm_norm.weight", take(att + "o_norm")),
                    (b + "attn_output.weight", take(att + "o_proj.weight"))]
        else:
            out.append((b + "attn_q_a.weight", take(att + "q_a_proj.weight")))
            out.append((b + "attn_q_a_norm.weight", _norm_weight(take(att + "norm_q.weight"), zc)))
            out.append((b + "attn_q_b.weight", take(att + "q_b_proj.weight")))
            kv_a = torch.cat([take(att + "kv_a_proj.weight"), take(att + "k_rope_proj.weight")], 0)
            out.append((b + "attn_kv_a_mqa.weight", kv_a))
            out.append((b + "attn_kv_a_norm.weight", _norm_weight(take(att + "norm_kv.weight"), zc)))
            kv_b = take(att + "kv_b_proj.weight").view(H, nope + v_dim, -1)
            out.append((b + "attn_k_b.weight", kv_b[:, :nope, :].transpose(1, 2).contiguous()))
            out.append((b + "attn_v_b.weight", kv_b[:, nope:, :].contiguous()))
            if a.output_gate:
                out.append((b + "attn_gate.weight", take(att + "o_gate_proj.weight")))
            out.append((b + "attn_output.weight", take(att + "o_proj.weight")))
        out.append((b + "ffn_norm.weight", _norm_weight(take(p + "pre_moe_norm.weight"), zc)))
        out += _ffn_tensors(take, config, p, b, i < n_dense)

    out.append(("output_norm.weight", _norm_weight(take("norm.weight"), zc)))
    out.append(("output.weight", take("lm_head.weight")))
    unmapped = sorted(n for n in params - used if not n.startswith(_K3_TEXT_ONLY))
    if unmapped:
        raise ValueError(
            "parameters with no kimi-k3 counterpart (refusing to drop them): "
            + ", ".join(unmapped[:8]) + (" ..." if len(unmapped) > 8 else ""))
    return out, kda


def _ffn_tensors(take, config, p, b, dense):
    if dense:
        gate, up = _split_w13(take(p + "moe.w13.weight"))
        return [(b + "ffn_gate.weight", gate), (b + "ffn_up.weight", up),
                (b + "ffn_down.weight", take(p + "moe.w2.weight"))]
    out = [(b + "ffn_gate_inp.weight", take(p + "moe.router.weight")),
           (b + "exp_probs_b.bias", take(p + "moe.route_bias"))]
    if config.moe.latent_dim is not None:
        out += [(b + "ffn_routed_down.weight", take(p + "moe.down_proj.weight")),
                (b + "ffn_routed_up.weight", take(p + "moe.up_proj.weight"))]
    gates, ups, downs = [], [], []
    for e in range(config.moe.num_experts):
        gate, up = _split_w13(take(p + f"moe.experts.{e}.w13.weight"))
        gates.append(gate)
        ups.append(up)
        downs.append(take(p + f"moe.experts.{e}.w2.weight"))
    out += [(b + "ffn_gate_exps.weight", torch.stack(gates)),
            (b + "ffn_up_exps.weight", torch.stack(ups)),
            (b + "ffn_down_exps.weight", torch.stack(downs))]
    sg, su, sdn = [], [], []
    for s in range(config.moe.num_shared_experts):
        gate, up = _split_w13(take(p + f"moe.shared_experts.{s}.w13.weight"))
        sg.append(gate)
        su.append(up)
        sdn.append(take(p + f"moe.shared_experts.{s}.w2.weight"))
    out += [(b + "ffn_gate_shexp.weight", torch.cat(sg, 0)),
            (b + "ffn_up_shexp.weight", torch.cat(su, 0)),
            (b + "ffn_down_shexp.weight", torch.cat(sdn, 1))]
    return out


def kimi_k3_metadata(config, kda_layers, *, dtype="f16", name=None, tokenizer=None):
    """Typed GGUF keys for llama.cpp's kimi-k3 loader."""
    a, m, h = config.attention, config.moe, config.hybrid_attention
    arch = KIMI_K3
    kv = []

    def u32(key, value):
        kv.append((key, _VAL_UINT32, int(value)))

    def f32(key, value):
        kv.append((key, _VAL_FLOAT32, float(value)))

    kv.append(("general.architecture", _VAL_STRING, arch))
    u32("general.alignment", DEFAULT_ALIGNMENT)
    kv.append(("general.name", _VAL_STRING, name or getattr(config, "model_name", "Evastar")))
    kv.append(("general.license", _VAL_STRING, "Apache-2.0"))
    u32("general.file_type", _FILE_TYPE_ENUM[dtype])
    u32(f"{arch}.context_length", config.max_position_embeddings)
    u32(f"{arch}.embedding_length", config.hidden_size)
    u32(f"{arch}.block_count", config.num_hidden_layers)
    u32(f"{arch}.feed_forward_length", config.intermediate_size)
    u32(f"{arch}.vocab_size", config.vocab_size)
    u32(f"{arch}.attention.head_count", a.num_attention_heads)
    kv.append((f"{arch}.attention.head_count_kv", _VAL_ARRAY,
               (_VAL_UINT32, [0 if r else 1 for r in kda_layers])))
    u32(f"{arch}.attention.key_length", a.kv_latent_dim + a.rope_head_dim)
    u32(f"{arch}.attention.value_length", a.kv_latent_dim)
    u32(f"{arch}.attention.key_length_mla", a.no_rope_head_dim + a.rope_head_dim)
    u32(f"{arch}.attention.value_length_mla", a.v_head_dim)
    u32(f"{arch}.attention.q_lora_rank", a.q_lora_rank)
    u32(f"{arch}.attention.kv_lora_rank", a.kv_latent_dim)
    f32(f"{arch}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    u32(f"{arch}.rope.dimension_count", a.rope_head_dim)
    u32(f"{arch}.ssm.conv_kernel", h.short_conv_kernel if h.enabled else 4)
    u32(f"{arch}.kda.head_dim", h.linear_head_dim if h.enabled else 1)
    if h.enabled:
        f32(f"{arch}.kda.gate_lower_bound", h.gate_lower_bound)
    u32(f"{arch}.leading_dense_block_count", int(m.first_k_dense or 0))
    u32(f"{arch}.expert_count", m.num_experts)
    u32(f"{arch}.expert_used_count", m.num_activated_experts)
    u32(f"{arch}.expert_shared_count", m.num_shared_experts)
    u32(f"{arch}.expert_feed_forward_length", m.expert_hidden_size)
    f32(f"{arch}.expert_weights_scale", 1.0)
    kv.append((f"{arch}.expert_weights_norm", _VAL_BOOL, False))
    u32(f"{arch}.expert_gating_func", GATING_SIGMOID)
    if m.latent_dim is not None:
        u32(f"{arch}.expert_latent_length", m.latent_dim)
    u32(f"{arch}.attn_res.block_size",
        config.attn_res_block_size if config.use_attention_residuals else 0)
    if m.activation == "situ_k3":
        f32(f"{arch}.activation.situ_beta", m.situ_beta)
        f32(f"{arch}.activation.situ_linear_beta", m.situ_linear_beta)
    else:
        f32(f"{arch}.activation.situ_beta", SWIGLU_SITU_BETA)
        f32(f"{arch}.activation.situ_linear_beta", 0.0)
    if tokenizer is not None:
        kv.extend(gpt2_tokenizer_metadata(tokenizer, config.vocab_size))
        return kv
    kv.append(("tokenizer.ggml.model", _VAL_STRING, "none"))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, key, None)
        if value is not None:
            u32(f"tokenizer.ggml.{key}", value)
    return kv
