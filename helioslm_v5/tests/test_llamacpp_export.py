"""T131 — HeliosLM -> llama.cpp ``deepseek2`` GGUF export, v5.130.

``_deepseek2_forward`` recomputes logits from the exported tensors the way
llama.cpp's ``src/models/deepseek2.cpp`` graph does (NORM RoPE on adjacent
pairs, absorbed MLA through ``attn_k_b`` / ``attn_v_b``, sigmoid gating with
``exp_probs_b`` for selection only, shared experts), reading only GGUF
names and metadata, never the HeliosLM modules. Matching HeliosLM's own
logits checks the whole mapping, RoPE permutation included, without
llama.cpp in CI. The same files were also run through llama.cpp itself
during development (see the v5.130 CHANGELOG entry).
"""
import math
import struct

import pytest
import torch
import torch.nn.functional as F

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.export.gguf import (
    GGML_TYPE_F32, _VAL_BOOL, _VAL_FLOAT32, _VAL_UINT32, read_gguf)
from helioslm_v5.src.export.llamacpp import (
    deepseek2_metadata, export_llamacpp_gguf, llamacpp_incompatibilities,
    to_deepseek2_tensors)
from helioslm_v5.src.model_v5 import HeliosLMv5


def _model(seed, layers=2, **moe):
    torch.manual_seed(seed)
    config = HeliosLMv5Config(size="lite", num_hidden_layers=layers)
    for k, v in moe.items():
        setattr(config.moe, k, v)
    model = HeliosLMv5(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:                # non-trivial norm gains
                p.add_(torch.randn_like(p) * 0.1)
        for layer in model.layers:
            if hasattr(layer.moe, "route_bias"):
                layer.moe.route_bias.copy_(torch.randn_like(layer.moe.route_bias) * 0.05)
    return model, config


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope_norm(x, pos, base):
    """ggml NORM rope: rotate adjacent pairs (2i, 2i+1) by pos * base^(-2i/d)."""
    d = x.shape[-1]
    theta = pos[:, None].float() * base ** (-torch.arange(0, d, 2).float() / d)
    theta = theta.view(theta.shape[0], *[1] * (x.dim() - 2), d // 2)  # [T, (H,) d/2]
    cos, sin = theta.cos(), theta.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def _swiglu(x, gate, up, down):
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def _deepseek2_forward(t, md, ids):
    arch = md["general.architecture"]
    g = lambda k: md[f"{arch}.{k}"]                          # noqa: E731
    n_layer, n_head = g("block_count"), g("attention.head_count")
    kv_lora, rope = g("attention.kv_lora_rank"), g("rope.dimension_count")
    nope = g("attention.key_length_mla") - rope
    eps, base = g("attention.layer_norm_rms_epsilon"), g("rope.freq_base")
    n_dense, k_used = g("leading_dense_block_count"), g("expert_used_count")
    kq_scale = 1.0 / math.sqrt(g("attention.key_length_mla"))
    T = ids.shape[0]
    pos = torch.arange(T)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    x = t["token_embd.weight"][ids]
    for i in range(n_layer):
        b = lambda n: t[f"blk.{i}.{n}"]                      # noqa: E731
        h = _rms(x, b("attn_norm.weight"), eps)
        q = _rms(h @ b("attn_q_a.weight").T, b("attn_q_a_norm.weight"), eps)
        q = (q @ b("attn_q_b.weight").T).view(T, n_head, nope + rope)
        q_nope, q_pe = q[..., :nope], _rope_norm(q[..., nope:], pos, base)
        kv = h @ b("attn_kv_a_mqa.weight").T
        c = _rms(kv[:, :kv_lora], b("attn_kv_a_norm.weight"), eps)  # [T, r]
        k_pe = _rope_norm(kv[:, kv_lora:], pos, base)               # [T, rope]
        k_b, v_b = b("attn_k_b.weight"), b("attn_v_b.weight")       # [H,r,nope], [H,v,r]
        q_abs = torch.einsum("thn,hrn->thr", q_nope, k_b)           # absorbed query
        scores = (torch.einsum("thr,sr->hts", q_abs, c)
                  + torch.einsum("thp,sp->hts", q_pe, k_pe)) * kq_scale
        probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        lat = torch.einsum("hts,sr->thr", probs, c)
        o = torch.einsum("thr,hvr->thv", lat, v_b).reshape(T, -1)
        x = x + o @ b("attn_output.weight").T
        h = _rms(x, b("ffn_norm.weight"), eps)
        if i < n_dense:
            x = x + _swiglu(h, b("ffn_gate.weight"), b("ffn_up.weight"), b("ffn_down.weight"))
            continue
        probs_e = torch.sigmoid(h @ b("ffn_gate_inp.weight").T)
        sel = (probs_e + b("exp_probs_b.bias")).topk(k_used, -1).indices
        w = probs_e.gather(1, sel)                   # weights_norm false, scale 1
        out = torch.zeros_like(x)
        for tok in range(T):
            for j in range(k_used):
                e = sel[tok, j]
                out[tok] += w[tok, j] * _swiglu(
                    h[tok:tok + 1], b("ffn_gate_exps.weight")[e],
                    b("ffn_up_exps.weight")[e], b("ffn_down_exps.weight")[e])[0]
        out += _swiglu(h, b("ffn_gate_shexp.weight"), b("ffn_up_shexp.weight"),
                       b("ffn_down_shexp.weight"))
        x = x + out
    return _rms(x, t["output_norm.weight"], eps) @ t["output.weight"].T


@pytest.mark.parametrize("moe", [{}, {"first_k_dense": 1}, {"num_shared_experts": 2},
                                 {"num_activated_experts": 4}])
def test_deepseek2_graph_matches_helioslm(tmp_path, moe):
    model, config = _model(11, **moe)
    path = export_llamacpp_gguf(model, tmp_path / "m.gguf", config, dtype="f32")
    gf = read_gguf(path)
    tensors = {n: gf.get_tensor(n) for n in gf.tensor_names()}
    ids = torch.randint(3, config.vocab_size, (32,), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(ids[None])[0][0]
        got = _deepseek2_forward(tensors, gf.metadata, ids)
    assert (got - ref).abs().max().item() < 1e-4, (got - ref).abs().max().item()


def test_zero_centered_norm_is_folded(tmp_path):
    torch.manual_seed(14)
    config = HeliosLMv5Config(size="lite", zero_centered_norm=True)
    model = HeliosLMv5(config).eval()
    gf = read_gguf(export_llamacpp_gguf(model, tmp_path / "m.gguf", config, dtype="f32"))
    tensors = {n: gf.get_tensor(n) for n in gf.tensor_names()}
    ids = torch.arange(3, 23)
    with torch.no_grad():
        ref = model(ids[None])[0][0]
        got = _deepseek2_forward(tensors, gf.metadata, ids)
    assert (got - ref).abs().max().item() < 1e-4


def test_rope_permutation_is_needed(tmp_path):
    # Without the NeoX -> adjacent-pair reorder the logits would differ:
    # guards against the permutation silently becoming a no-op.
    model, config = _model(12)
    tensors = dict(to_deepseek2_tensors(model, config))
    rope = config.attention.rope_head_dim
    raw = model.state_dict()["layers.0.attention.k_rope_proj.weight"]
    assert not torch.equal(tensors["blk.0.attn_kv_a_mqa.weight"][-rope:], raw)
    assert torch.equal(tensors["blk.0.attn_kv_a_mqa.weight"][-rope:][0::2], raw[:rope // 2])
    assert torch.equal(tensors["blk.0.attn_kv_a_mqa.weight"][-rope:][1::2], raw[rope // 2:])


def test_metadata_types_and_tensor_shapes(tmp_path):
    model, config = _model(13)
    path = export_llamacpp_gguf(model, tmp_path / "m.gguf", config, dtype="q8_0")
    raw = path.read_bytes()
    gf = read_gguf(path)
    md = gf.metadata
    assert md["general.architecture"] == "deepseek2"
    assert md["tokenizer.ggml.model"] == "none"
    assert md["deepseek2.expert_gating_func"] == 2           # sigmoid
    assert md["deepseek2.expert_weights_norm"] is False
    assert md["deepseek2.attention.head_count_kv"] == 1
    a = config.attention
    assert md["deepseek2.attention.key_length"] == a.kv_latent_dim + a.rope_head_dim
    assert md["deepseek2.attention.value_length"] == a.kv_latent_dim
    # llama.cpp checks key types: look at the raw type code of a few keys.
    for key, vtype in (("deepseek2.block_count", _VAL_UINT32),
                       ("deepseek2.rope.freq_base", _VAL_FLOAT32),
                       ("deepseek2.expert_weights_scale", _VAL_FLOAT32),
                       ("deepseek2.expert_weights_norm", _VAL_BOOL)):
        k = key.encode()
        assert struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype) in raw, key
    shapes = {t.name: t.shape for t in gf.tensors}
    types = {t.name: t.ggml_type for t in gf.tensors}
    H, r = a.num_attention_heads, a.kv_latent_dim
    E, ff, D = config.moe.num_experts, config.moe.expert_hidden_size, config.hidden_size
    # PyTorch order; llama.cpp lists the same dims reversed.
    assert shapes["blk.0.attn_k_b.weight"] == (H, r, a.no_rope_head_dim)
    assert shapes["blk.0.attn_v_b.weight"] == (H, a.v_head_dim, r)
    assert shapes["blk.0.attn_kv_a_mqa.weight"] == (r + a.rope_head_dim, D)
    assert shapes["blk.0.ffn_gate_exps.weight"] == (E, ff, D)
    assert shapes["blk.0.ffn_down_exps.weight"] == (E, D, ff)
    assert shapes["blk.0.exp_probs_b.bias"] == (E,)
    for name in ("blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight",
                 "blk.0.ffn_gate_inp.weight", "blk.0.attn_norm.weight",
                 "blk.0.exp_probs_b.bias"):
        assert types[name] == GGML_TYPE_F32, name
    assert not any(n.startswith("mtp") or "nextn" in n for n in shapes)
    assert deepseek2_metadata(config)[0] == ("general.architecture", 8, "deepseek2")


@pytest.mark.parametrize("field,value", [
    ("hybrid_attention.enabled", True), ("attention.nope", True),
    ("attention.qk_norm", True), ("attention.kv_share_last_n", 2),
    ("attention.sliding_window", 64), ("moe.latent_dim", 64),
    ("moe.activation", "situ"), ("moe.num_shared_experts", 0),
    ("use_attention_residuals", True), ("per_layer_embed_dim", 16),
    ("gated_conv_layers", [0]), ("attention.q_lora_rank", 0),
])
def test_incompatible_configs_are_refused(field, value):
    config = HeliosLMv5Config(size="lite")
    obj, attr = config, field
    if "." in field:
        head, attr = field.split(".")
        obj = getattr(config, head)
    setattr(obj, attr, value)
    problems = llamacpp_incompatibilities(config)
    assert len(problems) == 1 and problems[0].startswith(field), problems
    with pytest.raises(ValueError, match="cannot run as llama.cpp deepseek2"):
        to_deepseek2_tensors({}, config)


def test_lite_config_is_compatible_and_unmapped_params_fail():
    model, config = _model(15)
    assert llamacpp_incompatibilities(config) == []
    model.layers[0].extra = torch.nn.Linear(4, 4)        # a module the map does not know
    with pytest.raises(ValueError, match="no deepseek2 counterpart"):
        to_deepseek2_tensors(model, config)
