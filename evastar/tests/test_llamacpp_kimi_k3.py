"""T138 — hybrid (KDA + MLA) models -> llama.cpp ``kimi-k3`` GGUF, v5.142.

``_kimi_k3_forward`` recomputes logits from the exported tensors the way
llama.cpp's ``src/models/kimi-k3.cpp`` graph does (separate q/k/v short
convs, L2-normed q/k, ``lower_bound * sigmoid`` log-decay with
``ssm_a = -exp(A_log)``, decay-first gated delta rule, per-head RMS norm
times ``sigmoid(ssm_g x)``; nope MLA through ``attn_k_b`` / ``attn_v_b``
with the optional sigmoid output gate; situ MoE), reading only GGUF names
and metadata. Matching Evastar's logits checks the mapping without
llama.cpp in CI. The same files were run through llama.cpp f5027fb during
development (see the v5.142 CHANGELOG entry).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from evastar.configs.config_v5 import (
    AttentionConfig, EvastarV5Config, HybridAttentionConfig)
from evastar.src.export.gguf import read_gguf
from evastar.src.export.llamacpp import (
    export_llamacpp_gguf, kimi_k3_incompatibilities, llamacpp_arch,
    to_kimi_k3_tensors)
from evastar.src.model_v5 import EvastarV5
from evastar.tests.test_llamacpp_export import _rms


def _config(layers=4, attn=None, hybrid=None, moe=None, **top):
    c = EvastarV5Config(
        size="lite", num_hidden_layers=layers,
        attention=AttentionConfig(**{"nope": True, **(attn or {})}),
        hybrid_attention=HybridAttentionConfig(**{
            "enabled": True, "style": "k3", "per_channel_decay": True,
            "short_conv_kernel": 4, "first_layer_full": False,
            "full_attention_every": 4, **(hybrid or {})}),
        **top)
    for k, v in (moe or {}).items():
        setattr(c.moe, k, v)
    return c


def _model(config, seed):
    torch.manual_seed(seed)
    model = EvastarV5(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name or name.endswith(("A_log", "dt_bias", "res_score")):
                p.add_(torch.randn_like(p) * 0.1)
        for layer in model.layers:
            if hasattr(layer.moe, "route_bias"):
                layer.moe.route_bias.copy_(torch.randn_like(layer.moe.route_bias) * 0.05)
    return model


def _situ(g, u, beta, lin):
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g) if beta < 1e20 else F.silu(g)
    return a * (lin * torch.tanh(u / lin) if lin > 0 else u)


def _ffn(h, gate, up, down, beta, lin):
    return _situ(h @ gate.T, h @ up.T, beta, lin) @ down.T


def _kda(h, b, md, arch, eps):
    T = h.shape[0]
    H, D = md[f"{arch}.attention.head_count"], md[f"{arch}.kda.head_dim"]
    K, lb = md[f"{arch}.ssm.conv_kernel"], md[f"{arch}.kda.gate_lower_bound"]

    def conv(x):
        y = h @ b(f"attn_{x}.weight").T                         # [T, C]
        w = b(f"ssm_conv1d_{x}.weight").reshape(-1, K)          # [C, K], tap K-1 = now
        y = F.conv1d(F.pad(y.T[None], (K - 1, 0)), w[:, None, :], groups=w.shape[0])
        return F.silu(y[0].T).reshape(T, H, D)

    def l2(t):
        return t / torch.sqrt((t * t).sum(-1, keepdim=True) + eps)

    q, k, v = l2(conv("q")) / math.sqrt(D), l2(conv("k")), conv("v")
    z = (h @ b("ssm_f_a.weight").T) @ b("ssm_f_b.weight").T + b("ssm_dt.bias")
    g = lb * torch.sigmoid(-(z.view(T, H, D) * b("ssm_a")[:, None]))
    beta = torch.sigmoid(h @ b("ssm_beta.weight").T)           # [T, H]
    S = h.new_zeros(H, D, D)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[t])[..., None]
        d = (v[t] - torch.einsum("hkv,hk->hv", S, k[t])) * beta[t, :, None]
        S = S + k[t, :, :, None] * d[:, None, :]
        outs.append(torch.einsum("hkv,hk->hv", S, q[t]))
    o = _rms(torch.stack(outs), b("ssm_norm.weight"), eps)
    o = o * torch.sigmoid(h @ b("ssm_g.weight").T).view(T, H, D)
    return o.reshape(T, -1) @ b("attn_output.weight").T


def _mla(h, b, md, arch, eps):
    T = h.shape[0]
    n_head, kv_lora = md[f"{arch}.attention.head_count"], md[f"{arch}.attention.kv_lora_rank"]
    rope = md[f"{arch}.rope.dimension_count"]
    nope = md[f"{arch}.attention.key_length_mla"] - rope
    q = _rms(h @ b("attn_q_a.weight").T, b("attn_q_a_norm.weight"), eps)
    q = (q @ b("attn_q_b.weight").T).view(T, n_head, nope + rope)
    kv = h @ b("attn_kv_a_mqa.weight").T
    c = _rms(kv[:, :kv_lora], b("attn_kv_a_norm.weight"), eps)
    q_abs = torch.einsum("thn,hrn->thr", q[..., :nope], b("attn_k_b.weight"))
    scores = (torch.einsum("thr,sr->hts", q_abs, c)
              + torch.einsum("thp,sp->hts", q[..., nope:], kv[:, kv_lora:]))
    scores = scores / math.sqrt(nope + rope)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    probs = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    lat = torch.einsum("hts,sr->thr", probs, c)
    o = torch.einsum("thr,hvr->thv", lat, b("attn_v_b.weight")).reshape(T, -1)
    if b("attn_gate.weight", None) is not None:
        o = o * torch.sigmoid(h @ b("attn_gate.weight").T)
    return o @ b("attn_output.weight").T


def _res_mix(bank, cur, w, eps):
    """kimi-k3 res_mix: softmax over sum(rms_norm(x) * w) of the bank + cur."""
    if not bank:
        return cur
    xs = torch.stack(bank + [cur], 1)                        # [T, n+1, d]
    scores = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps) * w).sum(-1)
    return (scores.softmax(-1)[..., None] * xs).sum(1)


def _kimi_k3_forward(t, md, ids):
    arch = md["general.architecture"]
    g = lambda k: md[f"{arch}.{k}"]                          # noqa: E731
    eps = g("attention.layer_norm_rms_epsilon")
    beta, lin = g("activation.situ_beta"), g("activation.situ_linear_beta")
    n_dense, k_used = g("leading_dense_block_count"), g("expert_used_count")
    res_bs = g("attn_res.block_size")
    x = t["token_embd.weight"][ids]
    bank = []
    for i in range(g("block_count")):
        b = lambda n, *d: t.get(f"blk.{i}.{n}", *d) if d else t[f"blk.{i}.{n}"]  # noqa: E731
        cur = _res_mix(bank, x, b("attn_res_score.weight"), eps) if res_bs else x
        banked = bool(res_bs) and i % res_bs == 0
        if banked:
            bank.append(x)
        h = _rms(cur, b("attn_norm.weight"), eps)
        recurrent = g("attention.head_count_kv")[i] == 0
        a = (_kda if recurrent else _mla)(h, b, md, arch, eps)
        x = a if banked else x + a
        cur = _res_mix(bank, x, b("ffn_res_score.weight"), eps) if res_bs else x
        h = _rms(cur, b("ffn_norm.weight"), eps)
        if i < n_dense:
            x = x + _ffn(h, b("ffn_gate.weight"), b("ffn_up.weight"), b("ffn_down.weight"), beta, lin)
            continue
        probs = torch.sigmoid(h @ b("ffn_gate_inp.weight").T)   # full-width router
        sel = (probs + b("exp_probs_b.bias")).topk(k_used, -1).indices
        w = probs.gather(1, sel)
        z = h @ b("ffn_routed_down.weight").T if b("ffn_routed_down.weight", None) is not None else h
        out = torch.zeros_like(z)
        for tok in range(ids.shape[0]):
            for j in range(k_used):
                e = sel[tok, j]
                out[tok] += w[tok, j] * _ffn(
                    z[tok:tok + 1], b("ffn_gate_exps.weight")[e], b("ffn_up_exps.weight")[e],
                    b("ffn_down_exps.weight")[e], beta, lin)[0]
        if b("ffn_routed_up.weight", None) is not None:
            out = out @ b("ffn_routed_up.weight").T
        out = out + _ffn(h, b("ffn_gate_shexp.weight"), b("ffn_up_shexp.weight"),
                         b("ffn_down_shexp.weight"), beta, lin)
        x = x + out
    if res_bs:
        x = _res_mix(bank, x, t["output_res_score.weight"], eps)
    return _rms(x, t["output_norm.weight"], eps) @ t["output.weight"].T


def _check(config, seed, tmp_path, T=40):
    model = _model(config, seed)
    path = export_llamacpp_gguf(model, tmp_path / "m.gguf", config, dtype="f32")
    gf = read_gguf(path)
    assert gf.metadata["general.architecture"] == "kimi-k3"
    tensors = {n: gf.get_tensor(n) for n in gf.tensor_names()}
    ids = torch.randint(3, config.vocab_size, (T,), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        ref = model(ids[None])[0][0]
        got = _kimi_k3_forward(tensors, gf.metadata, ids)
    assert (got - ref).abs().max().item() < 1e-4, (got - ref).abs().max().item()
    return gf


@pytest.mark.parametrize("variant", [
    {}, {"attn": {"output_gate": True}}, {"moe": {"first_k_dense": 1}},
    {"moe": {"num_shared_experts": 2}}, {"zero_centered_norm": True},
    {"hybrid": {"chunk_size": 16}}, {"layers": 6, "hybrid": {"full_attention_every": 2}},
    {"moe": {"latent_dim": 64, "router_input": "full"}},
    {"moe": {"activation": "situ_k3", "situ_beta": 3.0, "situ_linear_beta": 2.0,
             "first_k_dense": 1}},
    {"moe": {"activation": "situ_k3", "situ_linear_beta": 0.0}},
    {"use_attention_residuals": True, "attention_residual_style": "block",
     "attn_res_block_size": 2},
    {"layers": 8, "use_attention_residuals": True, "attention_residual_style": "block",
     "attn_res_block_size": 3, "attn": {"output_gate": True},
     "moe": {"latent_dim": 64, "router_input": "full", "activation": "situ_k3"}},
])
def test_kimi_k3_graph_matches_evastar(tmp_path, variant):
    _check(_config(**variant), 3, tmp_path)


def test_metadata_and_layer_types(tmp_path):
    config = _config()
    gf = _check(config, 4, tmp_path)
    md = gf.metadata
    assert list(md["kimi-k3.attention.head_count_kv"]) == [0, 0, 0, 1]
    assert md["kimi-k3.kda.head_dim"] == config.hybrid_attention.linear_head_dim
    assert md["kimi-k3.ssm.conv_kernel"] == 4
    names = set(gf.tensor_names())
    assert "blk.0.ssm_conv1d_q.weight" in names and "blk.3.attn_kv_a_mqa.weight" in names
    assert "blk.0.attn_kv_a_mqa.weight" not in names


def test_arch_selection():
    assert llamacpp_arch(EvastarV5Config(size="lite")) == "deepseek2"
    assert llamacpp_arch(_config()) == "kimi-k3"
    assert llamacpp_arch(_config(hybrid={"style": "default"})) is None
    assert llamacpp_arch(_config(attn={"nope": False})) is None


@pytest.mark.parametrize("change,needle", [
    ({"hybrid": {"style": "default"}}, "hybrid_attention.style"),
    ({"attn": {"nope": False}}, "attention.nope"),
    ({"hybrid": {"gate_rank": 8}}, "hybrid_attention.gate_rank"),
    ({"moe": {"latent_dim": 64}}, "moe.router_input"),
    ({"moe": {"activation": "situ"}}, "moe.activation"),
    ({"use_attention_residuals": True}, "use_attention_residuals"),
])
def test_incompatible_configs_are_refused(change, needle):
    config = _config(**change)
    problems = kimi_k3_incompatibilities(config)
    assert len(problems) == 1 and problems[0].startswith(needle), problems
    with pytest.raises(ValueError, match="cannot run as llama.cpp kimi-k3"):
        to_kimi_k3_tensors({}, config)


def test_head_counts_must_match():
    config = _config()
    config.hybrid_attention.linear_num_heads += 1
    assert any("linear_num_heads" in p for p in kimi_k3_incompatibilities(config))


def test_unmapped_params_fail():
    config = _config()
    model = _model(config, 5)
    model.layers[0].extra = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no kimi-k3 counterpart"):
        to_kimi_k3_tensors(model, config)
