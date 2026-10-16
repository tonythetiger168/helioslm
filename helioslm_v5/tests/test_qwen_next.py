"""T101 — roadmap Phase B3 (Qwen comparison), Q1–Q5.

Pins:

Q1 short conv in GDA: off = no parameter and a (state,) cache; on =
   cached decode and chunked prefill equal the full forward, packed
   documents stay isolated, left padding does not leak, bad kernels are
   refused.
Q2 zero-centered RMSNorm: bit-identical at init; with weight w it equals
   the standard norm with weight 1 + w; decay_param_groups partitions
   every trainable parameter and decays zero-centered norm weights only.
Q3 global-batch balance loss: over micro-batches it equals the
   full-batch loss and router gradients; the micro scope does not;
   eval forwards record nothing; records are cleared.
Q4 qwen_next_config: layout, sparsity, decode == full.
Q5 the committed MTP artifact: greedy-equivalence gate holds for both
   models, the toy head's acceptance and the hash.

Run from repo root: python3 helioslm_v5/tests/test_qwen_next.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, HybridAttentionConfig, qwen_next_config)
from helioslm_v5.src.attention.linear_attention import GatedDeltaAttention  # noqa: E402
from helioslm_v5.src.attention.mla import RMSNorm  # noqa: E402
from helioslm_v5.src.model_v5 import (  # noqa: E402
    HeliosLMv5, balance_loss, decay_param_groups)


def _gda(kernel):
    cfg = HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        enabled=True, short_conv_kernel=kernel))
    torch.manual_seed(0)
    return GatedDeltaAttention(cfg).eval(), cfg


def test_q1_short_conv():
    off, _ = _gda(0)
    assert not any("conv" in n for n, _ in off.named_parameters())
    x = torch.randn(2, 9, 256)
    with torch.no_grad():
        _, past = off(x, use_cache=True)
    assert len(past) == 1

    gda, _ = _gda(4)
    with torch.no_grad():
        gda.conv_weight.normal_(0, 0.5)            # non-identity taps
        full, _ = gda(x)
        a, p = gda(x[:, :4], use_cache=True)
        outs = [a]
        for t in range(4, 9):
            o, p = gda(x[:, t:t + 1], past_key_value=p, use_cache=True)
            outs.append(o)
        step = torch.cat(outs, 1)
        b, p2 = gda(x[:, :5], use_cache=True)
        c, _ = gda(x[:, 5:], past_key_value=p2,
                   position_ids=torch.arange(5, 9).expand(2, 4))
        chunk = torch.cat([b, c], 1)
    assert len(p) == 2 and getattr(p[1], "_is_recurrent_state", False)
    assert (step - full).abs().max() < 1e-5
    assert (chunk - full).abs().max() < 1e-5

    # packed: [doc A (4) | doc B (5)] == each document alone
    pos = torch.cat([torch.arange(4), torch.arange(5)]).expand(2, 9)
    with torch.no_grad():
        packed, _ = gda(x, position_ids=pos)
        da, _ = gda(x[:, :4])
        db, _ = gda(x[:, 4:])
    assert (packed - torch.cat([da, db], 1)).abs().max() < 1e-5

    # left padding: the real tokens' outputs match the unpadded run
    xp = torch.cat([torch.randn(2, 3, 256) * 5, x], 1)
    mask = torch.cat([torch.zeros(2, 3), torch.ones(2, 9)], 1).long()
    with torch.no_grad():
        padded, _ = gda(xp, attention_mask=mask)
    assert (padded[:, 3:] - full).abs().max() < 1e-5

    for bad in (1, -2, True, 2.5):
        try:
            HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
                enabled=True, short_conv_kernel=bad))
        except ValueError as e:
            assert "short_conv_kernel" in str(e)
        else:
            raise AssertionError(bad)
    print("PASS test_q1_short_conv")


def test_q2_zero_centered_norm():
    ids = torch.randint(3, 1000, (2, 10))
    outs = []
    for zc in (False, True):
        cfg = HeliosLMv5Config(size="lite", zero_centered_norm=zc)
        cfg.attention.qk_norm = True
        cfg._validate()
        torch.manual_seed(0)
        model = HeliosLMv5(cfg).eval()
        with torch.no_grad():
            outs.append(model(ids)[0])
        groups = decay_param_groups(model, 0.1)
        params = [p for p in model.parameters() if p.requires_grad]
        flat = groups[0]["params"] + groups[1]["params"]
        assert len(flat) == len(params) and \
            {id(p) for p in flat} == {id(p) for p in params}
        norm_w = {id(m.weight) for m in model.modules() if isinstance(m, RMSNorm)}
        decayed = {id(p) for p in groups[0]["params"]}
        assert (norm_w <= decayed) if zc else not (norm_w & decayed)
    assert torch.equal(outs[0], outs[1])

    torch.manual_seed(1)
    w = torch.randn(16) * 0.1
    x = torch.randn(3, 16)
    std, zc = RMSNorm(16), RMSNorm(16, zero_centered=True)
    with torch.no_grad():
        std.weight.copy_(1 + w)
        zc.weight.copy_(w)
    assert torch.allclose(std(x), zc(x), atol=1e-6)
    try:
        HeliosLMv5Config(size="lite", zero_centered_norm=1)
    except ValueError as e:
        assert "zero_centered_norm" in str(e)
    else:
        raise AssertionError("non-bool accepted")
    print("PASS test_q2_zero_centered_norm")


def _balance(scope, split):
    cfg = HeliosLMv5Config(size="lite")
    cfg.moe.balance_loss_scope = scope
    cfg._validate()
    torch.manual_seed(0)
    model = HeliosLMv5(cfg).train()
    ids = torch.randint(3, 1000, (4, 12),
                        generator=torch.Generator().manual_seed(1))
    for chunk in ids.split(split):
        model(chunk)
    loss = balance_loss(model)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for n, p in model.named_parameters()
                      if "router" in n and p.grad is not None])
    return loss.item(), grad, model


def test_q3_global_batch_balance():
    full, g_full, model = _balance("global", 4)
    split, g_split, _ = _balance("global", 1)
    micro, g_micro, _ = _balance("micro", 1)
    assert abs(full - split) < 1e-6 and (g_full - g_split).abs().max() < 1e-7
    assert abs(full - micro) > 1e-3 and (g_full - g_micro).abs().max() > 1e-4
    assert balance_loss(model) is None                 # records were cleared
    model.eval()
    with torch.no_grad():
        model(torch.randint(3, 1000, (1, 6)))
    assert balance_loss(model) is None                 # eval records nothing
    off = HeliosLMv5(HeliosLMv5Config(size="lite")).train()
    off(torch.randint(3, 1000, (1, 6)))
    assert balance_loss(off) is None
    try:
        c = HeliosLMv5Config(size="lite")
        c.moe.balance_loss_scope = "batch"
        c._validate()
    except ValueError as e:
        assert "balance_loss_scope" in str(e)
    else:
        raise AssertionError("bad scope accepted")
    print(f"PASS test_q3_global_batch_balance (global {full:.4f} == split, "
          f"micro {micro:.4f})")


def test_q4_preset():
    full = qwen_next_config("full")
    assert (full.moe.num_experts, full.moe.num_activated_experts,
            full.moe.num_shared_experts) == (512, 10, 1)
    cfg = qwen_next_config()
    assert cfg.zero_centered_norm and cfg.attention.output_gate \
        and cfg.hybrid_attention.short_conv_kernel == 4 \
        and cfg.moe.balance_loss_scope == "global"
    torch.manual_seed(0)
    model = HeliosLMv5(cfg).eval()
    layout = "".join("L" if getattr(l.attention, "is_recurrent_attention", False)
                     else "A" for l in model.layers)
    assert layout == "LLLA", layout
    ids = torch.randint(3, cfg.vocab_size, (2, 14))
    with torch.no_grad():
        ref = model(ids)[0]
        lg, _, past = model(ids[:, :7], use_cache=True)
        outs = [lg]
        for t in range(7, 14):
            lg, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
            outs.append(lg)
    assert (torch.cat(outs, 1) - ref).abs().max() < 1e-4
    print("PASS test_q4_preset")


def test_q5_mtp_artifact():
    art = json.loads((_REPO / "benchmarks"
                      / "mtp_acceptance_2026-10-16.json").read_text())
    res = {r["model"]: r for r in art["results"]}
    assert set(res) == {"toy_v5.13", "qwen_next_lite_untrained"}
    assert all(r["gate_greedy_equal"] for r in res.values())
    assert all(p["greedy_equal"] for r in res.values() for p in r["per_prompt"])
    assert res["toy_v5.13"]["acceptance_rate"] > 0.5
    canon = [{**r, "per_prompt": [{k: v for k, v in p.items()
                                   if not k.startswith("tok_s")}
                                  for p in r["per_prompt"]]}
             for r in art["results"]]
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()).hexdigest()
    print("PASS test_q5_mtp_artifact "
          f"(toy acceptance {res['toy_v5.13']['acceptance_rate']})")


if __name__ == "__main__":
    test_q1_short_conv()
    test_q2_zero_centered_norm()
    test_q3_global_batch_balance()
    test_q4_preset()
    test_q5_mtp_artifact()
