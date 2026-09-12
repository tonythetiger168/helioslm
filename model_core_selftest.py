"""Self-tests for model-core fixes (config/mla/moe/model_v5)."""
import sys, torch
sys.path.insert(0, "/mnt/agents/output")

from helioslm_v5.configs.config_v5 import HeliosLMv5Config

torch.manual_seed(0)
results = []

def check(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"[PASS] {name}")
    except Exception as e:
        results.append((name, "FAIL", repr(e)))
        print(f"[FAIL] {name}: {e!r}")


# 1. lite config + single forward shape + 3-tuple
def t1():
    cfg = HeliosLMv5Config(size="lite")
    assert cfg.hidden_size == 256 and cfg.num_hidden_layers == 2
    assert cfg.attention.num_attention_heads == 4
    assert cfg.attention.kv_latent_dim == 64 and cfg.attention.rope_head_dim == 8
    assert cfg.attention.v_head_dim == 32 and cfg.attention.no_rope_head_dim == 24
    assert cfg.vocab_size == 1024 and cfg.moe.num_experts == 8
    assert cfg.moe.num_activated_experts == 2 and cfg.mtp.num_modules == 1
    assert cfg.max_position_embeddings == 2048
    assert cfg.moe.expert_hidden_size == 512  # from intermediate_size
    from helioslm_v5.src.model_v5 import HeliosLMv5
    model = HeliosLMv5(cfg)
    ids = torch.randint(0, 1024, (2, 7))
    out = model(ids)
    assert isinstance(out, tuple) and len(out) == 3, "must return 3-tuple"
    logits, hidden, past = out
    assert logits.shape == (2, 7, 1024) and hidden.shape == (2, 7, 256)
    assert past is None
    # invalid config must raise
    try:
        HeliosLMv5Config(size="full", hidden_size=4096,
                         attention=type(cfg.attention)(num_attention_heads=96))
        raise AssertionError("should have raised")
    except ValueError:
        pass
check("1. lite config + forward 3-tuple + validation", t1)


# 2. cached decode: growth + numerical match vs full forward
def t2():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).eval()
    ids = torch.randint(0, 1024, (2, 6))
    with torch.no_grad():
        full_logits, _, _ = model(ids)
        past = None
        step_logits = []
        for t in range(ids.shape[1]):
            logits, _, past = model(ids[:, t:t+1], past_key_values=past, use_cache=True)
            step_logits.append(logits[:, -1, :])
            assert past[0][0].shape[2] == t + 1, f"past len {past[0][0].shape[2]} != {t+1}"
        # extra 5 decode steps beyond prefill
        cur = full_logits[:, -1, :].argmax(-1, keepdim=True)
        for t in range(5):
            expected_len = ids.shape[1] + t
            logits, _, past = model(cur, past_key_values=past, use_cache=True)
            assert past[0][0].shape[2] == expected_len + 1
            cur = logits[:, -1, :].argmax(-1, keepdim=True)
        step_logits = torch.stack(step_logits, dim=1)
        diff = (step_logits - full_logits).abs().max().item()
        assert diff < 1e-4, f"cache vs full forward mismatch: {diff}"
        print(f"    max|cached-full| = {diff:.2e}")
check("2. cached decode == full forward (C1/C2/C3)", t2)


# 3. generate batch=2, temperature=0, determinism
def t3():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).eval()
    ids = torch.randint(0, 1024, (2, 5))
    g1 = model.generate(ids, max_new_tokens=8, temperature=0)
    g2 = model.generate(ids, max_new_tokens=8, temperature=0)
    assert g1.shape[0] == 2 and g1.shape[1] <= 13
    assert torch.equal(g1, g2), "greedy must be deterministic"
    # temperature>0 path also works for batch>1
    g3 = model.generate(ids, max_new_tokens=4, temperature=0.8, top_p=0.9)
    assert g3.shape[0] == 2
check("3. generate batch=2 greedy + sampling", t3)


# 4. MoE bias semantics + update_bias
def t4():
    from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE
    cfg = HeliosLMv5Config(size="lite")
    moe = DeviceLimitedMoE(cfg)
    assert moe.route_bias.requires_grad == False
    x = torch.randn(3, 4, cfg.hidden_size)
    flat = x.reshape(-1, cfg.hidden_size)
    scores = torch.sigmoid(moe.router(flat))
    with torch.no_grad():
        moe.route_bias += torch.linspace(0, 0.5, cfg.moe.num_experts)
    idx, gates = moe._route(flat)
    sel = torch.topk(scores + moe.route_bias, cfg.moe.num_activated_experts, -1).indices
    assert torch.equal(idx, sel), "selection must use bias-augmented scores"
    assert torch.allclose(gates, scores.gather(1, idx)), "gates must be bias-free"
    assert not torch.allclose(gates, (scores + moe.route_bias).gather(1, idx)), \
        "gates must NOT contain the bias"
    # update_bias: overloaded expert's bias decreases
    load = torch.tensor([100., 0, 0, 0, 0, 0, 0, 0])
    before = moe.route_bias.clone()
    moe.update_bias(load, step=0.1)
    assert moe.route_bias[0] < before[0], "overloaded expert bias must decrease"
    assert (moe.route_bias[1:] > before[1:]).all(), "underloaded expert bias must increase"
    # forward accumulates load stats; update_bias() consumes + resets them
    moe(x)
    assert moe.expert_load.sum().item() == 3 * 4 * cfg.moe.num_activated_experts
    moe.update_bias()
    assert moe.expert_load.sum().item() == 0
check("4. MoE bias selection/gating + update_bias direction", t4)


# 5. RoPE lazy extension beyond budget
def t5():
    from helioslm_v5.src.attention.mla import RotaryEmbedding, MLA
    rope = RotaryEmbedding(8, max_position_embeddings=2048)
    assert rope.max_seq_len == 2048  # min(2048, 8192)
    pos = torch.arange(5000).unsqueeze(0)
    cos, sin = rope(pos)
    assert cos.shape == (1, 1, 5000, 8) and rope.max_seq_len >= 5000
    # full MLA over initial budget via max_position_embeddings smaller than budget
    cfg = HeliosLMv5Config(size="lite")
    cfg.max_position_embeddings = 100  # but rope budget = min(100, 8192) = 100
    mla = MLA(cfg)
    assert mla.rope.max_seq_len == 100
    h = torch.randn(1, 150, cfg.hidden_size)
    out, _ = mla(h)
    assert out.shape == h.shape and mla.rope.max_seq_len >= 150
check("5. RoPE lazy extension", t5)


# 6. backward: grads flow to q_b_proj / kv_b_proj / router
def t6():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    ids = torch.randint(0, 1024, (2, 6))
    logits, hidden, _ = model(ids)
    loss = logits.sum() + hidden.pow(2).mean()
    loss.backward()
    layer = model.layers[0]
    for name, p in [("q_b_proj", layer.attention.q_b_proj.weight),
                    ("kv_b_proj", layer.attention.kv_b_proj.weight),
                    ("router", layer.moe.router.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} grad missing"
    assert model.layers[0].moe.route_bias.grad is None or \
        model.layers[0].moe.route_bias.grad.abs().sum() == 0
check("6. backward grads (q_b/kv_b/router)", t6)


# 7. attention_mask (padding) support
def t7():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).eval()
    ids = torch.randint(1, 1024, (1, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        logits_m, _, _ = model(ids, attention_mask=mask)
        # changing pad positions' token ids must not affect real-token logits
        ids2 = ids.clone(); ids2[0, 4:] = 5
        logits_m2, _, _ = model(ids2, attention_mask=mask)
        diff = (logits_m[:, :4] - logits_m2[:, :4]).abs().max().item()
        assert diff < 1e-4, f"padding leaked into attention: {diff}"
check("7. padding mask respected (M11)", t7)


# 8. use_cache=False with past passed (defensive) + attention_mask in decode
def t8():
    from helioslm_v5.src.model_v5 import HeliosLMv5
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).eval()
    ids = torch.randint(0, 1024, (1, 4))
    with torch.no_grad():
        _, _, past = model(ids, use_cache=True)
        # past passed but use_cache=False must not crash
        logits, _, p = model(ids[:, :1], past_key_values=past, use_cache=False)
        assert p is None and logits.shape == (1, 1, 1024)
        # decode with explicit full-length attention mask (4 past + 1 current)
        mask = torch.ones(1, 5, dtype=torch.long)
        logits, _, past = model(ids[:, :1], attention_mask=mask,
                                past_key_values=past, use_cache=True)
        assert logits.shape == (1, 1, 1024) and past[0][0].shape[2] == 5
        # current-only mask is left-padded with ones for the cached prefix
        logits, _, past = model(ids[:, :1], attention_mask=torch.ones(1, 1),
                                past_key_values=past, use_cache=True)
        assert past[0][0].shape[2] == 6
check("8. defensive past/use_cache + mask in decode", t8)

print("=" * 50)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
print(f"{n_pass}/{len(results)} passed")
sys.exit(0 if n_pass == len(results) else 1)
