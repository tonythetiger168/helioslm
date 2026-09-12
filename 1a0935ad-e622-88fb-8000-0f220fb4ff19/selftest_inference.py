"""Self-tests for helioslm_v5 inference fixes (scratch, not part of repo)."""
import math
import sys
from types import SimpleNamespace

sys.path.insert(0, "/mnt/agents/output")

import torch
import torch.nn as nn
import torch.nn.functional as F

from helioslm_v5.src.inference.mtp import MTPModule, MTPDecoder
from helioslm_v5.src.inference.paged_attention import BlockManager, PagedAttention
from helioslm_v5.src.inference.vllm_engine import VLLMEngine, GPUMonitor, Request


# ----------------------------------------------------------------------
# Dummy main model implementing the HeliosLMv5 contract:
#   forward(input_ids, attention_mask=None, position_ids=None,
#           past_key_values=None, use_cache=False, images=None,
#           audio_features=None) -> (logits[B,L,V], hidden[B,L,H], past)
#   generate(input_ids, max_new_tokens, temperature=0, eos_token_id) batch-ok
#   attrs: embed_tokens, lm_head, config
# ----------------------------------------------------------------------
class DummyConfig:
    def __init__(self, vocab=64, hidden=32, nhead=4, eos=2, num_mtp=2):
        self.vocab_size = vocab
        self.hidden_size = hidden
        self.num_hidden_layers = 1
        self.intermediate_size = hidden * 4
        self.eos_token_id = eos
        self.attention = SimpleNamespace(num_attention_heads=nhead)
        self.mtp = SimpleNamespace(num_modules=num_mtp)
        self.batch_size = 4


class DummyLayer(nn.Module):
    def __init__(self, hidden, nhead):
        super().__init__()
        self.nh = nhead
        self.dh = hidden // nhead
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.o = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                 nn.Linear(4 * hidden, hidden))

    def forward(self, x, past=None, use_cache=False):
        B, L, _ = x.shape
        q, k, v = (t.view(B, L, self.nh, self.dh).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1))
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        S = k.shape[2]
        offset = S - L
        mask = torch.triu(torch.full((L, S), float("-inf"), device=x.device),
                          diagonal=1 + offset)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        attn = F.softmax(scores + mask.unsqueeze(0).unsqueeze(0), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        x = self.ln1(x + self.o(out))
        x = self.ln2(x + self.mlp(x))
        return x, present


class DummyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DummyLayer(config.hidden_size, config.attention.num_attention_heads)
            for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, images=None,
                audio_features=None):
        h = self.embed_tokens(input_ids)
        new_past = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            p = past_key_values[i] if past_key_values else None
            h, present = layer(h, p, use_cache)
            if use_cache:
                new_past.append(present)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, h, new_past

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0, eos_token_id=None):
        self.eval()
        generated = input_ids.clone()
        past = None
        B = input_ids.shape[0]
        finished = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new_tokens):
            ids = generated if past is None else generated[:, -1:]
            logits, _, past = self(ids, past_key_values=past, use_cache=True)
            nl = logits[:, -1, :]
            if temperature and temperature > 0:
                nxt = torch.multinomial(F.softmax(nl / temperature, -1), 1)
            else:
                nxt = nl.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None:
                finished |= (nxt.squeeze(1) == eos_token_id)
                if finished.all():
                    break
        return generated


def make_model_and_mtp(seed=0):
    torch.manual_seed(seed)
    cfg = DummyConfig()
    model = DummyLM(cfg)
    mtps = [MTPModule(cfg, module_index=k, embed_tokens=model.embed_tokens,
                      lm_head=model.lm_head) for k in range(cfg.mtp.num_modules)]
    return cfg, model, mtps


def test_mtp_training_shift():
    print("== MTP training-path shift depth ==")
    cfg, model, mtps = make_model_and_mtp(0)
    L = 12
    torch.manual_seed(1)
    hidden = torch.randn(1, L, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (1, L))

    for k, m in enumerate(mtps):
        logits = m(hidden, ids)
        d = k + 1
        assert logits.shape == (1, L - d - 1, cfg.vocab_size), logits.shape
        # tokens before position d are never inputs -> changing them is a no-op
        for j in range(d):
            ids2 = ids.clone()
            ids2[0, j] = (ids2[0, j] + 1) % cfg.vocab_size
            assert torch.equal(m(hidden, ids2), logits), f"module {k}: shift leak at {j}"
        # last token is a target only -> changing it is a no-op
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 1) % cfg.vocab_size
        assert torch.equal(m(hidden, ids2), logits), f"module {k}: target leak"
        # the first used input ids[d] must affect position 0
        ids2 = ids.clone()
        ids2[0, d] = (ids2[0, d] + 1) % cfg.vocab_size
        assert not torch.equal(m(hidden, ids2), logits), f"module {k}: ids[{d}] unused"

    # causal mask: changing a *future* input must not affect earlier outputs
    m0 = mtps[0]  # d=1, uses ids[1..L-2]; position t consumes ids[1..t+1]
    base = m0(hidden, ids)
    ids2 = ids.clone()
    ids2[0, 7] = (ids2[0, 7] + 5) % cfg.vocab_size   # first affects position 6
    after = m0(hidden, ids2)
    assert torch.allclose(base[:, :6], after[:, :6]), "causal mask violated"
    assert not torch.allclose(base[:, 6:], after[:, 6:]), "ids[7] had no effect"

    # weight sharing (M24)
    assert mtps[0].embed_tokens is model.embed_tokens
    assert mtps[0].lm_head is model.lm_head
    print("  [OK] shift depth / causal mask / weight sharing")


class BadMTP(MTPModule):
    """MTP whose drafts are always a fixed token (forces rejections)."""
    def forward_with_hidden(self, h, t):
        logits, hid = super().forward_with_hidden(h, t)
        forced = torch.full_like(logits, -100.0)
        forced[..., 63] = 100.0
        return forced, hid


def test_mtp_generate():
    print("== MTPDecoder.generate ==")
    cfg, model, mtps = make_model_and_mtp(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), generator=torch.Generator().manual_seed(7))
    ref = model.generate(prompt, max_new_tokens=8, temperature=0)

    dec = MTPDecoder(model, mtps, cfg)
    res = dec.generate(prompt, max_new_tokens=8, temperature=0)
    assert res.sequences.shape[1] == prompt.shape[1] + 8, res.sequences.shape
    assert torch.equal(res.sequences, ref), "MTP greedy != main-model greedy"
    assert res.num_drafted > 0 and 0.0 <= res.acceptance_rate <= 1.0
    assert abs(res.acceptance_rate - res.num_accepted / res.num_drafted) < 1e-9
    print(f"  [OK] greedy parity; acceptance={res.acceptance_rate:.2f} "
          f"({res.num_accepted}/{res.num_drafted}), rounds={res.num_rounds}")

    # drafts always wrong -> verify/rollback must still reproduce greedy
    cfg2, model2, _ = make_model_and_mtp(0)
    bad = [BadMTP(cfg2, module_index=k, embed_tokens=model2.embed_tokens,
                  lm_head=model2.lm_head) for k in range(2)]
    dec2 = MTPDecoder(model2, bad, cfg2)
    res2 = dec2.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(res2.sequences, ref), "rollback failed: output != greedy"
    assert res2.acceptance_rate < 1.0, "forced-wrong drafts were accepted"
    print(f"  [OK] verify/rollback: output==greedy, acceptance={res2.acceptance_rate:.2f}")

    # small budget degenerates gracefully (m1)
    res3 = dec.generate(prompt, max_new_tokens=2, temperature=0)
    assert res3.sequences.shape[1] == prompt.shape[1] + 2
    assert torch.equal(res3.sequences, ref[:, : prompt.shape[1] + 2])
    print("  [OK] max_new_tokens < num_mtp+1 degenerates to plain generation")

    # remainder tokens: 7 = 2 full rounds (3+3) + 1 leftover
    res4 = dec.generate(prompt, max_new_tokens=7, temperature=0)
    assert torch.equal(res4.sequences, ref[:, : prompt.shape[1] + 7])
    print("  [OK] remainder tokens filled correctly")

    # sampling path smoke test
    res5 = dec.generate(prompt, max_new_tokens=9, temperature=1.0)
    assert res5.sequences.shape[1] == prompt.shape[1] + 9
    assert 0.0 <= res5.acceptance_rate <= 1.0
    print("  [OK] temperature>0 speculative sampling runs")

    # eos stop
    res6 = dec.generate(prompt, max_new_tokens=20, temperature=0,
                        eos_token_id=int(ref[0, prompt.shape[1] + 2]))
    assert res6.sequences.shape[1] == prompt.shape[1] + 3
    print("  [OK] eos early stop")

    # drafts always right -> accept-all + bonus-token path, minimal rounds
    cfg3 = DummyConfig()
    model3 = DummyLM(cfg3)
    with torch.no_grad():  # constant model: greedy next-token is always 0
        model3.lm_head.weight.zero_()
    class GoodMTP(MTPModule):
        def forward_with_hidden(self, h, t):
            logits, hid = super().forward_with_hidden(h, t)
            forced = torch.full_like(logits, -100.0)
            forced[..., 0] = 100.0
            return forced, hid
    good = [GoodMTP(cfg3, module_index=k, embed_tokens=model3.embed_tokens,
                    lm_head=model3.lm_head) for k in range(2)]
    dec3 = MTPDecoder(model3, good, cfg3)
    res7 = dec3.generate(prompt, max_new_tokens=9, temperature=0)
    ref3 = model3.generate(prompt, max_new_tokens=9, temperature=0)
    assert torch.equal(res7.sequences, ref3)
    assert res7.acceptance_rate == 1.0, res7
    assert res7.num_rounds == 3, f"expected 3 speculative rounds, got {res7.num_rounds}"
    print("  [OK] accept-all + bonus token: 9 tokens in 3 rounds, acceptance=1.0")


def test_paged_attention():
    print("== PagedAttention ==")
    cfg = DummyConfig()
    torch.manual_seed(0)
    attn = PagedAttention(cfg)
    nh, dh = cfg.attention.num_attention_heads, cfg.hidden_size // cfg.attention.num_attention_heads

    # C11: fp32 forward works; explicit bf16 cache + fp32 query also works
    bm = BlockManager(block_size=4, num_blocks=8, device="cpu")
    bm.allocate(0, 1, nh, dh)
    out = attn(torch.randn(1, 1, cfg.hidden_size), bm, [0])
    assert out.shape == (1, 1, cfg.hidden_size)
    bm_bf = BlockManager(block_size=4, num_blocks=8, device="cpu", dtype=torch.bfloat16)
    bm_bf.allocate(0, 1, nh, dh)
    out = attn(torch.randn(1, 1, cfg.hidden_size), bm_bf, [0])
    assert out.shape == (1, 1, cfg.hidden_size) and out.dtype == torch.float32
    print("  [OK] C11 dtype: fp32 default + bf16 cache w/ fp32 query")

    # M19/M20/m4: cross-block-boundary writes, vectorized read-back
    bm2 = BlockManager(block_size=4, num_blocks=8, device="cpu")
    bm2.allocate(1, 0, nh, dh)
    hseq = torch.randn(1, 6, cfg.hidden_size)  # block_size+2 tokens
    for t in range(6):
        attn(hseq[:, t:t + 1], bm2, [1])
    assert bm2.get_context_length(1) == 6
    assert len(bm2.get_block_table(1)) == 2, "expected block boundary crossing"
    kg, vg = bm2.gather_kv(1)
    exp_k = attn.k_proj(hseq[0]).view(6, nh, dh)
    exp_v = attn.v_proj(hseq[0]).view(6, nh, dh)
    assert torch.allclose(kg, exp_k, atol=1e-5), "K mismatch after boundary crossing"
    assert torch.allclose(vg, exp_v, atol=1e-5), "V mismatch after boundary crossing"
    print("  [OK] M19/M20: append-at-context_len across block boundary, values exact")

    # M21: fork + true CoW
    bm3 = BlockManager(block_size=4, num_blocks=8, device="cpu")
    bm3.allocate(10, 0, nh, dh)
    attn(torch.randn(1, 2, cfg.hidden_size), bm3, [10])  # 2 tokens, partial block
    parent_block = bm3.get_block_table(10)[0]
    kg_parent, _ = bm3.gather_kv(10)
    bm3.fork(10, 11)
    assert bm3.refcounts[parent_block] == 2
    # parent appends inside the partially-filled shared block
    h_new = torch.randn(1, 1, cfg.hidden_size)
    attn(h_new, bm3, [10])
    new_parent_block = bm3.get_block_table(10)[0]
    assert new_parent_block != parent_block, "CoW did not copy shared block"
    assert bm3.refcounts[new_parent_block] == 1 and bm3.refcounts[parent_block] == 1
    # child view unchanged: only its original 2 tokens, values intact
    kg_child, _ = bm3.gather_kv(11)
    assert torch.equal(kg_child, kg_parent), "child K/V mutated by parent write"
    assert bm3.get_context_length(11) == 2 and bm3.get_context_length(10) == 3
    # child writes don't disturb parent either
    attn(torch.randn(1, 1, cfg.hidden_size), bm3, [11])
    kg_child2, _ = bm3.gather_kv(11)
    assert kg_child2.shape[0] == 3 and torch.equal(kg_child2[:2], kg_parent)
    print("  [OK] M21: refcounted fork + copy-on-write isolation both directions")

    # free: shared blocks not double-freed; pool fully restored afterwards
    free_before = bm3.num_free_blocks()
    bm3.free(10)
    assert bm3.num_free_blocks() == free_before + 1  # only parent's private block
    bm3.free(11)
    assert bm3.num_free_blocks() == bm3.num_blocks
    assert not bm3.block_tables and not bm3.refcounts
    # reuse: allocate again, write, read back -- no stale data
    bm3.allocate(20, 0, nh, dh)
    h_r = torch.randn(1, 1, cfg.hidden_size)
    attn(h_r, bm3, [20])
    kg_r, _ = bm3.gather_kv(20)
    assert torch.allclose(kg_r[0], attn.k_proj(h_r[0, 0]).view(nh, dh), atol=1e-5)
    print("  [OK] free/reuse: no double-free, no stale-data crosstalk")

    # m5: cache shape fixed at first allocation
    try:
        bm3.allocate(21, 1, nh + 1, dh)
        raise AssertionError("expected ValueError on shape change")
    except ValueError:
        print("  [OK] m5: shape mismatch raises ValueError")


def test_vllm_engine():
    print("== VLLMEngine ==")
    cfg, model, _ = make_model_and_mtp(0)
    engine = VLLMEngine(model, cfg, block_size=4, max_num_blocks=16)
    p1 = [5]
    p2 = [10, 20, 30]
    r1 = engine.add_request(p1, max_new_tokens=6, temperature=0.0, eos_token_id=None)
    r2 = engine.add_request(p2, max_new_tokens=6, temperature=0.0, eos_token_id=None)
    results = engine.run()

    ref1 = model.generate(torch.tensor([p1]), 6, temperature=0)[0, len(p1):].tolist()
    ref2 = model.generate(torch.tensor([p2]), 6, temperature=0)[0, len(p2):].tolist()
    assert results[r1] == ref1, f"{results[r1]} != {ref1}"
    assert results[r2] == ref2, f"{results[r2]} != {ref2}"
    print("  [OK] two unequal-length requests match plain greedy generation")

    # no block-manager leakage after completion
    assert engine.block_manager.num_free_blocks() == engine.block_manager.num_blocks
    assert not engine.block_manager.block_tables
    print("  [OK] BlockManager fully freed (no leak)")

    # eos early stop: set eos to the 3rd token of a known greedy trajectory
    engine2 = VLLMEngine(model, cfg, block_size=4, max_num_blocks=16)
    traj = model.generate(torch.tensor([p2]), 6, temperature=0)[0, len(p2):].tolist()
    r3 = engine2.add_request(p2, max_new_tokens=6, temperature=0.0,
                             eos_token_id=traj[2])
    res3 = engine2.run()[r3]
    assert res3 == traj[:3], res3
    assert engine2.block_manager.num_free_blocks() == engine2.block_manager.num_blocks
    print("  [OK] eos stop + cleanup")

    # temperature>0 smoke
    engine3 = VLLMEngine(model, cfg, block_size=4, max_num_blocks=16)
    r4 = engine3.add_request(p1, max_new_tokens=5, temperature=1.0)
    assert len(engine3.run()[r4]) == 5
    print("  [OK] temperature>0 sampling path")

    # HPA: unit-consistent ratios, direction from current load
    mon = GPUMonitor(export_prometheus=False)
    assert mon.get_hpa_recommendation()["action"] == "hold"  # no data
    mon.metrics["gpu_utilization"] = [10.0, 12.0]
    mon.metrics["gpu_memory_ratio"] = [0.20, 0.25]
    rec = mon.get_hpa_recommendation(current_replicas=4)
    assert rec["action"] == "scale_down" and rec["replicas"] < 4, rec
    mon.metrics["gpu_utilization"] = [95.0, 97.0]
    mon.metrics["gpu_memory_ratio"] = [0.90, 0.92]
    rec = mon.get_hpa_recommendation(current_replicas=2)
    assert rec["action"] == "scale_up" and rec["replicas"] > 2, rec
    mon.metrics["gpu_utilization"] = [50.0]
    mon.metrics["gpu_memory_ratio"] = [0.50]
    rec = mon.get_hpa_recommendation(current_replicas=3)
    assert rec["action"] == "hold" and rec["replicas"] == 3, rec
    print("  [OK] HPA: scale_down / scale_up / hold directions sane")

    # Request protocol
    req = Request(request_id=0, prompt_token_ids=[1, 2], max_new_tokens=2, eos_token_id=9)
    assert req.get_input_ids() == [1, 2] and not req.is_done()
    req.append_token(3)
    assert req.get_input_ids() == [1, 2, 3] and not req.is_done()
    req.append_token(9)
    assert req.is_done()
    print("  [OK] Request dataclass protocol")


if __name__ == "__main__":
    test_mtp_training_shift()
    test_mtp_generate()
    test_paged_attention()
    test_vllm_engine()
    print("\nALL INFERENCE SELF-TESTS PASSED")
