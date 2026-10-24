"""HeliosLM v5 test cases — paged attention + vLLM-style serving engine.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_lite_model, _pass)


# ----------------------------------------------------------------------
# PagedAttention: fp32 write/read across blocks, CoW fork, no leaks
# ----------------------------------------------------------------------
def test_paged_attention():
    from evastar.src.inference.paged_attention import (
        BlockManager, PagedAttention)

    config = EvastarV5Config(size="lite")
    H = config.attention.num_attention_heads
    D = config.hidden_size // H  # PagedAttention head_dim = hidden / heads

    # 1) Cross-block write -> read back value-for-value.
    bm = BlockManager(block_size=4, num_blocks=16, device="cpu",
                      dtype=torch.float32)
    bm.allocate(0, 3, H, D)
    bm.append_tokens(0, 5)  # total 8 tokens = 2 full blocks (crosses boundary)
    torch.manual_seed(105)
    ks, vs = torch.randn(8, H, D), torch.randn(8, H, D)
    for i in range(8):
        bm.write_kv(0, i, ks[i], vs[i])
    k_g, v_g = bm.gather_kv(0)
    assert torch.equal(k_g, ks) and torch.equal(v_g, vs), \
        "paged K/V read-back mismatch across block boundary"

    # 2) Copy-on-write fork: writing the child must not touch the parent.
    bm.fork(0, 1)
    parent_block = bm.get_block_table(0)[0]
    assert bm.refcounts[parent_block] == 2, "fork must share blocks (refcount 2)"
    new_k, new_v = torch.randn(H, D), torch.randn(H, D)
    bm.write_kv(1, 1, new_k, new_v)  # write inside the shared block
    assert bm.get_block_table(0)[0] == parent_block, "parent block table changed"
    assert bm.get_block_table(1)[0] != parent_block, "CoW did not copy the block"
    assert bm.refcounts[parent_block] == 1, "parent refcount wrong after CoW"
    k_p, _ = bm.gather_kv(0)
    k_c, _ = bm.gather_kv(1)
    assert torch.equal(k_p[1], ks[1]), "parent data corrupted by child write"
    assert torch.equal(k_c[1], new_k) and torch.equal(k_c[0], ks[0]), \
        "child did not see its own write / lost shared prefix"

    # 3) Freeing both sequences returns every block, no leaks.
    bm.free(0)
    bm.free(1)
    assert bm.num_free_blocks() == 16 and not bm.block_tables and not bm.refcounts, \
        "block leak after free"

    # 4) PagedAttention module: fp32 end-to-end, cache dtype follows input,
    #    and output matches a manual causal-attention reference.
    torch.manual_seed(106)
    pa = PagedAttention(config).eval()
    bm2 = BlockManager(block_size=4, num_blocks=16, device="cpu")  # dtype=None
    bm2.allocate(0, 0, pa.num_heads, pa.head_dim)
    h = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        out = pa(h, bm2, [0])
    assert bm2.k_cache is not None and bm2.k_cache.dtype == torch.float32, \
        "cache dtype must follow fp32 input"
    assert out.shape == h.shape and torch.isfinite(out).all()

    # Manual reference: single-shot causal attention over the same projections.
    with torch.no_grad():
        q = pa.q_proj(h).view(1, 5, H, D).transpose(1, 2)
        k = pa.k_proj(h).view(1, 5, H, D).transpose(1, 2)
        v = pa.v_proj(h).view(1, 5, H, D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        ref = pa.o_proj((torch.softmax(scores, dim=-1) @ v)
                        .transpose(1, 2).reshape(1, 5, H * D))
    ref_diff = (out - ref).abs().max().item()
    assert ref_diff < 1e-5, f"PagedAttention vs manual causal ref: {ref_diff:.3e}"
    _pass("test_paged_attention",
          f"cross-block readback exact, CoW isolated, no leaks, "
          f"vs manual ref max|diff|={ref_diff:.2e}")


# ----------------------------------------------------------------------
# VLLMEngine: continuous batching == per-request greedy; no block leaks
# ----------------------------------------------------------------------
def test_vllm_engine():
    from evastar.src.inference.vllm_engine import VLLMEngine

    model, config = _lite_model(seed=107)
    prompts = [[5, 100, 200, 7], [42, 900]]  # unequal lengths
    max_new = 6
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    req_ids = [engine.add_request(p, max_new_tokens=max_new, temperature=0.0)
               for p in prompts]
    results = engine.run()

    for p, rid in zip(prompts, req_ids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=max_new,
                             temperature=0)
        ref_new = ref[0, len(p):].tolist()
        got = results[rid]
        # The engine stops at EOS while generate() freezes with pad; the
        # engine output must be an exact prefix of the greedy reference.
        assert ref_new[:len(got)] == got, \
            f"request {rid}: engine {got} != greedy reference {ref_new[:len(got)]}"
        assert len(got) >= 1

    bm = engine.block_manager
    assert len(bm.block_tables) == 0 and len(bm.refcounts) == 0 \
        and bm.num_free_blocks() == 64, \
        f"block leak: free={bm.num_free_blocks()}/64 tables={len(bm.block_tables)}"
    _pass("test_vllm_engine",
          "2 unequal-length requests match per-request greedy; blocks 64/64 free")


def test_batched_prefill():
    """v5.12: equal-length prompts batch through one engine prefill forward.

    Hybrid (recurrent-state) model: the engine cannot use watermark pad
    prefixes, so equal-length admission waves must produce identical output
    to solo greedy generation while sharing a single [B, L] prefill.
    """
    from evastar.src.inference.vllm_engine import VLLMEngine
    from evastar.src.model_v5 import EvastarV5

    torch.manual_seed(1721)
    config = EvastarV5Config(size="lite")
    config.hybrid_attention.enabled = True
    config.hybrid_attention.full_attention_every = 3
    config.num_hidden_layers = 4
    model = EvastarV5(config).eval()

    prompts = [[11, 22, 33, 44], [55, 66, 77, 88], [1, 2, 3]]  # 2x len-4 + 1x len-3
    max_new = 6
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    req_ids = [engine.add_request(p, max_new_tokens=max_new, temperature=0.0)
               for p in prompts]
    results = engine.run()

    for p, rid in zip(prompts, req_ids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=max_new,
                             temperature=0)
        ref_new = ref[0, len(p):].tolist()
        got = results[rid]
        assert ref_new[:len(got)] == got, \
            f"request {rid}: engine {got} != greedy reference {ref_new[:len(got)]}"
        assert len(got) >= 1
    _pass("test_batched_prefill",
          "hybrid engine: equal-length prompts batched, == solo greedy")


# ----------------------------------------------------------------------
# v5.20 code-review regression tests (review pass 2026-09-23)
# ----------------------------------------------------------------------

def test_vllm_engine_padded_batched_prefill():
    """v5.20: equal-length prompts below the admission watermark must not
    crash batched prefill (regression: prefill grouping ignored
    prompt_pad, so a padded group hit the pad==0 assert and healthy
    requests were retired as errors)."""
    from evastar.src.inference.vllm_engine import VLLMEngine

    model, config = _lite_model(seed=118)
    # one admission wave: the two short prompts share a length AND both
    # get watermark padding — the exact combination the old grouping
    # crashed on
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]]
    max_new = 5
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    req_ids = [engine.add_request(p, max_new_tokens=max_new, temperature=0.0)
               for p in prompts]
    results = engine.run()

    for p, rid in zip(prompts, req_ids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=max_new,
                             temperature=0)
        ref_new = ref[0, len(p):].tolist()
        got = results[rid]
        assert ref_new[:len(got)] == got, \
            f"request {rid}: engine {got} != greedy reference {ref_new[:len(got)]}"
        assert len(got) >= 1
    bm = engine.block_manager
    assert len(bm.block_tables) == 0 and bm.num_free_blocks() == 64, \
        f"block leak: free={bm.num_free_blocks()}/64 tables={len(bm.block_tables)}"
    _pass("test_vllm_engine_padded_batched_prefill",
          "equal-length padded group decodes == per-request greedy; no errors")
