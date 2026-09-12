"""Self-tests for helioslm v5.3 -> v5.4 inference fixes (subagent scratch)."""
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output")

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.inference.vllm_engine import VLLMEngine
from helioslm_v5.src.inference.mtp import MTPDecoder, MTPModule


def _lite_model(seed):
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(seed)
    config = HeliosLMv5Config(size="lite")
    return HeliosLMv5(config).eval(), config


def check(name, cond, msg=""):
    assert cond, f"[FAIL] {name}: {msg}"
    print(f"[ok] {name}")


# ---------------------------------------------------------------- 1
# max_new_tokens=0 -> step() returns {} ; run() path unaffected
def test_zero_max_new():
    model, config = _lite_model(500)
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    rid0 = engine.add_request([1, 2, 3], max_new_tokens=0)
    out = engine.step()
    check("zero-max step returns {}", out == {}, f"got {out}")
    req = engine.finished_requests[0]
    check("zero-max retired, no tokens", req.request_id == rid0
          and req.generated_token_ids == [] and req.finished)
    check("zero-max blocks freed",
          len(engine.block_manager.block_tables) == 0
          and engine.block_manager.num_free_blocks() == 64)

    # mixed batch: zero-max request must not consume a prefill slot
    model2, config2 = _lite_model(501)
    eng = VLLMEngine(model2, config2, block_size=4, max_num_blocks=64)
    r0 = eng.add_request([5, 6], max_new_tokens=0)
    r1 = eng.add_request([42, 900], max_new_tokens=4, temperature=0.0)
    out1 = eng.step()
    check("mixed step: only live request emits", set(out1) == {r1},
          f"got {out1}")
    results = eng.run()
    check("run(): zero-max gives empty list", results[r0] == [])
    ref = model2.generate(torch.tensor([[42, 900]]), max_new_tokens=4,
                          temperature=0)[0, 2:].tolist()
    check("run(): live request matches greedy", results[r1] == ref,
          f"{results[r1]} vs {ref}")
    check("mixed: no block leak",
          eng.block_manager.num_free_blocks() == 64
          and not eng.block_manager.block_tables)

    # pure run() path with only a zero-max request (regression)
    model3, config3 = _lite_model(502)
    eng3 = VLLMEngine(model3, config3, block_size=4, max_num_blocks=64)
    rz = eng3.add_request([7, 8, 9], max_new_tokens=0)
    res3 = eng3.run()
    check("run() zero-max only", res3 == {rz: []}, f"got {res3}")


# ---------------------------------------------------------------- 2
# forward exception -> failed request freed, engine keeps serving
class _FailOn(nn.Module):
    """Delegate module that raises on selected forward call indices."""

    def __init__(self, inner, fail_calls):
        super().__init__()
        self.inner = inner
        self.calls = 0
        self.fail_calls = set(fail_calls)

    def forward(self, *a, **k):
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError("injected forward failure")
        return self.inner(*a, **k)


def test_forward_exception_reclaim():
    # (a) prefill failure
    model, config = _lite_model(503)
    failing = _FailOn(model, fail_calls={1})  # first prefill raises
    eng = VLLMEngine(failing, config, block_size=4, max_num_blocks=64)
    r_bad = eng.add_request([5, 100, 200, 7], max_new_tokens=4)
    r_ok = eng.add_request([42, 900], max_new_tokens=4, temperature=0.0)
    try:
        eng.step()
        raise AssertionError("step should have raised")
    except RuntimeError as e:
        check("prefill failure re-raised", "injected" in str(e))
    bad = eng.finished_requests[0]
    check("failed request error state", bad.request_id == r_bad
          and bad.error and "RuntimeError" in bad.error and bad.finished)
    check("failed request not running", bad not in eng.running_requests
          and r_bad not in eng.block_manager.block_tables)
    ok_req = eng.running_requests[0]
    used = eng.block_manager.num_free_blocks()
    check("survivor still holds exactly its blocks", used == 63,
          f"free={used}/64")  # watermark 4 -> 1 block held by survivor
    # engine keeps serving the other request to completion
    results = eng.run()
    ref = model.generate(torch.tensor([[42, 900]]), max_new_tokens=4,
                         temperature=0)[0, 2:].tolist()
    check("post-failure run matches greedy", results[r_ok] == ref,
          f"{results[r_ok]} vs {ref}")
    check("post-failure: no block leak",
          eng.block_manager.num_free_blocks() == 64
          and not eng.block_manager.block_tables)

    # (b) decode-batch failure (whole group retired, blocks freed)
    model2, config2 = _lite_model(504)
    # calls: 1 = prefill req1, 2 = prefill req2, 3 = first decode batch
    failing2 = _FailOn(model2, fail_calls={3})
    eng2 = VLLMEngine(failing2, config2, block_size=4, max_num_blocks=64)
    eng2.add_request([1, 2, 3], max_new_tokens=4)
    eng2.add_request([4, 5, 6], max_new_tokens=4)
    # First step: prefill req1 (call 1), prefill req2 (call 2), then the
    # freshly prefilled rows join the decode batch in the SAME step
    # (call 3 -> raises).
    try:
        eng2.step()
        raise AssertionError("decode step should have raised")
    except RuntimeError:
        pass
    check("decode failure: both rows errored",
          len(eng2.finished_requests) == 2
          and all(r.error for r in eng2.finished_requests))
    check("decode failure: all blocks freed",
          eng2.block_manager.num_free_blocks() == 64
          and not eng2.block_manager.block_tables)
    # engine still usable afterwards
    rid = eng2.add_request([42, 900], max_new_tokens=3, temperature=0.0)
    res = eng2.run()
    ref2 = model2.generate(torch.tensor([[42, 900]]), max_new_tokens=3,
                           temperature=0)[0, 2:].tolist()
    check("engine reusable after decode failure", res[rid] == ref2)
    check("final: no leak", eng2.block_manager.num_free_blocks() == 64)


# ---------------------------------------------------------------- 3
# MTP generate restores module modes (success + exception paths)
def test_mtp_mode_restore():
    model, config = _lite_model(505)
    model.train()
    assert model.training and all(m.training for m in model.mtp_modules)
    decoder = MTPDecoder(model, model.mtp_modules, config)
    ids = torch.randint(3, config.vocab_size, (1, 6))
    res = decoder.generate(ids, max_new_tokens=4, temperature=0)
    assert res.sequences.shape[0] == 1
    check("decoder.generate restores train mode",
          model.training and all(m.training for m in model.mtp_modules))

    # exception path
    orig = model.forward

    def boom(*a, **k):
        raise RuntimeError("mtp boom")

    model.forward = boom
    try:
        decoder.generate(ids, max_new_tokens=4, temperature=0)
        raise AssertionError("should raise")
    except RuntimeError:
        pass
    finally:
        model.forward = orig
    check("decoder.generate restores mode on exception",
          model.training and all(m.training for m in model.mtp_modules))

    # eval-mode caller: must stay eval
    model.eval()
    decoder.generate(ids, max_new_tokens=4, temperature=0)
    check("decoder.generate preserves eval mode",
          not model.training and not any(m.training for m in model.mtp_modules))

    # MTPModule.generate: mode restore incl. exception path
    model.train()
    m = model.mtp_modules[0]
    h = torch.randn(1, 1, config.hidden_size)
    tok = torch.tensor([[3]])
    m.generate(h, tok, temperature=0)
    check("MTPModule.generate restores train mode", m.training)
    try:
        m.generate(h, None, temperature=0)  # missing token -> ValueError
        raise AssertionError("should raise")
    except ValueError:
        pass
    check("MTPModule.generate restores mode on exception", m.training)


# ---------------------------------------------------------------- 4
# batched == per-request greedy; no leaks; LRU behavior intact
def test_batched_greedy_and_lru():
    model, config = _lite_model(506)
    prompts = [[5, 100, 200, 7], [42, 900], [1, 2, 3, 4, 5, 6]]
    eng = VLLMEngine(model, config, block_size=4, max_num_blocks=128)
    eng._pad_caches_max = 2  # force LRU eviction pressure
    rids = [eng.add_request(p, max_new_tokens=5, temperature=0.0)
            for p in prompts]
    results = eng.run()
    for p, rid in zip(prompts, rids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=5,
                             temperature=0)[0, len(p):].tolist()
        got = results[rid]
        check(f"batched==greedy rid={rid}", ref[:len(got)] == got,
              f"{got} vs {ref}")
    check("no block leak (batched)",
          eng.block_manager.num_free_blocks() == 128
          and not eng.block_manager.block_tables)
    check("LRU memo bounded", len(eng._pad_caches) <= 2,
          f"memo size {len(eng._pad_caches)}")
    check("LRU memo held distinct pad lengths",
          all(isinstance(k, int) and k > 0 for k in eng._pad_caches))


# ---------------------------------------------------------------- O5
def test_config_defaults():
    model, config = _lite_model(507)
    config.paged_attention.block_size = 8
    config.paged_attention.num_blocks = 32
    eng = VLLMEngine(model, config)
    check("block_size from config", eng.block_size == 8
          and eng.block_manager.block_size == 8)
    check("num_blocks from config",
          eng.block_manager.num_blocks == 32)
    check("max_batch_size default 32", eng.max_batch_size == 32)
    check("enabled flag recorded", eng.paged_attention_enabled is True)
    eng2 = VLLMEngine(model, config, block_size=4, max_num_blocks=64,
                      max_batch_size=2)
    check("explicit args win", eng2.block_size == 4
          and eng2.block_manager.num_blocks == 64
          and eng2.max_batch_size == 2)


# ---------------------------------------------------------------- O11
def test_sampling_reproducible():
    model, config = _lite_model(508)
    decoder = MTPDecoder(model, model.mtp_modules, config)
    ids = torch.randint(3, config.vocab_size, (2, 6))
    outs = []
    for _ in range(2):
        torch.manual_seed(999)
        res = decoder.generate(ids, max_new_tokens=8, temperature=0.9,
                               top_p=0.9)
        outs.append(res.sequences.clone())
    check("torch.manual_seed reproduces MTP sampling",
          torch.equal(outs[0], outs[1]))

    # dedicated generator also reproducible (the acceptance draw uses
    # decoder.generator; multinomial draws still use the global stream,
    # so seed both to align the two runs)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    seqs = []
    for g in (g1, g2):
        torch.manual_seed(555)
        decoder.generator = g
        seqs.append(decoder.generate(ids, max_new_tokens=8,
                                     temperature=0.9, top_p=0.9).sequences.clone())
    decoder.generator = None
    check("explicit torch.Generator reproducible", torch.equal(seqs[0], seqs[1]))


if __name__ == "__main__":
    test_zero_max_new()
    test_forward_exception_reclaim()
    test_mtp_mode_restore()
    test_batched_greedy_and_lru()
    test_config_defaults()
    test_sampling_reproducible()
    print("ALL SELF-TESTS PASSED")
