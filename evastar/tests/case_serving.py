"""HeliosLM v5 test cases — serving stack: pools, evolver, expert streaming, toy checkpoint.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

from pathlib import Path

import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_lite_model, _pass)


def test_spec_breakeven():
    """v5.16: spec break-even instrumentation — schema, helper, MTP fields."""
    from benchmarks.bench_spec_breakeven import (best_draft_per_cache_state,
                                                 bench_expert_hit_curve)
    from evastar.src.inference.expert_store import (attach_streaming_store,
                                                        detach_streaming_store)
    from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE

    # 1) pure helper: best draft per cache state (synthetic rows, no timing)
    rows = [
        {"cache_state": "cold",  "draft_depth": 0, "tok_s_out": 1.0},
        {"cache_state": "cold",  "draft_depth": 1, "tok_s_out": 0.7},
        {"cache_state": "warm",  "draft_depth": 0, "tok_s_out": 1.0},
        {"cache_state": "warm",  "draft_depth": 1, "tok_s_out": 1.4},
    ]
    best = best_draft_per_cache_state(rows)
    assert best["cold"]["draft_depth"] == 0
    assert best["warm"]["draft_depth"] == 1

    # 2) expert-store hit-rate curve rows carry the full P3-compatible schema
    curve = bench_expert_hit_curve(budgets=[8])
    r = curve[0]
    for k in ("resident_experts", "budget_bytes", "hit_rate", "misses",
              "evictions", "ram_saving_vs_dense"):
        assert k in r, f"missing field {k}"
    assert 0.0 <= r["hit_rate"] <= 1.0
    assert r["misses"] > 0 and r["evictions"] == 0  # full residency: no churn

    # 3) MTP acceptance field present and in range on the toy checkpoint
    ck = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if ck.exists():
        from evastar.src.inference.mtp import MTPDecoder
        from evastar.src.model_v5 import EvastarV5
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        model = EvastarV5(EvastarV5Config(size="lite"))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        dec = MTPDecoder(model, model.mtp_modules, EvastarV5Config(size="lite"))
        ids = torch.tensor([[ord(c) for c in "HeliosLM"]])
        with torch.no_grad():
            res = dec.generate(ids, max_new_tokens=16, temperature=0)
        assert 0.0 <= res.acceptance_rate <= 1.0
    _pass("test_spec_breakeven",
          "P3-schema rows, best-draft helper, acceptance field in range")


def test_score_stream():
    """v5.17: standalone token-stream scorer — engine-agnostic quality gate."""
    from evastar.eval.score_stream import score_record
    from evastar.src.model_v5 import EvastarV5

    ck = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if not ck.exists():
        _pass("test_score_stream", "skipped (checkpoint not built)")
        return
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    model = EvastarV5(EvastarV5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = ckpt["vocab_size"]

    prompt = "HeliosLM"
    ids = torch.tensor([[ord(c) for c in prompt]])
    with torch.no_grad():
        greedy = "".join(chr(int(t)) for t in
                         model.generate(ids, max_new_tokens=16,
                                        temperature=0)[0, len(prompt):])
    import random as _r
    _r.seed(5)
    noise = "".join(chr(_r.randint(97, 122)) for _ in range(16))

    s_greedy = score_record(model, prompt, greedy, vocab)
    s_noise = score_record(model, prompt, noise, vocab)
    assert s_greedy["nll_per_token"] < s_noise["nll_per_token"], \
        "model's own greedy output must score better than random chars"
    assert s_greedy["n_tokens"] == 16 and s_greedy["ppl"] > 0
    assert set(s_greedy) >= {"sum_logprob", "n_tokens", "nll_per_token", "ppl"}
    _pass("test_score_stream",
          f"greedy nll={s_greedy['nll_per_token']:.2f} < noise "
          f"nll={s_noise['nll_per_token']:.2f}; schema complete")


def test_prefix_pool():
    """v5.18: content-addressed KV prefix pool — pooled == from-scratch."""
    from evastar.src.inference.prefix_pool import PrefixPool
    from evastar.src.model_v5 import EvastarV5

    ck = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if not ck.exists():
        _pass("test_prefix_pool", "skipped (checkpoint not built)")
        return
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    model = EvastarV5(EvastarV5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def ref(prompt, n=12):
        ids = torch.tensor([[ord(c) for c in prompt]])
        return model.generate(ids, max_new_tokens=n, temperature=0)

    def enc(prompt):
        return [ord(c) for c in prompt]

    A = "# HeliosLM config"      # 17 chars -> one full 16-block + remainder
    B = " import torch"

    # 1) full + partial prefix hit: pooled == from-scratch, bitwise
    pool = PrefixPool(model, block_size=16, max_blocks=16,
                      fingerprint="v5.18/fp32")
    pool.store(enc(A))
    out_hit = pool.generate(enc(A + B), max_new_tokens=12, temperature=0)
    assert torch.equal(out_hit, ref(A + B)), "prefix hit must be bit-exact"
    assert pool.stats()["hits"] >= 1

    # 2) fingerprint mismatch: same tokens, other fingerprint -> miss,
    #    and the result is STILL identical (pool can never change answers)
    pool2 = PrefixPool(model, block_size=16, max_blocks=16,
                       fingerprint="v5.18/int4")
    out_fp = pool2.generate(enc(A + B), max_new_tokens=12, temperature=0)
    assert torch.equal(out_fp, ref(A + B))
    assert pool2.stats()["misses"] >= 1

    # 3) eviction: one-block pool evicts A; re-request still correct
    pool3 = PrefixPool(model, block_size=16, max_blocks=1,
                       fingerprint="v5.18/fp32")
    pool3.generate(enc(A), max_new_tokens=4, temperature=0)
    pool3.generate(enc("class HeliosLM:"), max_new_tokens=4, temperature=0)
    assert pool3.stats()["evictions"] >= 1
    out_ev = pool3.generate(enc(A + B), max_new_tokens=12, temperature=0)
    assert torch.equal(out_ev, ref(A + B)), "post-eviction must be bit-exact"

    st = pool.stats()
    assert st["hit_tokens"] > 0 and 0.0 <= st["hit_rate"] <= 1.0
    _pass("test_prefix_pool",
          f"full/partial hit + fingerprint miss + eviction all bit-exact; "
          f"hit_tokens={st['hit_tokens']}")


def test_prefix_pool_high_token_ids():
    """v5.20: pool hashing must accept token ids >= 256 (regression:
    bytes(int(t)) raised ValueError, so any non-ASCII id crashed store —
    and generate() stores unconditionally)."""
    from evastar.src.inference.prefix_pool import PrefixPool

    model, config = _lite_model(seed=119)
    pool = PrefixPool(model, block_size=8, max_blocks=32,
                      fingerprint="v5.20/fp32")
    prompt = [300, 301, 302, 303, 400, 500, 600, 700, 800]  # ids >= 256
    ref = model.generate(torch.tensor([prompt]), max_new_tokens=8,
                         temperature=0)
    out = pool.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(out, ref), \
        "pooled generate with ids >= 256 must be bit-exact"
    # repeat: the stored blocks (prompt + generated ids) must hash fine too
    out2 = pool.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(out2, ref)
    assert pool.stats()["hits"] >= 1
    _pass("test_prefix_pool_high_token_ids",
          "ids>=256 store/lookup round-trip bit-exact")


def test_harness_evolver():
    """v5.19: oracle-gated serving-harness evolution (ModularRSI-style)."""
    from evastar.src.inference.harness_evolver import HarnessEvolver
    from evastar.src.model_v5 import EvastarV5

    ck = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if not ck.exists():
        _pass("test_harness_evolver", "skipped (checkpoint not built)")
        return
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    model = EvastarV5(EvastarV5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # shared-prefix workload: pool=on should be profitable; overlapping
    # prompts mirror the agentic multi-turn setting the paper targets
    base = "# HeliosLM v5.19 config"
    workload = [[ord(c) for c in base + suffix] for suffix in
                (" import torch", " attention", " moe router", " engine")]

    ev = HarnessEvolver(model, workload, max_new=6)
    report = ev.evolve()

    # invariants: the oracle can never accept a math-changing config
    assert report["gate_rejects"] == 0, \
        "temp-0 invariance: cache/draft/tier must not change greedy outputs"
    for row in report["contrastive"]:
        assert row["gate_passed"], "every evaluated config must pass the gate"

    # structure + the search actually evaluated candidates
    assert report["evaluations"] == 3
    assert report["cost_best"] <= report["cost_baseline"]
    assert report["speedup"] >= 1.0
    assert report["best_config"]["draft"] in (True, False)
    # draft was accepted or rejected on data, both fine — but if the
    # workload shares prefixes, the pool should have been accepted
    labels = [a["config"] for a in report["accepted"]]
    assert "pool:on" in labels, "shared-prefix workload must profit from pool"
    _pass("test_harness_evolver",
          f"best={report['best_label'] or 'baseline'} "
          f"speedup={report['speedup']}x accepted={labels}")


def test_evolver_pareto_adaptation():
    """v5.20: memory axis + Pareto frontier + cross-workload adaptation."""
    from evastar.src.inference.harness_evolver import (AdaptationLoop,
                                                           HarnessEvolver)
    from evastar.src.model_v5 import EvastarV5

    ck = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if not ck.exists():
        _pass("test_evolver_pareto_adaptation", "skipped (checkpoint not built)")
        return
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    model = EvastarV5(EvastarV5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    base = "# HeliosLM v5.20 config"
    warm_wl = [[ord(c) for c in base + suf] for suf in
               (" import torch", " attention", " moe router", " engine")]

    # 1) Pareto frontier over evaluated configs: all entries non-dominated
    ev = HarnessEvolver(model, warm_wl, max_new=6)
    rep = ev.evolve()
    assert rep["gate_rejects"] == 0
    for p in rep["pareto"]:
        assert p["gate"]
        for q in rep["pareto"]:
            assert not (q is not p and q["cost"] <= p["cost"]
                        and q["memory"] <= p["memory"]
                        and (q["cost"] < p["cost"] or q["memory"] < p["memory"]))

    # 2) memory budget: zero budget rejects pool, draft (memory-free) survives
    ev_m = HarnessEvolver(model, warm_wl, max_new=6, memory_budget=0.0)
    rep_m = ev_m.evolve()
    assert all(a["module"] != "pool" for a in rep_m["accepted"]), \
        "zero memory budget must reject the pool module"
    for row in rep_m["contrastive"]:
        if row["module"] == "pool":
            assert row["memory"] > 0 and not row["memory_budget_ok"]

    # 3) adaptation: source (rich prefix overlap, generous memory) ->
    #    target (no shared prefixes, zero memory) drops the pool module
    src = HarnessEvolver(model, warm_wl, max_new=6, memory_budget=None)
    src_rep = src.evolve()
    assert src_rep["best_config"]["pool"], "source workload must profit from pool"
    cold_wl = [[ord(c) for c in w] for w in
               ("xyz alpha", "qrs beta", "mno gamma", "jkl delta")]
    loop = AdaptationLoop(model, rounds=1)
    tr = loop.transfer(src_rep, cold_wl, max_new=6, memory_budget=0.0)
    assert "pool" in tr["dropped_modules"], \
        "pool must be dropped on a prefix-free, zero-memory target"
    assert tr["adapted_config"]["pool"] is False
    assert tr["decisions"][0]["gate_rejects"] == 0
    _pass("test_evolver_pareto_adaptation",
          f"pareto={len(rep['pareto'])}pts, memory-budget rejects pool, "
          f"adaptation dropped={tr['dropped_modules']}")


def test_expert_streaming():
    """v5.15: disk-tier expert store — streaming forward is bit-exact vs dense."""
    from evastar.src.inference.expert_store import (attach_streaming_store,
                                                        detach_streaming_store)
    from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE

    def small_moe():
        cfg = EvastarV5Config(size="lite")
        cfg.hidden_size = 32
        cfg.moe.num_experts = 8
        cfg.moe.num_shared_experts = 0
        cfg.moe.num_activated_experts = 2
        cfg.moe.expert_hidden_size = 64
        m = DeviceLimitedMoE(cfg)
        m.eval()
        return m

    torch.manual_seed(1515)
    moe = small_moe()
    x = torch.randn(2, 5, 32)
    dense = moe(x)
    ref_state = {k: v.clone() for k, v in moe.state_dict().items()}

    # full working set resident: repeat pass must hit
    store = attach_streaming_store(moe, budget_bytes=8 * 64 * 32 * 3 * 4)
    stream = moe(x)
    assert torch.equal(dense, stream), "streaming forward must be bit-exact"
    assert store.misses > 0, "experts must page in from disk"
    hits_after_first = store.hits
    _ = moe(x)   # same routing -> resident hits
    assert store.hits > hits_after_first, "repeat pass should hit residents"
    detach_streaming_store(moe, store)

    # eviction path: tiny budget forces churn; correctness must survive
    moe2 = small_moe()
    moe2.load_state_dict(ref_state)   # captured pre-attach (attached list hides expert keys)
    moe2.eval()
    st = attach_streaming_store(moe2, budget_bytes=64 * 32 * 3 * 4 + 256)
    for trial in range(6):
        xt = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(trial))
        out = moe2(xt)
        assert out.shape == xt.shape
    assert st.evictions > 0, "tiny budget must evict"
    detach_streaming_store(moe2, st)   # no-eviction reload
    xt = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(99))
    dense_xt = moe2(xt)
    st2 = attach_streaming_store(moe2, budget_bytes=64 * 32 * 3 * 4 + 256)
    stream_xt = moe2(xt)
    detach_streaming_store(moe2, st2)
    assert torch.equal(dense_xt, stream_xt), "eviction path must stay bit-exact"
    _pass("test_expert_streaming",
          f"bit-exact, hits={st.hits}, misses={st.misses}, "
          f"evictions={st.evictions}, file={st.total_bytes // 1024}KB")


def test_toy_checkpoint():
    """v5.13: shipped toy checkpoint loads, generates deterministically."""
    from evastar.src.model_v5 import EvastarV5

    ckpt_path = Path(__file__).resolve().parents[2] / "checkpoints" / "toy_v5.13.pt"
    if not ckpt_path.exists():
        _pass("test_toy_checkpoint", "skipped (checkpoint not built)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["tokenizer"] == "char-ascii"
    model = EvastarV5(EvastarV5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ids = torch.tensor([[ord(c) for c in "HeliosLM"]])
    a = model.generate(ids, max_new_tokens=24, temperature=0)
    b = model.generate(ids, max_new_tokens=24, temperature=0)
    assert torch.equal(a, b), "greedy generation must be deterministic"
    txt = "".join(chr(int(t)) for t in a[0, ids.shape[1]:])
    assert len(txt.strip()) > 0 and any(c.isalpha() for c in txt), \
        f"degenerate output: {txt!r}"
    assert ckpt["meta"]["val_loss"] < 3.5, "toy val loss regressed"
    _pass("test_toy_checkpoint",
          f"deterministic, val_loss={ckpt['meta']['val_loss']:.2f}")
