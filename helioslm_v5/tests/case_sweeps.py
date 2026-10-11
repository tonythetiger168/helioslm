"""HeliosLM v5 test cases — sparse-MTP / distill / engine sweeps (checkpoint-gated).

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

from pathlib import Path

import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_REPO_ROOT, _expect_raises, _pass)


# ----------------------------------------------------------------------
# limitations round 2026-10-06: sparse top-k x MTP end-to-end acceptance
# sweep — the axis the v5.16 break-even sweep (draft x cache state) does
# not cover. Oracle: k >= kv_len must match dense draft=1 bitwise (ids
# AND acceptance); k < kv_len reports overlap as data, never claims parity.
# ----------------------------------------------------------------------
def test_sparse_mtp_acceptance_sweep():
    ckpt = str(_REPO_ROOT / "checkpoints" / "toy_v5.13.pt")
    if not Path(ckpt).exists():
        _pass("test_sparse_mtp_acceptance_sweep",
              "skipped (checkpoint not built)")
        return
    from benchmarks.bench_sparse_mtp import SCHEMA_KEYS, sweep_sparse_mtp

    rows, details = sweep_sparse_mtp(ckpt, ks=[4, 10**6], max_new=8,
                                     iters=2)

    # Schema completeness and row count: 2 dense baselines + 2 sparse rows.
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    for r in rows:
        assert tuple(r.keys()) == SCHEMA_KEYS, (
            f"row keys diverge from the P3-superset schema: "
            f"{tuple(r.keys())}")

    by_k = {r["sparse_top_k"]: r for r in rows if r["draft_depth"] == 1}
    dense = by_k[None]
    big = by_k[10**6]
    sparse = by_k[4]

    # k >= kv_len oracle: outputs AND acceptance bitwise-equal to dense.
    for i in range(2):
        assert details[(10**6, i)] == details[(None, i)], (
            f"k=10**6 prompt {i} diverged from dense — the k >= kv_len "
            "bit-identity oracle is broken")
    assert big["acceptance"] == dense["acceptance"], (
        "k >= kv_len acceptance must equal dense exactly")
    assert big["output_overlap_vs_dense"] == 1.0

    # Sparse row: measured values are in range and finite. Measured on
    # this fixture (2 prompts x 8 tokens): overlap 0.625 — real
    # divergence from dense, reported as data rather than hidden (the
    # 1-prompt smoke run happened to be 1.0 on this checkpoint's large
    # greedy margins; neither extreme is guaranteed). The assertion is
    # range-only; any future drift shows up in the JSONL, not as a test
    # failure.
    assert 0.0 <= sparse["acceptance"] <= 1.0
    assert 0.0 <= sparse["output_overlap_vs_dense"] <= 1.0
    assert sparse["tok_s_out"] > 0 and sparse["n_tokens"] == 8
    assert dense["gain_vs_dense_draft0"] is not None

    # Determinism: a repeated run produces identical ids and acceptance
    # (wall-clock fields excluded by construction — they are rounded
    # measurements, not part of the oracle).
    rows2, details2 = sweep_sparse_mtp(ckpt, ks=[4], max_new=8, iters=2)
    for i in range(2):
        assert details2[(4, i)] == details[(4, i)], (
            "sparse MTP sweep is not deterministic")
    assert rows2[2]["acceptance"] == sparse["acceptance"]

    # Loud errors: missing checkpoint, bad k, learned indexer on a
    # checkpoint without indexer weights.
    _expect_raises(FileNotFoundError,
                   lambda: sweep_sparse_mtp(str(_REPO_ROOT / "nope.pt"),
                                            ks=[4], max_new=4, iters=1),
                   "missing checkpoint")
    _expect_raises(ValueError,
                   lambda: sweep_sparse_mtp(ckpt, ks=[0], max_new=4,
                                            iters=1),
                   "sparse k = 0")
    _expect_raises(ValueError,
                   lambda: sweep_sparse_mtp(ckpt, ks=[4], max_new=4,
                                            iters=1, indexer="magic"),
                   "unknown indexer")
    _expect_raises(ValueError,
                   lambda: sweep_sparse_mtp(ckpt, ks=[4], max_new=4,
                                            iters=1, indexer="learned"),
                   "learned indexer w/o checkpoint weights")
    _pass("test_sparse_mtp_acceptance_sweep",
          f"schema ok, k>=L bitwise (ids+acceptance), dense acc "
          f"{dense['acceptance']}, k=4 acc {sparse['acceptance']} overlap "
          f"{sparse['output_overlap_vs_dense']}, deterministic, 4 loud "
          "guards")


# ----------------------------------------------------------------------
# limitations round 2026-10-07: distill-train the learned lightning
# indexer on the toy checkpoint so indexer="learned" sweep rows become
# measurable end to end (they were refused loudly before — every
# checkpoint predated the indexer). Oracles: distill loss/overlap improve
# by measured margins; trunk weights bit-preserved; k >= L bitwise.
# ----------------------------------------------------------------------
def test_indexer_distill_learned_sweep():
    ckpt_path = _REPO_ROOT / "checkpoints" / "toy_v5.13.pt"
    if not ckpt_path.exists():
        _pass("test_indexer_distill_learned_sweep",
              "skipped (checkpoint not built)")
        return
    import os
    import tempfile

    from benchmarks.bench_sparse_mtp import SCHEMA_KEYS, sweep_sparse_mtp
    from examples.train_indexer_distill import (build_corpus_ids,
                                                distill_indexer,
                                                load_with_new_indexer)
    from helioslm_v5.src.model_v5 import HeliosLMv5

    corpus = build_corpus_ids(1024)
    ckpt = torch.load(str(_REPO_ROOT / "checkpoints" / "toy_v5.13.pt"),
                      map_location="cpu", weights_only=False)
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.sparse_top_k = 4
    cfg.attention.sparse_indexer = "learned"
    torch.manual_seed(1234)
    model = HeliosLMv5(cfg).eval()
    n_missing = load_with_new_indexer(model, ckpt["state_dict"])
    assert n_missing == 6, (
        f"2 layers x (q_proj, k_proj, head_weights) = 6 indexer keys, "
        f"got {n_missing}")

    # Trunk bit-preservation probe (distill must freeze the trunk).
    ids = torch.tensor([[ord(c) for c in "HeliosLM"]])
    with torch.no_grad():
        out_pre = model(ids)[0]

    # Untrained learned-indexer baseline sweep (the "noise" row).
    payload = dict(ckpt)
    payload["state_dict"] = model.state_dict()
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(payload, tmp)
    rows_pre, _ = sweep_sparse_mtp(tmp, ks=[4], max_new=8, iters=2,
                                   indexer="learned")

    stats = distill_indexer(model, corpus, steps=400, top_k=4, seed=1234)
    # Measured on this fixture: loss 5.13 -> 3.81 (ratio 0.74), teacher
    # top-4 overlap 0.165 -> 0.377 (gain +0.21), ~9 s CPU total. Bounds
    # below sit well inside the measurements. The residual overlap is
    # honest: near-tied teacher scores cap fidelity (same phenomenon as
    # the 09-21 unit-level distill).
    assert stats["layers"] == 2 and stats["steps"] == 400
    assert stats["loss_final"] < 0.85 * stats["loss_initial"], (
        f"distill loss did not drop: {stats['loss_initial']} -> "
        f"{stats['loss_final']}")
    assert (stats["teacher_overlap_final"]
            > stats["teacher_overlap_initial"] + 0.10), (
        f"teacher overlap did not rise: "
        f"{stats['teacher_overlap_initial']} -> "
        f"{stats['teacher_overlap_final']}")

    with torch.no_grad():
        out_post = model(ids)[0]
    assert torch.equal(out_pre, out_post), (
        "distill must freeze the trunk — dense logits changed")

    # The trained checkpoint now passes the learned-indexer gate and the
    # k >= L oracle stays bitwise end to end.
    payload["state_dict"] = model.state_dict()
    torch.save(payload, tmp)
    rows, details = sweep_sparse_mtp(tmp, ks=[4, 10**6], max_new=8,
                                     iters=2, indexer="learned")
    os.unlink(tmp)
    for r in rows:
        assert tuple(r.keys()) == SCHEMA_KEYS
    assert all(details[(10**6, i)] == details[(None, i)]
               for i in range(2)), "k >= L oracle broken after distill"
    by_k = {r["sparse_top_k"]: r for r in rows if r["draft_depth"] == 1}
    assert by_k[10**6]["acceptance"] == by_k[None]["acceptance"]
    learned = by_k[4]
    assert learned["indexer"] == "learned"
    assert 0.0 <= learned["acceptance"] <= 1.0
    assert 0.0 <= learned["output_overlap_vs_dense"] <= 1.0
    pre4 = [r for r in rows_pre if r["sparse_top_k"] == 4][0]
    # NOTE (honest): end-output overlap at 2 prompts x 8 tokens has 1/16
    # granularity — across fixture variants we measured pre-distill
    # 0.5625 and post-distill 0.5-0.6875, i.e. single-position noise in
    # both directions; the SENSITIVE training signal is the teacher
    # overlap asserted above, so no end-output ordering is asserted here.
    assert 0.0 <= pre4["output_overlap_vs_dense"] <= 1.0

    # Loud errors: distill without a learned indexer; trunk-mismatched
    # checkpoint (a non-indexer key missing).
    plain_cfg = HeliosLMv5Config(size="lite")
    plain = HeliosLMv5(plain_cfg)
    _expect_raises(ValueError,
                   lambda: distill_indexer(plain, corpus, steps=1),
                   "distill without learned indexer")
    corrupted = {k: v for k, v in ckpt["state_dict"].items()
                 if k != "embed_tokens.weight"}
    _expect_raises(ValueError,
                   lambda: load_with_new_indexer(model, corrupted),
                   "trunk-mismatched checkpoint")
    _pass("test_indexer_distill_learned_sweep",
          f"distill loss {stats['loss_initial']}->{stats['loss_final']}, "
          f"teacher overlap {stats['teacher_overlap_initial']}->"
          f"{stats['teacher_overlap_final']}, trunk bit-preserved, "
          f"learned sweep acc {learned['acceptance']} overlap "
          f"{learned['output_overlap_vs_dense']}, k>=L bitwise, guards ok")


# ----------------------------------------------------------------------
# limitations round 2026-10-08: sparse top-k through the vLLM-style
# SERVING engine (watermark-padded continuous batching) — the engine
# half of the residual "sparse x MTP/engine" item; the draft half stays
# in bench_sparse_mtp.py because VLLMEngine has no draft/MTP interface
# (asserted structurally below). Oracles: engine output is an exact
# prefix of the same model's solo greedy reference on EVERY row; k >= L
# bitwise vs the dense engine; overlap with k < L reported as data.
# ----------------------------------------------------------------------
def test_engine_sparse_serving():
    from benchmarks.bench_engine_sparse import (ENGINE_NAME, SCHEMA_KEYS,
                                                sweep_engine_sparse)
    from helioslm_v5.src.inference.vllm_engine import VLLMEngine

    ckpt = str(_REPO_ROOT / "checkpoints" / "toy_v5.13.pt")
    if not Path(ckpt).exists():
        _pass("test_engine_sparse_serving",
              "skipped (checkpoint not built)")
        return
    rows, details = sweep_engine_sparse(ckpt, ks=[4, 10**6], max_new=8,
                                        n_reqs=4)

    # Schema completeness and row count: 1 dense baseline + 2 sparse rows.
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
    for r in rows:
        assert tuple(r.keys()) == SCHEMA_KEYS, (
            f"row keys diverge from the engine-serving schema: "
            f"{tuple(r.keys())}")
        assert r["engine"] == ENGINE_NAME
        assert r["n_requests"] == 4
        # Unequal prompts must admit at least one request with a
        # watermark pad prefix — otherwise the padded batched decode
        # path (the point of this sweep) never fired.
        assert r["watermark_pad_requests"] >= 1, (
            "no request was watermark-padded — the padded sparse "
            "decode path was not exercised")

    # Serving-correctness oracle: engine output is an exact prefix of the
    # SAME model's solo greedy reference on EVERY row — including the
    # sparse ones (pad K/V entries are masked out of the top-k selection
    # by mla.py's masked_fill before topk, so padded batched decode must
    # stay exact). Measured on this fixture: 1.0 on all rows.
    for r in rows:
        assert r["solo_prefix_match"] == 1.0, (
            f"engine diverged from solo greedy on row k="
            f"{r['sparse_top_k']}: {r['solo_prefix_match']}")

    by_k = {r["sparse_top_k"]: r for r in rows}
    big = by_k[10**6]
    sparse = by_k[4]

    # k >= kv_len oracle: the sparse path never activates, so the k >= L
    # engine row MUST match the dense engine row on ids bitwise.
    for i in range(4):
        assert details[(10**6, i)] == details[(None, i)], (
            f"k=10**6 request {i} diverged from the dense engine — the "
            "k >= kv_len bit-identity oracle is broken")
    assert big["output_overlap_vs_dense_engine"] == 1.0

    # Sparse row: measured values are in range and finite. Measured on
    # this fixture (4 requests x 8 tokens): overlap 0.375 vs the dense
    # engine — real divergence, reported as data rather than hidden.
    # The assertion is range-only; any drift shows up in the JSONL.
    assert 0.0 <= sparse["output_overlap_vs_dense_engine"] <= 1.0
    assert sparse["tok_s_out"] > 0 and sparse["n_tokens"] == 8
    assert sparse["gain_vs_dense_engine"] is not None

    # Determinism: a repeated run produces identical ids and overlap
    # (wall-clock fields excluded by construction).
    rows2, details2 = sweep_engine_sparse(ckpt, ks=[4], max_new=8,
                                          n_reqs=4)
    for i in range(4):
        assert details2[(4, i)] == details[(4, i)], (
            "engine sparse sweep is not deterministic")
    assert (rows2[1]["output_overlap_vs_dense_engine"]
            == sparse["output_overlap_vs_dense_engine"])

    # Draft-path finding (loud documentation, 2026-10-08): VLLMEngine
    # has NO draft/MTP/speculative interface — step() runs one trunk
    # forward per cache-length group. A combined engine x draft
    # measurement would require adding a draft path to the engine
    # first; until then the draft half of the residual item lives in
    # bench_sparse_mtp.py. Asserted structurally so this note cannot
    # silently go stale if the engine later grows speculative decode.
    draftish = [a for a in dir(VLLMEngine)
                if any(s in a.lower() for s in ("draft", "speculat", "mtp"))]
    assert not draftish, (
        f"VLLMEngine gained a draft-like interface {draftish}: the "
        "engine-half measurement can now include drafts — update "
        "bench_engine_sparse.py and this note")

    # Loud errors: missing checkpoint, bad k, unknown indexer, degenerate
    # request count, learned indexer on a checkpoint without indexer
    # weights.
    _expect_raises(FileNotFoundError,
                   lambda: sweep_engine_sparse(
                       str(_REPO_ROOT / "nope.pt"), ks=[4], max_new=4,
                       n_reqs=2),
                   "missing checkpoint")
    _expect_raises(ValueError,
                   lambda: sweep_engine_sparse(ckpt, ks=[0], max_new=4,
                                               n_reqs=2),
                   "sparse k = 0")
    _expect_raises(ValueError,
                   lambda: sweep_engine_sparse(ckpt, ks=[4], max_new=4,
                                               n_reqs=2, indexer="magic"),
                   "unknown indexer")
    _expect_raises(ValueError,
                   lambda: sweep_engine_sparse(ckpt, ks=[4], max_new=4,
                                               n_reqs=1),
                   "n_reqs < 2")
    _expect_raises(ValueError,
                   lambda: sweep_engine_sparse(ckpt, ks=[4], max_new=4,
                                               n_reqs=2, indexer="learned"),
                   "learned indexer w/o checkpoint weights")
    _pass("test_engine_sparse_serving",
          f"schema ok, solo-prefix 1.0 on all rows, k>=L bitwise, k=4 "
          f"overlap {sparse['output_overlap_vs_dense_engine']} tok/s "
          f"{sparse['tok_s_out']} (gain {sparse['gain_vs_dense_engine']}), "
          f"deterministic, no draft API, 5 loud guards")
