"""T134 — route B pilot: mixture fetch and continued-pretraining loop
(v5.137).

Pins:

1. fetch_mixture.parse_source / licence_matches: hf:repo[:config];
   licence families match across CC versions and aliases, not across
   families;
2. fetch_mixture.plan: review licences, non-hf sources, unreadable and
   mismatched cards are skipped with a reason; the usable sources share
   each phase's tokens by renormalised weight; sources without download
   access (v5.144) are skipped and listed by blocked();
3. write_source: heuristics + zh_variants filter, BOS + ids + EOS rows
   as uint32 (ids above 65,535 survive), stops at the budget;
4. cpt_hf.MixedSource: deterministic per (seed, step), draws sources by
   weight, refuses bad weights;
5. cpt_hf.load_phases + train on a tiny torch LM: the loss falls, the
   WSD schedule switches to the decay sources at the decay step,
   max_hours stops early, freeze() leaves matching parameters untouched;
   fit_warmup (v5.138) keeps the decay switch in a run shorter than the
   default warmup; dry_run_report lists the frozen parameter groups;
6. base_gap_eval --base / --refs measure gaps from a checkpoint.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_cpt_pilot.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from examples import fetch_mixture as fm  # noqa: E402

TW = ("這是一篇關於臺灣歷史與地理的文章，說明國家公園的生態保育、"
      "鐵路發展與傳統飲食文化，並且討論學校教育如何讓學生認識鄉土。") * 3
CN = ("这是一篇关于历史与地理的文章，说明国家公园的生态保护、"
      "铁路发展与传统饮食文化，并且讨论学校教育如何让学生认识乡土。") * 3


def test_parse_and_licence():
    assert fm.parse_source("hf:a/b") == ("a/b", None)
    assert fm.parse_source("hf:a/b:cmn_Hani") == ("a/b", "cmn_Hani")
    assert fm.parse_source("path:x.jsonl") is None
    assert fm.licence_matches("ODC-BY-1.0", "odc-by")
    assert fm.licence_matches("CC-BY-SA-4.0", ["cc-by-sa-3.0", "gfdl"])
    assert fm.licence_matches("MIT", "mit")
    assert not fm.licence_matches("CC-BY-SA-4.0", "cc-by-nc-sa-4.0")
    assert not fm.licence_matches("MIT", None)


def _mix():
    return {"name": "t", "sources": [
        {"name": "a", "source": "hf:x/a", "license": "MIT", "phase": "main",
         "weight": 0.5},
        {"name": "b", "source": "hf:x/b", "license": "MIT", "phase": "main",
         "weight": 0.3},
        {"name": "c", "source": "hf:x/c", "license": "review",
         "license_note": "n", "phase": "main", "weight": 0.2},
        {"name": "d", "source": "hf:x/d", "license": "MIT", "phase": "decay",
         "weight": 0.6},
        {"name": "e", "source": "path:e.jsonl", "license": "MIT",
         "phase": "decay", "weight": 0.4},
    ]}


def test_plan():
    rows = fm.plan(_mix(), 1000, 0.2, {"a": "mit", "b": "apache-2.0",
                                       "d": "mit"})
    assert rows["a"]["usable"] and rows["d"]["usable"]
    assert "card says" in rows["b"]["reason"]
    assert rows["c"]["reason"] == "licence under review"
    assert rows["e"]["reason"] == "not a hf: or git: source"
    assert rows["a"]["budget"] == 800 and rows["d"]["budget"] == 200
    assert rows["a"]["weight"] == 1.0
    with pytest.raises(SystemExit):
        fm.plan(_mix(), 1000, 0.2, {})
    assert fm.blocked(rows) == []


def test_plan_no_access():
    cards = {"a": "mit", "b": "mit", "d": "mit"}
    rows = fm.plan(_mix(), 1000, 0.2, cards, {"a": "GatedRepoError"})
    assert not rows["a"]["usable"] and rows["a"]["budget"] == 0
    assert rows["a"]["reason"] == "no download access (GatedRepoError)"
    assert rows["b"]["budget"] == 800          # a's share moves to b
    assert fm.blocked(rows) == ["a"]
    with pytest.raises(SystemExit):           # no main source left
        fm.plan(_mix(), 1000, 0.2, cards, {"a": "x", "b": "x"})


def _enc(batch):
    return [[70000 + (ord(c) % 7) for c in t[:20]] for t in batch]


def test_write_source():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.bin"
        texts = [TW, CN, "short", TW, TW, TW]
        r = fm.write_source(iter(texts), _enc, p, budget=50, bos=2, eos=1,
                            variants={"tw"}, batch=2)
        arr = np.fromfile(p, dtype=np.uint32)
        assert r["tokens"] == len(arr) == 66          # 3 docs of 22 > 50
        assert arr[0] == 2 and arr[21] == 1 and arr[1] >= 70000
        assert r["docs"] == 3 and r["filter"]["dropped"]
        r = fm.write_source(iter([TW]), _enc, p, budget=10**6, bos=None,
                            eos=1)
        assert r["tokens"] == 21 and r["docs"] == 1


try:
    import torch
    from examples import cpt_hf
    from helioslm_v5.src.training.pretrain import TokenShard
except ImportError:                                    # CPU-only checkout
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch")


def _shard(d, name, vals):
    p = Path(d) / f"{name}.bin"
    np.asarray(vals, dtype=np.uint32).tofile(p)
    return TokenShard(p, dtype="uint32")


@needs_torch
def test_mixed_source():
    with tempfile.TemporaryDirectory() as d:
        a = _shard(d, "a", [1] * 400)
        b = _shard(d, "b", [2] * 400)
        m = cpt_hf.MixedSource([a, b], [3, 1])
        x1, _ = m.batch(64, 8, 5, 0)
        x2, _ = m.batch(64, 8, 5, 0)
        assert torch.equal(x1, x2)
        share = (x1[:, 0] == 1).float().mean().item()
        assert 0.55 < share < 0.95
        for bad in (([a], [1, 2]), ([a], [0]), ([], [])):
            with pytest.raises(ValueError):
                cpt_hf.MixedSource(*bad)


def _LM(vocab=32):
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.per_layer_table = torch.nn.Embedding(vocab, 16)
            self.e = torch.nn.Embedding(vocab, 16)
            self.o = torch.nn.Linear(16, vocab)

        def forward(self, input_ids):
            h = self.e(input_ids) + self.per_layer_table(input_ids)
            return type("Out", (), {"logits": self.o(h)})()
    return LM()


def _data(d):
    rng = np.random.default_rng(0)
    pattern = np.tile(np.arange(4, 12), 200)           # learnable
    (Path(d) / "main").mkdir()
    (Path(d) / "decay").mkdir()
    pattern.astype(np.uint32).tofile(Path(d) / "main" / "m.bin")
    rng.integers(4, 30, 600).astype(np.uint32).tofile(
        Path(d) / "decay" / "z.bin")
    man = {"dtype": "uint32", "sources": {
        "m": {"phase": "main", "path": "main/m.bin", "tokens": 1600,
              "weight": 1.0},
        "z": {"phase": "decay", "path": "decay/z.bin", "tokens": 600,
              "weight": 1.0},
        "skipped": {"phase": "main", "usable": False, "tokens": 0}}}
    (Path(d) / "manifest.json").write_text(json.dumps(man))


@needs_torch
def test_train_loop():
    with tempfile.TemporaryDirectory() as d:
        _data(d)
        phases = cpt_hf.load_phases(d, seq=16)
        model = _LM()
        tr, fr = cpt_hf.freeze(model, ["per_layer"])
        assert fr == 32 * 16 and tr > 0
        frozen = model.per_layer_table.weight.detach().clone()
        rows = []
        res = cpt_hf.train(model, phases, steps=40, micro_batch=4, seq=16,
                           grad_accum=2, lr=3e-2, warmup=2, decay_frac=0.25,
                           log=rows.append, autocast=False)
        assert res["steps"] == 40 and res["stopped"] is None
        assert res["tokens"] == 40 * 4 * 16 * 2
        main_losses = [r["loss"] for r in rows if r["phase"] == "main"]
        assert main_losses[-1] < main_losses[0] * 0.6
        assert [r["phase"] for r in rows].index("decay") == 30
        assert torch.equal(model.per_layer_table.weight, frozen)
        res = cpt_hf.train(_LM(), phases, steps=10, micro_batch=1, seq=16,
                           grad_accum=1, lr=1e-3, warmup=1, max_hours=0.0,
                           autocast=False)
        assert res["stopped"] == "max_hours" and res["steps"] == 0


@needs_torch
def test_short_run_still_decays():
    assert cpt_hf.fit_warmup(2860, 100, 0.2) == 100       # pod run unchanged
    assert cpt_hf.fit_warmup(20, 100, 0.2) == 16
    assert cpt_hf.fit_warmup(1, 100, 0.2) == 0
    with tempfile.TemporaryDirectory() as d:
        _data(d)
        phases = cpt_hf.load_phases(d, seq=16)
        rows = []
        cpt_hf.train(_LM(), phases, steps=20, micro_batch=1, seq=16,
                     grad_accum=1, lr=1e-3, warmup=cpt_hf.fit_warmup(20, 100, 0.2),
                     log=rows.append, autocast=False)
        assert [r["phase"] for r in rows].index("decay") == 16


@needs_torch
def test_dry_run_report():
    model = _LM()
    tr, fr = cpt_hf.freeze(model, ["per_layer"])
    with tempfile.TemporaryDirectory() as d:
        rep = cpt_hf.dry_run_report(model, d, tr, fr)
        assert json.loads((Path(d) / "dry_run.json").read_text()) == rep
    assert rep["frozen_groups"] == {"per_layer_table.weight": 32 * 16}
    assert rep["trainable_params"] == tr and rep["model_class"] == "LM"


def test_gap_from_checkpoint():
    from examples import base_gap_eval as bg
    ids = [f"t/{i}" for i in range(20)]
    scores = {"runs/b0/final": {"tmmluplus": {i: True for i in ids}},
              "google/gemma-4-E2B": {"tmmluplus": {i: False for i in ids}}}
    g = bg.gap_table(scores, "runs/b0/final", ["google/gemma-4-E2B"])
    assert g["tmmluplus"]["google/gemma-4-E2B"]["diff"] == 1.0
