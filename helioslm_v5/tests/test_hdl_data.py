"""T156 — HDL track H1: the HDL continued-pretraining mixture and the
per-source loading options of fetch_mixture (v5.178).

Pins:

1. data/mixtures/hdl_h1_cpt_v1.json loads; HDL sources skip the prose
   heuristics and name their text column; the-stack sources stay under
   review, so plan() gives their share to the usable sources;
2. load_mixture refuses a bad field / data_dir / data_files / heuristics;
3. write_source(heuristics=False) keeps a short Verilog module the prose
   heuristics drop;
4. fetch_mixture.main passes data_dir and the source's field to the
   dataset loader (fake datasets / huggingface_hub / tokenizers modules).

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_data.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, validate_mixture)
from examples import fetch_mixture as fm  # noqa: E402

MIX = ROOT / "data" / "mixtures" / "hdl_h1_cpt_v1.json"
HDL = {"verilog-github", "stack-systemverilog", "stack-vhdl", "vhdl-github",
       "verithoughts-code"}
VERILOG = """module and2(input a, input b, output y);
  assign y = a & b;
endmodule
"""


def test_hdl_mixture():
    mix = load_mixture(MIX)
    src = {s["name"]: s for s in mix["sources"]}
    assert HDL <= set(src)
    for n in HDL:
        assert src[n]["heuristics"] is False and src[n]["field"]
    assert src["stack-vhdl"]["data_dir"] == "data/vhdl"
    assert abs(sum(src[n]["weight"] for n in HDL) - 0.75) < 1e-9
    cards = {n: s["license"].lower() for n, s in src.items()
             if s["license"] != "review"}
    rows = fm.plan(mix, 10**6, 0.2, cards)
    assert rows["stack-vhdl"]["reason"] == "licence under review"
    assert rows["stack-systemverilog"]["budget"] == 0
    usable = [n for n, r in rows.items() if r["usable"]]
    assert sum(rows[n]["budget"] for n in usable) == pytest.approx(10**6, abs=5)
    assert rows["verilog-github"]["weight"] == pytest.approx(0.35 / 0.73)


@pytest.mark.parametrize("extra", [
    {"field": ""}, {"field": 3}, {"data_dir": None}, {"data_files": []},
    {"data_files": ["a", ""]}, {"heuristics": "no"}])
def test_bad_options(extra):
    mix = {"name": "t", "sources": [dict(
        {"name": "a", "source": "hf:x/a", "license": "MIT", "weight": 1.0},
        **extra)]}
    with pytest.raises(ValueError):
        validate_mixture(mix)


def _enc(batch):
    return [[ord(c) for c in t] for t in batch]


def test_code_skips_heuristics():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.bin"
        r = fm.write_source(iter([VERILOG]), _enc, p, 10**6, None, 0)
        assert r["docs"] == 0 and "heuristic:too_short" in r["filter"]["dropped"]
        r = fm.write_source(iter([VERILOG]), _enc, p, 10**6, None, 0,
                            heuristics=False)
        assert r["docs"] == 1
        assert np.fromfile(p, dtype=np.uint32).size == len(VERILOG) + 1


def test_main_uses_source_options(monkeypatch):
    calls = []

    def load_dataset(repo, cfg, **kw):
        calls.append((repo, cfg, kw))
        return iter([{"content": VERILOG, "text": "wrong column"}])

    class Api:
        def dataset_info(self, repo):
            return types.SimpleNamespace(card_data={"license": "mit"},
                                         sha="r1")

        def auth_check(self, repo, repo_type):
            pass

    class Tok:
        @staticmethod
        def from_file(path):
            return Tok()

        def token_to_id(self, t):
            return {"<eos>": 1}.get(t)

        def encode_batch(self, b, add_special_tokens):
            return [types.SimpleNamespace(ids=[ord(c) for c in t]) for t in b]

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=Api))
    monkeypatch.setitem(sys.modules, "tokenizers",
                        types.SimpleNamespace(Tokenizer=Tok))
    with tempfile.TemporaryDirectory() as d:
        mix = Path(d) / "m.json"
        mix.write_text(json.dumps({"name": "t", "sources": [
            {"name": "sv", "source": "hf:x/stack", "license": "MIT",
             "weight": 1.0, "data_dir": "data/systemverilog",
             "field": "content", "heuristics": False}]}))
        tok = Path(d) / "tok.json"
        tok.write_text("{}")
        m = fm.main(["--mixture", str(mix), "--tokenizer", str(tok),
                     "--tokens", "1000", "--out", d])
        assert calls == [("x/stack", None, {
            "split": "train", "streaming": True, "revision": "r1",
            "data_dir": "data/systemverilog"})]
        assert m["sources"]["sv"]["docs"] == 1
        assert m["sources"]["sv"]["tokens"] == len(VERILOG) + 1
