"""T132 — route B base-gap eval: Hugging Face scorer and gap table (v5.129).

Pins:

1. _single_token_continuations takes the one-pass path only when the
   context's ids are an exact prefix of every continuation and each adds
   one token;
2. hf_score_fn: the one-pass scores equal the per-continuation scores
   (a tiny torch model and a tokenizer that merges " A" into one token),
   and a continuation that does not merge falls back to one pass each;
3. gap_table: base minus reference per bench over the task ids both scored,
   via paired_bootstrap; missing models / benches are skipped;
4. load_tasks reads a <subject>_<split> directory, applies --limit, runs
   the Phase A gates and takes the shots from the dev split.

Run from repo root: python3 -m pytest evastar/tests/test_base_gap.py
"""
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from examples import base_gap_eval as bg  # noqa: E402
from evastar.eval import tw_bench  # noqa: E402


def test_single_token_path():
    f = tw_bench._single_token_continuations
    assert f([1, 5, 6], [[1, 5, 6, 9], [1, 5, 6, 8]]) == ([1, 5, 6], [9, 8])
    assert f([1, 5, 6], [[1, 5, 7, 9], [1, 5, 6, 8]]) is None   # merge
    assert f([1, 5, 6], [[1, 5, 6, 9, 2], [1, 5, 6, 8]]) is None
    assert f([], [[3]]) is None


class _Tok:
    """BOS, then " A".." D" as one token each, other chars one token."""
    MERGE = {" A": 2, " B": 3, " C": 4, " D": 5}

    def __call__(self, text, add_special_tokens=True):
        ids, i = [1] if add_special_tokens else [], 0
        while i < len(text):
            if text[i:i + 2] in self.MERGE:
                ids.append(self.MERGE[text[i:i + 2]])
                i += 2
            else:
                ids.append(6 + ord(text[i]) % 40)
                i += 1
        return {"input_ids": ids}


def _model():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.e = torch.nn.Embedding(64, 16)
            self.o = torch.nn.Linear(16, 64)

        def forward(self, input_ids):
            h = self.e(input_ids).cumsum(1)
            return type("Out", (), {"logits": self.o(h)})()
    return M().eval()


def test_hf_score_fast_equals_slow(monkeypatch):
    model, tok = _model(), _Tok()
    ctx = "題目\nA. 甲\nB. 乙\n答案："
    conts = [" A", " B", " C", " D"]
    fast = tw_bench.hf_score_fn(model, tok)(ctx, conts)
    monkeypatch.setattr(tw_bench, "_single_token_continuations",
                        lambda c, f: None)
    slow = tw_bench.hf_score_fn(model, tok)(ctx, conts)
    assert fast == pytest.approx(slow, abs=1e-5)
    two = tw_bench.hf_score_fn(model, tok)(ctx, [" AB", " CD"])
    assert len(two) == 2 and all(x < 0 for x in two)


def test_gap_table():
    base, r1, r2 = bg.BASE, bg.REFS[0], bg.REFS[1]
    ids = [f"t/{i}" for i in range(40)]
    scores = {
        base: {"tmmluplus": {i: True for i in ids}},
        r1: {"tmmluplus": {i: (n % 2 == 0) for n, i in enumerate(ids)},
             "ceval": {"c/0": True}},
        r2: {"tmmluplus": {i: False for i in ids[:10]}},
    }
    g = bg.gap_table(scores)
    assert set(g) == {"tmmluplus"}                 # base has no ceval
    assert g["tmmluplus"][r1]["diff"] == pytest.approx(0.5)
    assert g["tmmluplus"][r1]["n"] == 40
    assert g["tmmluplus"][r2]["n"] == 10           # shared ids only
    assert g["tmmluplus"][r2]["ci95_low"] == pytest.approx(1.0)


def test_load_tasks():
    header = "question,A,B,C,D,answer\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tmmluplus"
        p.mkdir()
        for split, n in (("test", 6), ("dev", 5), ("val", 2)):
            rows = "".join(f"q{split}{i},a,b,c,d,{'ABCD'[i % 4]}\n"
                           for i in range(n))
            (p / f"law_{split}.csv").write_text(header + rows, "utf-8")
        tasks, shots = bg.load_tasks(d, "tmmluplus", limit=4)
        assert len(tasks) == 4 and tasks[0]["answer"] == "A"
        assert len(shots["law"]) == 5
        tasks, shots = bg.load_tasks(d, "tmmluplus")
        assert len(tasks) == 6


def test_assign():
    groups = bg.assign([bg.BASE, *bg.REFS], 4)
    assert len(groups) == 4
    assert [bg.REFS[0]] in groups                # Breexe alone
    assert sorted(m for g in groups for m in g) == sorted([bg.BASE, *bg.REFS])
    assert bg.assign([bg.BASE], 4) == [[bg.BASE]]
    assert bg.assign([bg.BASE, *bg.REFS], 1) == [
        sorted([bg.BASE, *bg.REFS], key=lambda m: -bg.SIZE_B[m])]


def test_main_without_models_writes_report():
    with tempfile.TemporaryDirectory() as d:
        r = bg.main(["--models", "--benches", "ceval", "--data", d,
                     "--out-dir", d])
        assert r["models"] == {} and "ceval" in r["task_errors"]
        assert list(Path(d).glob("base_gap_*.json"))
