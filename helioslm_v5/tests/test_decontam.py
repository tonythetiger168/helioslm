"""Plan E2: training / evaluation n-gram overlap check.

Pins:

1. words(): lowercase alphanumeric runs, one word per CJK / kana / hangul
   character, punctuation ignored;
2. planted leaks are found (English and Chinese, reformatted, embedded in
   longer text) and disjoint text is not; an n-gram straddling the item is
   not a hit;
3. items shorter than n are checked whole; items under min_words are
   listed as unchecked, not silently passed;
4. digests are stable across processes (no Python hash randomisation);
5. report / decontaminate bookkeeping;
6. pilot_prepare_data.py --decontaminate drops exactly the leaking
   training documents and records it in meta.json (needs ``tokenizers``).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval.decontam import (  # noqa: E402
    EvalIndex, decontaminate, ngram_digests, words)

Q_EN = ("Which planet in the solar system has the largest number of "
        "confirmed moons as of the most recent survey")
Q_ZH = "台灣最高的山是哪一座，它的海拔大約是多少公尺，位於哪一個國家公園之內"


def test_words():
    assert words("Hello, World! x2") == ["hello", "world", "x2"]
    assert words("玉山 is 3952m") == ["玉", "山", "is", "3952m"]
    assert words("ひらがなとハングル한국") == list("ひらがなとハングル한국")
    assert words("snake_case") == ["snake", "case"]


def test_leaks_found_and_clean_text_passes():
    ix = EvalIndex({"en": Q_EN, "zh": Q_ZH})
    leak_en = "Trivia night. " + Q_EN.upper().replace(" ", "  ,") + " Answer: Saturn."
    leak_zh = "網友問：" + Q_ZH.replace("，", " ") + "？大家來回答"
    assert ix.scan(leak_en) == {"en"}
    assert ix.scan(leak_zh) == {"zh"}
    assert ix.scan(leak_en + leak_zh) == {"en", "zh"}
    assert ix.scan("The solar system has eight planets and many moons.") == set()
    # 12 of the 13 words of a window are not enough
    w = words(Q_EN)
    assert ix.scan(" ".join(w[:12])) == set()
    assert ix.scan(" ".join(w[:13])) == {"en"}


def test_short_items():
    ix = EvalIndex({"mid": "the capital city of taiwan is taipei today",   # 8 words
                    "tiny": "2 + 2 = ?"}, n=13)
    assert ix.unchecked == ["tiny"]
    assert ix.scan("we all know the capital city of Taiwan is Taipei today.") == {"mid"}
    assert ix.scan("the capital city of taiwan is taichung today") == set()
    rep = ix.report(["2 + 2 = ?"])
    assert rep["unchecked_items"] == ["tiny"] and rep["contaminated_items"] == 0
    with pytest.raises(ValueError):
        EvalIndex({}, n=0)


def test_digests_stable_across_processes():
    code = ("import sys; sys.path.insert(0, %r); "
            "from helioslm_v5.eval.decontam import ngram_digests; "
            "print(sorted(d.hex() for d in ngram_digests(%r)))" % (str(ROOT), Q_EN))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(s)}).stdout
            for s in (1, 2)}
    assert len(outs) == 1
    assert json.loads(outs.pop().replace("'", '"')) == sorted(
        d.hex() for d in ngram_digests(Q_EN))


def test_report_and_decontaminate():
    ix = EvalIndex({"en": Q_EN, "zh": Q_ZH})
    docs = ["clean text about cooking rice", "quiz: " + Q_EN, "more clean text",
            Q_ZH + " " + Q_EN]
    kept, rep = decontaminate(docs, ix)
    assert kept == ["clean text about cooking rice", "more clean text"]
    assert rep["overlapping_doc_indices"] == [1, 3]
    assert rep["items"] == {"en": 2, "zh": 1}
    assert rep["contaminated_items"] == 2 and rep["contaminated_fraction"] == 1.0
    assert rep["training_docs"] == 4


def test_prepare_data_decontaminates():
    pytest.importorskip("tokenizers")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "src"
        src.mkdir()
        for i in range(30):
            (src / f"{i:02d}.txt").write_text(
                f"Document {i} about rivers, rain and mountains in Taiwan. " * 10)
        # not sorted last: with --val-fraction 0 the last document is
        # moved to the validation split, which is not decontaminated
        (src / "05_leak.txt").write_text("Practice set. " + Q_EN + " Saturn.")
        ev = d / "eval.jsonl"
        ev.write_text(json.dumps({"question": Q_EN}) + "\n"
                      + json.dumps({"question": "unrelated item about chess "
                                                "openings and endgame theory "
                                                "for club players"}) + "\n")
        out = d / "out"
        r = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "pilot_prepare_data.py"),
             "--text", str(src), "--vocab", "600", "--zh-seed", "none",
             "--val-fraction", "0.0", "--decontaminate", str(ev),
             "--eval-field", "question", "--decontam-report", str(d / "rep.json"),
             "--out", str(out)],
            capture_output=True, text=True, timeout=300,
            env={"RAYON_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr[-2000:]
        meta = json.loads((out / "meta.json").read_text())
        dc = meta["decontamination"]
        assert dc["eval_items"] == 2 and dc["contaminated_items"] == 1
        assert dc["dropped_train_docs"] == 1
        # 31 documents: one held out for validation, the leak dropped
        assert meta["splits"]["train"]["docs"] + meta["splits"]["val"]["docs"] == 30
        rep = json.loads((d / "rep.json").read_text())
        assert rep["items"] == {"e0": 1}
