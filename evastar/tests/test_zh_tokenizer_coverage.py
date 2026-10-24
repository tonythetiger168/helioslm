"""T114 — roadmap TW4: zh-TW tokenizer coverage report, v5.75.

Pins:

1. the evaluation text is real Traditional Chinese from the held-out
   documents, and the held-out documents never reach a training corpus
   (assert_no_leak raises on an injected line; the real corpora pass);
2. the metric oracles: UTF-8 is exactly 3 tokens per CJK character with
   no single-token coverage; the char-toy vocabulary drops every CJK
   character; a CJK-stripped corpus really has no CJK left;
3. the committed report: the code-only BPE falls back to bytes (~3 per
   CJK char, 0 single-token coverage); the repo BPE that saw zh-TW text
   is cheaper and covers more, and the larger vocabulary is at least as
   good; recomputing the cheap rows reproduces them; hash.

Run from repo root: python3 evastar/tests/test_zh_tokenizer_coverage.py
"""
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval.zh_tokenizer_coverage import (  # noqa: E402
    CharToy, Utf8Bytes, _CJK, assert_no_leak, eval_split, held_out_texts,
    measure, training_corpus)

_ART = _REPO / "benchmarks" / "zh_tokenizer_coverage_2026-10-20.json"
_SIMPLIFIED = "这个们说为时实会与体档训练测试"
_TRADITIONAL = "這個們說為時實會與體檔訓練測試"


def test_eval_text_is_traditional_and_held_out():
    zh, en = eval_split(_REPO)
    text = "\n".join(zh)
    assert len(_CJK.findall(text)) > 1000
    assert sum(text.count(c) for c in _TRADITIONAL) > 10
    assert sum(text.count(c) for c in _SIMPLIFIED) == 0
    held = list(held_out_texts(_REPO).values())
    for strip in (False, True):
        assert_no_leak(training_corpus(_REPO, strip_cjk=strip), held)
    line = next(l for l in held[0].splitlines()
                if len(l) > 20 and _CJK.search(l))
    try:
        assert_no_leak("prefix\n" + line + "\nsuffix", held)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an injected held-out line was not caught")
    print("PASS test_eval_text_is_traditional_and_held_out")


def test_metric_oracles():
    zh, en = eval_split(_REPO)
    b = measure(Utf8Bytes(), zh, en)
    assert b["tokens_per_cjk_char"] == 3.0
    assert b["cjk_single_token_coverage"] == 0.0 and b["cjk_lost_fraction"] == 0.0
    c = measure(CharToy(1024), zh, en)
    assert c["cjk_lost_fraction"] == 1.0 and c["tokens_per_cjk_char"] == 0.0
    assert not _CJK.search(training_corpus(_REPO, strip_cjk=True))
    print("PASS test_metric_oracles")


def test_report_artifact():
    art = json.loads(_ART.read_text())
    rows = art["results"]
    by = {}
    for r in rows:
        by.setdefault(r["tokenizer"], []).append(r)
    zh, en = eval_split(_REPO)
    assert by["utf8_bytes"][0] == {"tokenizer": "utf8_bytes", "vocab": 256,
                                   **measure(Utf8Bytes(), zh, en)}
    assert by["char_toy"][0] == {"tokenizer": "char_toy", "vocab": 1024,
                                 **measure(CharToy(1024), zh, en)}
    for r in by["bpe_code_only"]:
        assert r["tokens_per_cjk_char"] > 2.9 and r["cjk_single_token_coverage"] == 0.0
    repo = sorted(by["bpe_repo"], key=lambda r: r["vocab"])
    code = sorted(by["bpe_code_only"], key=lambda r: r["vocab"])
    for rr, cc in zip(repo, code):
        assert rr["tokens_per_cjk_char"] < cc["tokens_per_cjk_char"] / 2
        assert rr["cjk_single_token_coverage"] > 0.3
    assert repo[-1]["tokens_per_cjk_char"] <= repo[0]["tokens_per_cjk_char"]
    assert repo[-1]["cjk_single_token_coverage"] >= repo[0]["cjk_single_token_coverage"]
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    print("PASS test_report_artifact")


if __name__ == "__main__":
    test_eval_text_is_traditional_and_held_out()
    test_metric_oracles()
    test_report_artifact()
