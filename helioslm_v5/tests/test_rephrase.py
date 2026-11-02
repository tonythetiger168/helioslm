"""v5.206: knowledge rephrasing (helioslm_v5/src/training/rephrase.py,
examples/rephrase_corpus.py).

Pins:
1. split_chunks: chunks never exceed max_chars, paragraph order and all
   text kept, an over-long sentence is hard-cut;
2. rephrase_prompts: one prompt per (style, chunk), stable ids, the
   preceding source chunk's tail as context, unknown style refused, the
   prompt passes distill_gen.normalize_prompt;
3. fidelity_check: a faithful rewrite passes; Simplified output, lost
   numbers, off-topic text, a verbatim copy, a bad length ratio and a
   truncated answer each fail with their reason;
4. assemble: chunks joined in order with the source licence and the
   teacher in the attribution; a (doc, style) with a failing or missing
   chunk is dropped and counted;
5. CLI round trip prompts -> fake teacher answers -> assemble.
"""
import json
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # v5.225: this file is also executed directly by the daily TESTS
    # suite (case_upstream_suites), where pytest is not installed. The
    # shim covers exactly the one pytest feature used below: raises
    # (pytest semantics: re.search on str(exception) when match is
    # given; none of the cases below pass match).
    import contextlib as _contextlib
    import re as _re

    class _PytestShim:
        @staticmethod
        @_contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match is not None and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"raised {e!r}, which does not match /{match}/"
                    ) from None
                return
            raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _PytestShim()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from helioslm_v5.src.training import distill_gen as dg  # noqa: E402
from helioslm_v5.src.training import rephrase as rp  # noqa: E402
import rephrase_corpus  # noqa: E402

SRC = ("臺北101位於臺灣臺北市信義區，於2004年完工，高度為508公尺，"
       "曾經是世界上最高的建築物。大樓共有101層樓，內部設有調諧質量阻尼器，"
       "用來減少颱風與地震造成的搖晃。")
GOOD = ("位於臺北市信義區的臺北101在2004年完工，樓高508公尺，共有101層，"
        "過去曾是全世界最高的建築物。為了降低颱風和地震帶來的晃動，"
        "大樓裡裝設了調諧質量阻尼器。")


def test_split_chunks_bounds_and_order():
    paras = ["第一段" + "甲" * 300, "第二段" + "乙" * 300, "第三段" + "丙" * 900]
    text = "\n\n".join(paras)
    chunks = rp.split_chunks(text, max_chars=700)
    assert all(len(c) <= 700 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert chunks[0].startswith("第一段") and "\n\n第二段" in chunks[0]
    long = rp.split_chunks("字" * 250, max_chars=100)
    assert [len(c) for c in long] == [100, 100, 50]
    with pytest.raises(ValueError):
        rp.split_chunks("abc", max_chars=10)


def test_prompts_ids_context_and_normalize():
    doc = {"id": "zhwiki-1", "text": "甲" * 120 + "。\n\n" + "乙" * 120 + "。"}
    ps = rp.rephrase_prompts(doc, ("qa", "notes"), max_chars=130, context_chars=20)
    assert [p["id"] for p in ps] == ["zhwiki-1~qa~0000", "zhwiki-1~qa~0001",
                                     "zhwiki-1~notes~0000", "zhwiki-1~notes~0001"]
    assert "前文" not in ps[0]["messages"][1]["content"]
    assert "】\n" + "甲" * 19 + "。\n" in ps[1]["messages"][1]["content"]
    assert ps[1]["meta"]["source_text"].startswith("乙")
    assert ps[1]["meta"]["n_chunks"] == 2
    n = dg.normalize_prompt(ps[0])
    assert n["id"] == ps[0]["id"] and n["domain"] == "rephrase"
    with pytest.raises(ValueError):
        rp.rephrase_prompts(doc, ("poem",))


def test_fidelity_pass_and_failures():
    ok, reasons, st = rp.fidelity_check(SRC, GOOD, "textbook")
    assert ok, (reasons, st)
    assert st["number_recall"] == 1.0

    simp = GOOD.replace("臺北", "台北").replace("颱風和地震帶來的晃動", "台风和地震带来的晃动") \
        .replace("過去曾是全世界最高的建築物", "过去曾是全世界最高的建筑物")
    assert "simplified" in rp.fidelity_check(SRC, simp, "textbook")[1]
    lost = GOOD.replace("2004", "某").replace("508", "很")
    assert "numbers_lost" in rp.fidelity_check(SRC, lost, "textbook")[1]
    other = ("日月潭是臺灣最大的淡水湖泊，湖面在南投縣魚池鄉，2004年508公尺101層。"
             "環湖公路適合騎腳踏車，沿途可以欣賞山光水色與邵族文化。")
    assert "off_topic" in rp.fidelity_check(SRC, other, "textbook")[1]
    assert "near_copy" in rp.fidelity_check(SRC, SRC, "textbook")[1]
    assert "length_ratio" in rp.fidelity_check(SRC, GOOD[:30], "textbook")[1]
    assert "finish_length" in rp.fidelity_check(SRC, GOOD, "textbook",
                                                finish_reason="length")[1]


def test_clean_output_drops_preamble():
    assert rp.clean_output("以下是改寫後的內容：\n" + GOOD) == GOOD
    assert rp.clean_output(GOOD) == GOOD


def _answers(prompts, fail=None):
    out = []
    for p in prompts:
        content = p["meta"]["source_text"].replace("臺北101位於", "位於").replace(
            "曾經是", "過去曾是").replace("用來減少", "可以降低")
        content = GOOD if p["meta"]["source_text"] == SRC else content
        if fail and p["id"] == fail:
            content = "完全無關的內容。" * 5
        out.append({"id": p["id"] + "#0", "meta": p["meta"], "content": content,
                    "finish_reason": "stop",
                    "teacher": {"repo": "deepseek-ai/DeepSeek-V4.1-Flash"}})
    return out


def test_assemble_joins_and_drops():
    docs = {"d1": {"id": "d1", "text": SRC, "source": "zhwiki",
                   "license": "CC-BY-SA-4.0", "attribution": "Wikipedia contributors"},
            "d2": {"id": "d2", "text": SRC, "source": "zhwiki",
                   "license": "CC-BY-SA-4.0", "attribution": ""}}
    prompts = [p for d in docs.values() for p in rp.rephrase_prompts(d, ("textbook", "qa"))]
    recs = _answers(prompts, fail="d2~textbook~0000")
    recs = [r for r in recs if r["id"] != "d2~qa~0000#0"]   # d2/qa missing
    for r in recs:
        if r["meta"]["style"] == "qa":
            r["content"] = "問：" + GOOD + "\n答：是的，" + GOOD
    out, rep = rp.assemble(recs, docs)
    ids = [d["id"] for d in out]
    assert ids == ["d1~qa", "d1~textbook"]
    d = out[1]
    assert d["text"] == GOOD and d["license"] == "CC-BY-SA-4.0"
    assert d["source"] == "zhwiki+rephrase"
    assert "Wikipedia contributors" in d["attribution"]
    assert "DeepSeek-V4.1-Flash" in d["attribution"]
    assert rep["chunks_failed"] == 1 and rep["docs_dropped"] == 0
    assert rep["reasons"].get("off_topic") == 1


def test_assemble_multichunk_order():
    text = SRC + "\n\n" + SRC.replace("臺北101", "這棟大樓")
    doc = {"id": "d", "text": text, "license": "X", "source": "s"}
    prompts = rp.rephrase_prompts(doc, ("encyclopedia",), max_chars=len(SRC) + 5)
    assert len(prompts) == 2
    recs = [{"meta": p["meta"], "content": p["meta"]["source_text"] .replace(
        "曾經是", "過去曾是").replace("位於", "坐落在").replace("用來減少", "可以減輕")
        .replace("大樓共有", "整棟共有").replace("內部設有", "裡面裝有"),
        "finish_reason": "stop", "teacher": None} for p in reversed(prompts)]
    out, rep = rp.assemble(recs, {"d": doc}, teacher="t")
    assert rep["chunks_failed"] == 0, rep
    assert out[0]["text"].index("坐落在臺灣") < out[0]["text"].index("這棟大樓")


def test_cli_round_trip(tmp_path):
    corpus = tmp_path / "tw2" / "zhwiki"
    corpus.mkdir(parents=True)
    rows = [{"id": "w1", "text": SRC, "source": "zhwiki", "license": "CC-BY-SA-4.0",
             "attribution": "Wikipedia contributors"},
            {"id": "w2", "text": "太短", "source": "zhwiki", "license": "CC-BY-SA-4.0"}]
    (corpus / "zhwiki-00000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    prompts = tmp_path / "p.jsonl"
    rephrase_corpus.main(["prompts", "--docs", str(tmp_path / "tw2"), "--styles",
                          "textbook", "--min-chars", "50", "--out", str(prompts)])
    ps = dg.read_jsonl(str(prompts))
    assert [p["id"] for p in ps] == ["w1~textbook~0000"]
    answers = tmp_path / "a.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in _answers(ps)), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    rephrase_corpus.main(["assemble", "--docs", str(tmp_path / "tw2"), "--answers",
                          str(answers), "--out", str(out)])
    got = dg.read_jsonl(str(out))
    assert len(got) == 1 and got[0]["id"] == "w1~textbook"


def test_plan_tokens():
    assert rp.plan_tokens(1e8, rewrites=5) == 600_000_000
