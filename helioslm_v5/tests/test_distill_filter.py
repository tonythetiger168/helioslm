"""D2 (v5.112): filtering teacher answers (src/training/distill_filter.py).

Pins:
1. simplified_rate: 0 on Traditional text (including 台/着, which are
   excluded from the list), high on simplified text, 0 without Han;
2. every single-record check drops for its own reason: truncated,
   empty / leaked think tag, reasoning and answer token caps, simplified
   answer or reasoning, eval overlap in prompt or answer, unparsable code;
3. math self-consistency: the majority answer's samples are kept, the
   minority dropped; no majority drops the prompt; final_answer reads
   boxed answers and the last number, normalised;
4. near-duplicate prompts are dropped but samples of one prompt are not;
5. the report adds up and the CLI writes kept records and the report.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.src.training import distill_filter as df  # noqa: E402

GOOD = "臺灣的健保制度讓民眾看病更方便，以下說明它的運作方式與費用。"


def rec(pid="p", content=GOOD, reasoning="先整理重點。", fin="stop", domain="chat",
        prompt="請說明台灣的健保制度如何運作，以及民眾要付多少費用？", out=100, reas=40, k=0):
    return {"id": f"{pid}#{k}", "prompt_id": pid, "sample": k, "domain": domain,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": reasoning, "content": content, "finish_reason": fin,
            "usage": {"input_tokens": 10, "output_tokens": out, "reasoning_tokens": reas}}


def test_simplified_rate():
    assert df.simplified_rate(GOOD) == 0.0
    assert df.simplified_rate("台灣的夜市看着很熱鬧") == 0.0
    assert df.simplified_rate("这个问题的数据") > 0.5
    assert df.simplified_rate("hello 123") == 0.0
    assert df.simplified_rate(None) == 0.0


def test_single_record_reasons():
    gsm = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins"
    flt = df.Filter(eval_items={"gsm:0": gsm}, max_reasoning_tokens=50, max_answer_tokens=80)
    cases = {
        "finish": rec(pid="a", fin="length"),
        "empty": rec(pid="b", content="  "),
        "length": rec(pid="c", reas=60),
        "script": rec(pid="d", content="这个问题的数据需要说明"),
        "decontam": rec(pid="e", prompt=gsm + " How much does she make?"),
        "verify": rec(pid="f", domain="code", content="```python\ndef f(:\n```"),
    }
    assert flt._single(rec(pid="ok")) is None
    assert flt._single(rec(pid="g", content="答案</think>")) == "empty"
    assert flt._single(rec(pid="h", out=200, reas=40)) == "length"       # answer 160 > 80
    assert flt._single(rec(pid="i", reasoning="这个问题的数据")) == "script"
    assert flt._single(rec(pid="j", content="原題：" + gsm)) == "decontam"
    assert flt._single(rec(pid="k", domain="code",
                           content="```python\ndef f(x):\n    return x\n```")) is None
    for why, r in cases.items():
        assert flt._single(r) == why, why
    kept, rep = flt.run(list(cases.values()) + [rec(pid="ok")])
    assert [r["prompt_id"] for r in kept] == ["ok"]
    assert rep["dropped"] == {"finish": 1, "empty": 1, "length": 1, "script": 1,
                              "decontam": 1, "verify": 1, "near_dup": 0}
    assert rep["input"] == 7 and rep["kept"] == 1 and rep["eval_items"] == 1
    with pytest.raises(ValueError):
        df.Filter(max_tokens=3)


def test_final_answer():
    assert df.final_answer("所以答案是 \\boxed{1,200}。") == "1200"
    assert df.final_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert df.final_answer("共 3 個，再加 4.0 個") == "4"
    assert df.final_answer("沒有數字") is None


def test_math_self_consistency():
    m = [rec(pid="m", domain="math", k=i, content=f"答案是 \\boxed{{{a}}}")
         for i, a in enumerate([42, 42, 41])]
    tie = [rec(pid="t", domain="math", k=i, content=f"答案 {a}",
               prompt="一個完全不同的機率問題：擲兩顆骰子的點數和是多少的機率最高？")
           for i, a in enumerate([1, 2])]
    single = [rec(pid="s", domain="math", content="答案 7",
                  prompt="另一個題目：三角形內角和是幾度？請寫出推導過程與答案。")]
    kept, rep = df.Filter().run(m + tie + single)
    assert sorted(r["id"] for r in kept) == ["m#0", "m#1", "s#0"]
    assert rep["dropped"]["verify"] == 3 and rep["kept_prompts"] == 2


def test_near_duplicate_prompts():
    base = "請說明台灣的健保制度如何運作，以及民眾看門診時大概要付多少部分負擔費用？"
    rs = [rec(pid="a", prompt=base, k=0), rec(pid="a", prompt=base, k=1),
          rec(pid="b", prompt=base + "謝謝"),
          rec(pid="c", prompt="用 Python 寫一個函式，輸入一個整數串列，回傳其中所有偶數的平方和。")]
    kept, rep = df.Filter().run(rs)
    assert [r["id"] for r in kept] == ["a#0", "a#1", "c#0"]
    assert rep["dropped"]["near_dup"] == 1


def test_cli(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import distill_filter as cli
    ans = tmp_path / "a.jsonl"
    ev = tmp_path / "ev.jsonl"
    ans.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                             [rec(pid="x"), rec(pid="y", fin="length")]), encoding="utf-8")
    ev.write_text(json.dumps({"question": "an unrelated english evaluation question "
                              "about ducks and eggs and markets"}) + "\n", encoding="utf-8")
    out = tmp_path / "kept.jsonl"
    cli.main(["--answers", str(ans), "--out", str(out), "--eval", str(ev)])
    assert [json.loads(line)["prompt_id"] for line in out.read_text().splitlines()] == ["x"]
    rep = json.loads((tmp_path / "kept.jsonl.report.json").read_text())
    assert rep["dropped"]["finish"] == 1 and rep["eval_items"] == 1
