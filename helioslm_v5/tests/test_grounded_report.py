"""v5.214: JSON -> report grounding grader (eval/grounded_report.py).

Pins:
1. 24 synthetic zh-TW / English fixtures (correct, fabricated number,
   missing nodule, wrong category, wrong unit, wrong type, swapped values,
   full-width digits, lingula, cm) get the expected unverified /
   contradiction / omission counts;
2. bound vs unbound sentences: a value from another nodule in a bound
   sentence is a contradiction;
3. a value stated for a null JSON field is unverified; 未測量 / not
   measured is returned as an abstention;
4. determinism: same input, byte-identical JSON output;
5. schema validation rejects unknown lobes / types / categories;
6. the runner sends one OpenAI-compatible request per (model, case) and
   grades the replies (stub opener, no network).
"""
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.eval.grounded_report import (  # noqa: E402
    SCHEMA_EXAMPLE, extract_facts, grade, summarize, validate_findings)

FIX = Path(__file__).parent / "fixtures" / "grounded_report"
FINDINGS = json.loads((FIX / "findings.json").read_text())
CASES = [json.loads(line) for line in (FIX / "cases.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]


def test_fixture_coverage():
    assert len(CASES) >= 20
    assert {c["lang"] for c in CASES} == {"zh", "en"}
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture(case):
    r = grade(FINDINGS[case["findings"]], case["text"])
    got = {k: r["counts"][k] for k in case["expect"]}
    assert got == case["expect"], (r["unverified_facts"], r["contradictions"], r["omissions"])
    assert r["grounded"] == (got["unverified_facts"] == 0 and got["contradictions"] == 0)


def test_example_schema_matches_data_file():
    assert json.loads((ROOT / "data" / "pulmoguard" / "findings_example.json").read_text()) \
        == SCHEMA_EXAMPLE == FINDINGS["A"]


def test_bound_sentence_uses_its_nodule():
    # 4.1 mm belongs to N2; stated in the RUL sentence it contradicts N1
    r = grade(SCHEMA_EXAMPLE, "右上葉 4.1 公釐實質結節。左下葉 4.1 mm 毛玻璃結節。Lung-RADS 4A。")
    assert [(c["nodule"], c["field"], c["expected"]) for c in r["contradictions"]] \
        == [("N1", "diameter_mm", 9.4)]
    # unbound, the same value only has to exist somewhere
    r = grade(SCHEMA_EXAMPLE, "結節大小 4.1 公釐。")
    assert r["counts"]["unverified_facts"] == 0 and r["counts"]["contradictions"] == 0


def test_null_field_and_abstention():
    r = grade(SCHEMA_EXAMPLE, "左下葉 4.1 公釐毛玻璃結節，體積未測量。")
    assert r["abstentions"] == [{"nodule": "N2", "fields": ["volume_mm3"],
                                 "text": "未測量", "span": r["abstentions"][0]["span"]}]
    r = grade(SCHEMA_EXAMPLE, "左下葉 4.1 公釐毛玻璃結節，體積 40 立方公釐。")
    assert [(u["nodule"], u["field"]) for u in r["unverified_facts"]] == [("N2", "volume_mm3")]


def test_extract_units_and_categories():
    f = extract_facts("６．２㎜，0.5 公分，12 mm³，Lung-RADS 4b，Lung-RADS v1.1 2")
    vals = [(x["field"], x["value"]) for x in f]
    assert ("diameter_mm", 6.2) in vals and ("diameter_mm", 5.0) in vals
    assert ("volume_mm3", 12.0) in vals
    assert ("lung_rads", "4B") in vals and ("lung_rads", "2") in vals
    assert not any(x["field"] == "type" for x in extract_facts("肺實質正常"))
    assert [x["value"] for x in extract_facts("part-solid and solid") if x["field"] == "type"] \
        == ["part-solid", "solid"]


def test_deterministic():
    for c in CASES:
        a = json.dumps(grade(FINDINGS[c["findings"]], c["text"]), sort_keys=True, ensure_ascii=False)
        b = json.dumps(grade(FINDINGS[c["findings"]], c["text"]), sort_keys=True, ensure_ascii=False)
        assert a == b


@pytest.mark.parametrize("bad", [
    {"nodules": [{"lobe": "XYZ"}]},
    {"nodules": [{"lobe": "RUL", "type": "calcified-ish"}]},
    {"lung_rads": "5", "nodules": []},
    {"nodules": [{"id": "N1"}, {"id": "N1"}]},
    {"lung_rads": "1"},
])
def test_schema_rejects(bad):
    with pytest.raises(ValueError):
        validate_findings(bad)


def test_summarize():
    rs = [grade(FINDINGS[c["findings"]], c["text"]) for c in CASES]
    s = summarize(rs)
    assert s["reports"] == len(CASES)
    assert s["unverified_facts"] == sum(c["expect"]["unverified_facts"] for c in CASES)
    assert 0 < s["zero_unverified_rate"] < 1


def test_runner_stub():
    import grounded_report_run as R
    sent = []
    good = "右上葉 9.4 公釐實質結節，體積 310 立方公釐，Lung-RADS 4A。左下葉 4.1 公釐毛玻璃結節，Lung-RADS 2。"

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        body = json.loads(req.data)
        sent.append((req.full_url, body["model"], body["temperature"]))
        text = good if body["model"] == "good" else "右上葉 12 公釐實質結節。"
        return Resp(json.dumps({"choices": [{"message": {"content": text}}]}).encode())

    clients = {}
    for spec in ("a=http://x/v1/good", "b=http://y/v1/bad"):
        name, base, mid = R.parse_model(spec)
        clients[name] = R.make_client(base, mid, opener=opener)
    res = R.run([SCHEMA_EXAMPLE], clients, lang="zh")
    assert sent == [("http://x/v1/chat/completions", "good", 0),
                    ("http://y/v1/chat/completions", "bad", 0)]
    assert res["summary"]["a"]["passed_rate"] == 1.0
    assert res["summary"]["b"]["contradictions"] == 1
    with pytest.raises(SystemExit):
        R.parse_model("nourl")
