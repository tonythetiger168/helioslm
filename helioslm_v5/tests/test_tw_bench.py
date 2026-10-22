"""Plan E3 / TW3: Traditional-Chinese MCQ pack (helioslm_v5/eval/tw_bench.py).

All fixtures are invented questions written for this test; no benchmark
text is in the repository (TW1 licence rule: evaluate locally, never
redistribute).

Pins:
1. the TMMLU+, C-Eval, CMMLU and column-mapped (TW-JusticeBench) layouts
   load to the same record shape; split filtering, subjects, letter and
   0-3 / full-width answers;
2. malformed records raise TWBenchError, never pass silently;
3. letter extraction oracles (zh-TW / zh-CN phrasing, full-width, no
   letter -> None);
4. the Phase A gates hold on a loaded set and catch a broken extractor;
5. loglik scoring picks the max, its confidence is the letter softmax;
   summarize's subject and category macros;
6. k-shot prompts; results JSON carries no question text;
7. eval_texts feeds the E2 overlap check;
8. model_score_fn equals log-softmax computed directly from the logits;
9. examples/tw_bench_run.py --check-only runs the gates end to end.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import tw_bench  # noqa: E402
from helioslm_v5.eval.decontam import EvalIndex  # noqa: E402
from helioslm_v5.eval.tw_bench import TWBenchError  # noqa: E402

QS = [
    ("臺灣最高的山是哪一座？", "玉山", "雪山", "合歡山", "阿里山", "A"),
    ("一年有幾個月？", "十個", "十一個", "十二個", "十三個", "C"),
    ("水的化學式是什麼？", "CO2", "H2O", "O2", "NaCl", "B"),
    ("下列何者是質數？", "4", "6", "8", "7", "D"),
]


def _csv(path, header, rows):
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def tmmlu(tmp_path):
    d = tmp_path / "tmmluplus"
    d.mkdir()
    hdr = ["question", "A", "B", "C", "D", "answer"]
    _csv(d / "geography_test.csv", hdr, [QS[0], QS[1]])
    _csv(d / "chemistry_test.csv", hdr, [QS[2], QS[3]])
    _csv(d / "geography_dev.csv", hdr, [QS[1]])
    _csv(d / "chemistry_dev.csv", hdr, [QS[3]])
    return d


def test_load_tmmluplus_layout(tmmlu):
    tasks = tw_bench.load_mcq(tmmlu, "tmmluplus")
    assert [t["task_id"] for t in tasks] == [
        "tmmluplus/chemistry/0", "tmmluplus/chemistry/1",
        "tmmluplus/geography/0", "tmmluplus/geography/1"]
    assert [t["answer"] for t in tasks] == ["B", "D", "A", "C"]
    assert tasks[2]["choices"] == ["玉山", "雪山", "合歡山", "阿里山"]
    dev = tw_bench.load_mcq(tmmlu, "tmmluplus", split="dev")
    assert len(dev) == 2 and {t["subject"] for t in dev} == {"geography", "chemistry"}


def test_load_ceval_cmmlu_and_mapped(tmp_path):
    ce = tmp_path / "law_val.csv"
    _csv(ce, ["id", "question", "A", "B", "C", "D", "answer", "explanation"],
         [(0, *QS[0], "")])
    t, = tw_bench.load_mcq(ce, "ceval", split="val")
    assert t["subject"] == "law" and t["answer"] == "A"
    cm = tmp_path / "cmmlu"
    cm.mkdir()
    _csv(cm / "agronomy.csv", ["", "Question", "A", "B", "C", "D", "Answer"],
         [(0, *QS[1])])
    t, = tw_bench.load_mcq(cm, "cmmlu")
    assert t["task_id"] == "cmmlu/agronomy/0" and t["answer"] == "C"
    js = tmp_path / "justice_test.jsonl"
    js.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"prompt": QS[2][0], "A": "CO2", "B": "H2O", "C": "O2", "D": "NaCl", "label": 1},
        {"prompt": QS[3][0], "A": "4", "B": "6", "C": "8", "D": "7", "label": "Ｄ"},
    ]) + "\n", encoding="utf-8")
    tasks = tw_bench.load_mcq(js, "tw_justice", columns={"question": "prompt"})
    assert [t["answer"] for t in tasks] == ["B", "D"]


def test_malformed_records_raise(tmp_path):
    bad = tmp_path / "x_test.csv"
    _csv(bad, ["question", "A", "B", "C", "answer"], [("q", "1", "2", "3", "A")])
    with pytest.raises(TWBenchError, match="'D'"):
        tw_bench.load_mcq(bad, "tmmluplus")
    _csv(bad, ["question", "A", "B", "C", "D", "answer"], [("q", "1", "2", "3", "4", "E")])
    with pytest.raises(TWBenchError, match="answer"):
        tw_bench.load_mcq(bad, "tmmluplus")
    with pytest.raises(TWBenchError, match="unknown bench"):
        tw_bench.load_mcq(bad, "mmlu")
    with pytest.raises(TWBenchError, match="not found"):
        tw_bench.load_mcq(tmp_path / "nope", "tmmluplus")
    with pytest.raises(TWBenchError, match="no dev files"):
        tw_bench.load_mcq(tmp_path, "tmmluplus", split="dev")


@pytest.mark.parametrize("text,letter", [
    ("答案：B", "B"), ("答案是 (C)", "C"), ("答案為Ｄ。", "D"), ("答案应为 A", "A"),
    ("我選 B，因為……", "B"), ("选择（C）", "C"), ("A. 玉山", "A"), ("(D)", "D"),
    ("", None), ("我不知道", None), ("ABC 都不對", None),
])
def test_extract_letter(text, letter):
    assert tw_bench.extract_letter(text) == letter


def test_phase_a_gates(tmmlu, monkeypatch):
    tasks = tw_bench.load_mcq(tmmlu, "tmmluplus")
    assert tw_bench.self_check(tasks) == {
        "reference_generate": 1.0, "reference_loglik": 1.0, "empty_generate": 0.0}
    monkeypatch.setattr(tw_bench, "extract_letter", lambda text: "A")
    with pytest.raises(TWBenchError, match="Phase A"):
        tw_bench.self_check(tasks)


def test_loglik_and_summary(tmmlu):
    tasks = tw_bench.load_mcq(tmmlu, "tmmluplus")
    lps = [-1.0, -0.5, -2.0, -3.0]                    # always picks B
    res = tw_bench.run_loglik(tasks, lambda p, c: lps)
    assert [r["final"] for r in res] == ["B"] * 4
    z = sum(math.exp(x) for x in lps)
    assert res[0]["conf"] == pytest.approx(math.exp(-0.5) / z)
    s = tw_bench.summarize(res, {"chemistry": "STEM", "geography": "social"})
    assert s["accuracy"] == 0.25 and s["subjects"] == {"chemistry": 0.5, "geography": 0.0}
    assert s["macro_subject"] == 0.25 and s["macro_category"] == 0.25
    with pytest.raises(TWBenchError, match="category"):
        tw_bench.summarize(res, {"chemistry": "STEM"})
    with pytest.raises(TWBenchError, match="score_fn"):
        tw_bench.run_loglik(tasks, lambda p, c: [0.0, float("nan"), 0.0, 0.0])


def test_shots_and_results_file(tmmlu, tmp_path):
    tasks = tw_bench.load_mcq(tmmlu, "tmmluplus")
    shots = tw_bench.shots_by_subject(
        tw_bench.load_mcq(tmmlu, "tmmluplus", split="dev"), 1)
    p = tw_bench.build_prompt(tasks[0], shots["chemistry"])
    assert p.startswith(QS[3][0]) and p.count("答案：") == 2
    assert p.split("\n\n")[0].endswith("答案： D") and p.endswith("答案：")
    assert tw_bench.build_prompt(tasks[0], template="lm_eval").endswith("Answer:")
    res = tw_bench.run_generate(tasks, lambda prompt, i: "答案：A")
    out = tmp_path / "r.json"
    tw_bench.write_results(out, "tmmluplus", res, tw_bench.summarize(res))
    text = out.read_text(encoding="utf-8")
    assert json.loads(text)["schema"] == tw_bench.SCHEMA_ID
    for q in QS:
        assert q[0] not in text and q[2] not in text


def test_eval_texts_feed_decontam(tmmlu):
    tasks = tw_bench.load_mcq(tmmlu, "tmmluplus")
    index = EvalIndex(tw_bench.eval_texts(tasks), n=8, min_words=4)
    doc = "網頁內容：臺灣最高的山是哪一座？玉山雪山合歡山阿里山。"
    assert index.scan(doc) == {"tmmluplus/geography/0"}


def test_model_score_fn():
    pytest.importorskip("tokenizers")
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.model_v5 import HeliosLMv5
    text = "".join(q[0] + "".join(q[1:5]) + "答案：" + q[5] + "\n" for q in QS) * 20
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text], trainers.BpeTrainer(
        vocab_size=400, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]"]))
    torch.manual_seed(0)
    model = HeliosLMv5(HeliosLMv5Config(size="lite")).eval()
    score = tw_bench.model_score_fn(model, tok, max_length=64)
    ctx = "一年有幾個月？\nA. 十個\n答案："
    got = score(ctx, [" A", " B"])
    for cont, lp in zip([" A", " B"], got):
        full = tok.encode(ctx + cont).ids
        n = len(full) - len(tok.encode(ctx).ids)
        assert n >= 1
        with torch.no_grad():
            logp = F.log_softmax(model(torch.tensor([full[:-1]]))[0][0].float(), -1)
        want = sum(logp[len(full) - 1 - n + j, full[len(full) - n + j]].item()
                   for j in range(n))
        assert lp == pytest.approx(want, abs=1e-5)
    long_ctx = "很長的前文。" * 100 + ctx                 # cut from the left
    assert all(math.isfinite(x) for x in score(long_ctx, [" A"]))


def test_example_check_only(tmmlu):
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "tw_bench_run.py"),
                        "--bench", "tmmluplus", "--data", str(tmmlu), "--check-only"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["n"] == 4 and out["phase_a"]["reference_loglik"] == 1.0
