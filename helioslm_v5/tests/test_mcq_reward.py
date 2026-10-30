"""v5.207: verifiable multiple-choice reward for GRPO
(helioslm_v5/eval/mcq_reward.py, examples/hdl_rl_hf.py --reward mcq).

Pins:
1. make_task: zh-TW prompt with the four choices, key normalised,
   bad key / choice count refused;
2. final_letter: the last 「答案：X」 after </think> wins over letters in
   the reasoning; full-width letters; last-line fallback; None when absent;
3. reward: 1 / 0 / no answer; Simplified zh-TW answer penalised, zh-CN
   task not;
4. tasks_from_distill: teacher majority letter as key, disagreeing and
   truncated samples dropped, prompts without A-D lines dropped;
5. drop_contaminated removes a task copied from the eval set;
6. hdl_rl_hf.py prepare-mcq writes tasks from a CSV split, decontaminated
   against another split.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from helioslm_v5.eval import mcq_reward as mr  # noqa: E402

Q = "臺灣最高的山是哪一座？"
CH = ["阿里山", "玉山", "雪山", "合歡山"]


def test_make_task():
    t = mr.make_task(Q, CH, " b ", task_id="q1")
    assert t["answer"] == "B" and t["kind"] == "mcq" and t["lang"] == "tw"
    assert "B. 玉山" in t["prompt"] and t["prompt"].startswith(mr.INSTRUCTION)
    with pytest.raises(ValueError):
        mr.make_task(Q, CH, "E")
    with pytest.raises(ValueError):
        mr.make_task(Q, CH[:3], "A")


def test_final_letter():
    assert mr.final_letter("<think>答案是A嗎？不對。</think>\n玉山最高。\n答案：B") == "B"
    assert mr.final_letter("先想想，答案：A 不對，應該是\n答案：Ｂ") == "B"
    assert mr.final_letter("推理……\nB") == "B"
    assert mr.final_letter("<think>答案：B</think>我不知道") is None
    assert mr.final_letter("") is None


def test_reward():
    t = mr.make_task(Q, CH, "B")
    assert mr.reward("玉山海拔3952公尺，是臺灣最高峰。\n答案：B", t)[0] == 1.0
    r, info = mr.reward("答案：A", t)
    assert r == 0.0 and info["pred"] == "A" and not info["ok"]
    assert mr.reward("不確定", t)[1]["why"] == "no answer"
    simp = "玉山海拔三千九百五十二米，是台湾最高的山峰，这个问题很简单。\n答案：B"
    r, info = mr.reward(simp, t)
    assert r == pytest.approx(0.8) and info["simplified"]
    cn = mr.make_task(Q, CH, "B", lang="cn")
    assert mr.reward(simp, cn)[0] == 1.0


def _rec(pid, k, letter, finish="stop", prompt=None):
    prompt = prompt or mr.make_task(Q, CH, "B")["prompt"]
    return {"id": f"{pid}#{k}", "prompt_id": pid, "finish_reason": finish,
            "messages": [{"role": "user", "content": prompt}],
            "content": f"想一想……\n答案：{letter}"}


def test_tasks_from_distill():
    recs = [_rec("p1", 0, "B"), _rec("p1", 1, "B"), _rec("p1", 2, "C", finish="length"),
            _rec("p2", 0, "A"), _rec("p2", 1, "C"),
            _rec("p3", 0, "A", prompt="沒有選項的問題"),
            {**_rec("p4", 0, "A"), "content": "不知道"}]
    tasks, st = mr.tasks_from_distill(recs)
    assert [t["task_id"] for t in tasks] == ["p1"]
    assert tasks[0]["answer"] == "B" and tasks[0]["choices"] == CH
    assert tasks[0]["question"].endswith(Q)
    assert st == {"prompts": 4, "no_letter": 1, "disagree": 1, "unparsed": 1, "tasks": 1}
    loose, _ = mr.tasks_from_distill(recs, min_agree=0.5)
    assert sorted(t["task_id"] for t in loose) == ["p1", "p2"]


def test_drop_contaminated():
    long_q = "下列關於臺灣地理位置與氣候特徵的敘述，哪一項是正確的選擇？"
    tasks = [mr.make_task(long_q, CH, "A", task_id="dup"),
             mr.make_task("請問一年有幾個月份呢？這是一個很簡單的常識問題。", ["10", "11", "12", "13"], "C",
                          task_id="ok")]
    kept, n = mr.drop_contaminated(tasks, {"e1": long_q + "\n" + "\n".join(CH)})
    assert n == 1 and [t["task_id"] for t in kept] == ["ok"]


def test_prepare_mcq_cli(tmp_path):
    import hdl_rl_hf as rl
    d = tmp_path / "tmmlu"
    d.mkdir()
    hdr = "question,A,B,C,D,answer\n"
    (d / "geo_train.csv").write_text(
        hdr + f"{Q},{','.join(CH)},B\n請問一年有幾個月份呢？這是一個很簡單的常識問題。,10,11,12,13,C\n",
        encoding="utf-8")
    (d / "geo_test.csv").write_text(hdr + f"{Q},{','.join(CH)},B\n", encoding="utf-8")
    out = tmp_path / "mcq.jsonl"
    st = rl.main(["prepare-mcq", "--mcq", str(d), "--bench", "tmmluplus", "--split", "train",
                  "--eval", str(d), "--eval-split", "test", "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert st["tasks"] == len(rows) == 1 and st["contaminated"] == 1
    assert rows[0]["answer"] == "C" and rows[0]["kind"] == "mcq"
