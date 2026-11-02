"""v5.216: field-level abstention for JSON -> report writing
(src/training/report_abstain.py).

Pins (zh-TW and English):
1. removed field + 未測量 / not measured -> positive reward;
2. removed field + a number -> -penalty (fabrication);
3. field present and stated correctly -> positive;
4. 未測量 and a number for the same field -> an answer (fabrication);
5. a removed field left unmentioned scores below a full abstention;
   abstention is attributed by the word before it (體積 / volume);
6. ReportAbstainEnv is deterministic per seed, removes fields at the
   requested rate and routes through MultiEnvBatch;
7. make_sft_records references all score 1.0; the data CLI writes them.
The existing abstain.py tests are untouched.
"""
import json
import random
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # v5.225: this file is also executed directly by the daily TESTS
    # suite (case_upstream_suites), where pytest is not installed. The
    # shim covers exactly the two pytest features used below:
    # mark.parametrize (collection-time only; a no-op when the functions
    # are called directly with explicit arguments) and raises
    # (pytest semantics: re.search on str(exception) when match is
    # given; none of the cases below pass match).
    import contextlib as _contextlib
    import re as _re

    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda f: f

    class _PytestShim:
        mark = _Mark()

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.src.training.report_abstain import (  # noqa: E402
    ReportAbstainEnv, make_sft_records, render_reference, report_abstain_reward,
    score_report, synth_findings)

F = {"exam_id": "SYN-1", "lung_rads": "4A", "nodules": [
    {"id": "N1", "lobe": "RUL", "diameter_mm": 9.4, "volume_mm3": None,
     "type": "solid", "lung_rads": "4A"}]}
T = {"findings": F, "missing": [["N1", "volume_mm3"]], "lang": "zh"}
FULL = {"findings": dict(F, nodules=[dict(F["nodules"][0], volume_mm3=435.0)]),
        "missing": [], "lang": "zh"}

ZH_OK = "右上葉見一顆實質結節，直徑 9.4 公釐，體積未測量，Lung-RADS 4A。整體 Lung-RADS 類別 4A。"
EN_OK = "Right upper lobe: solid nodule, diameter 9.4 mm, volume not measured, Lung-RADS 4A. Overall Lung-RADS category 4A."


@pytest.mark.parametrize("text", [ZH_OK, EN_OK])
def test_abstain_on_missing_is_positive(text):
    s = score_report(T, text)
    assert s["reward"] == 1.0 and s["abstained"] == [["N1", "volume_mm3"]]


@pytest.mark.parametrize("text", [
    ZH_OK.replace("體積未測量", "體積 435 立方公釐"),
    EN_OK.replace("volume not measured", "volume 435 mm3"),
])
def test_number_on_missing_is_negative(text):
    s = score_report(T, text, penalty=2.0)
    assert s["reward"] == -2.0 and s["fabricated"] == [["N1", "volume_mm3"]]


@pytest.mark.parametrize("text", [
    ZH_OK.replace("體積未測量", "體積 435 立方公釐"),
    EN_OK.replace("volume not measured", "volume 435 mm3"),
])
def test_present_and_correct_is_positive(text):
    assert report_abstain_reward(FULL, text) == 1.0


@pytest.mark.parametrize("text", [
    ZH_OK.replace("體積未測量", "體積未測量（約 435 立方公釐）"),
    EN_OK.replace("volume not measured", "volume not measured, about 435 mm3"),
])
def test_hedge_is_an_answer(text):
    assert report_abstain_reward(T, text) == -1.0


def test_silent_and_attribution():
    t = {"findings": dict(F, nodules=[dict(F["nodules"][0], diameter_mm=None)]),
         "missing": [["N1", "diameter_mm"], ["N1", "volume_mm3"]], "lang": "zh"}
    both = "右上葉見一顆實質結節，直徑未測量，體積未測量，Lung-RADS 4A。整體 Lung-RADS 類別 4A。"
    only_vol = "右上葉見一顆實質結節，體積未測量，Lung-RADS 4A。整體 Lung-RADS 類別 4A。"
    a, b = score_report(t, both), score_report(t, only_vol)
    assert a["reward"] == 1.0
    assert b["silent"] == [["N1", "diameter_mm"]] and 0 <= b["reward"] < 1.0
    silent = score_report(T, "右上葉見一顆實質結節，直徑 9.4 公釐，Lung-RADS 4A。整體 Lung-RADS 類別 4A。")
    assert silent["silent"] == [["N1", "volume_mm3"]] and 0 < silent["reward"] < 1.0


def test_contradiction_on_present_field_is_negative():
    assert report_abstain_reward(T, ZH_OK.replace("9.4", "12")) == -1.0


def test_env_deterministic_rate_and_batch():
    from helioslm_v5.src.training.envs import MultiEnvBatch
    env = ReportAbstainEnv(p_missing=0.5)
    assert env.task(seed=7) == env.task(seed=7)
    tasks = [env.task(seed=s) for s in range(300)]
    fields = sum(2 * len(t["target"]["findings"]["nodules"]) for t in tasks)
    missing = sum(len(t["target"]["missing"]) for t in tasks)
    assert 0.4 < missing / fields < 0.6
    assert {t["target"]["lang"] for t in tasks} == {"zh", "en"}
    t = tasks[0]["target"]
    assert env.reward(render_reference(t["findings"], t["lang"]), t) == 1.0
    b = MultiEnvBatch({"report-abstain": 3, "math": 1},
                      envs={"report-abstain": ReportAbstainEnv})
    bt = b.tasks(seed=0)
    rep = [x for x in bt if x["env"] == "report-abstain"]
    assert len(rep) == 3
    rs = b.rewards(rep, [render_reference(x["target"]["findings"], x["target"]["lang"])
                         for x in rep])
    assert rs == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        ReportAbstainEnv(p_missing=1.5)


def test_synth_findings_valid():
    from helioslm_v5.eval.grounded_report import validate_findings
    rng = random.Random(0)
    for _ in range(50):
        f = synth_findings(rng)
        validate_findings(f)
        assert len({n["lobe"] for n in f["nodules"]}) == len(f["nodules"])


def test_sft_records_and_cli(tmp_path):
    recs = make_sft_records(200, seed=3)
    assert {r["lang"] for r in recs} == {"zh", "en"}
    assert any(r["missing"] for r in recs)
    assert all("未測量" in r["response"] or "not measured" in r["response"]
               for r in recs if r["missing"])
    assert all(r["system"] not in r["prompt"] for r in recs)
    import abstain_report_sft
    out = tmp_path / "r.jsonl"
    st = abstain_report_sft.main(["--n", "20", "--lang", "en", "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert st["records"] == len(rows) == 20 and {r["lang"] for r in rows} == {"en"}
