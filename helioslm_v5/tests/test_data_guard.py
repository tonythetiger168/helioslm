"""v5.215: PHI and licence guard for medical training data
(src/training/data_guard.py, cpt_hf / sft_hf / distill_generate flags).

Pins:
1. the PHI scanner flags fake Taiwanese IDs, record numbers, phones,
   e-mail, date + name and DICOM tags, reports line numbers, masks the
   value; clean synthetic report text passes;
2. manifests: a medical source without deidentified / licence / origin is
   refused; research-only data is refused for releasable weights and
   allowed with release=False; general sources are never scanned;
3. guard() stops on a PHI hit in a medical source file;
4. distillation refuses an external teacher for medical data;
5. the CLIs: --medical without a manifest is refused, no flags is a no-op.
   All data here is fake.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.src.training.data_guard import (  # noqa: E402
    DataGuardError, check_manifest, check_teacher_endpoint, guard, guard_cli,
    guard_records, is_local_url, load_manifest, research_only, scan_file,
    scan_text)

CLEAN = [
    "右上葉見一顆 9.4 公釐實質結節，體積 310 立方公釐，Lung-RADS 4A。",
    "A 4.1 mm ground-glass nodule in the left lower lobe, Lung-RADS 2.",
    "建議 3 個月後追蹤低劑量電腦斷層。",
    "Follow-up LDCT in 12 months.",
]
DIRTY = [
    ("tw_national_id", "病人身分證 A123456789，右上葉結節。"),
    ("medical_record_no", "病歷號：12345678，左下葉毛玻璃結節。"),
    ("medical_record_no", "MRN: AB-99821 nodule follow-up"),
    ("phone", "聯絡電話 0912-345-678"),
    ("phone", "call (02) 2345-6789 for results"),
    ("email", "send to someone@example.com"),
    ("date_with_name", "姓名：王小明 檢查日期 2026/10/31"),
    ("date_with_name", "陳先生於 2026-10-01 接受 LDCT"),
    ("dicom_tag", "PatientName=DOE^JOHN"),
    ("dicom_tag", "(0010,0020) 0001234"),
]


@pytest.mark.parametrize("text", CLEAN)
def test_clean_passes(text):
    assert scan_text(text) == []


@pytest.mark.parametrize("kind,text", DIRTY)
def test_dirty_flagged(kind, text):
    kinds = [h["kind"] for h in scan_text(text)]
    assert kind in kinds


def test_masked_and_line_numbers(tmp_path):
    f = tmp_path / "pairs.jsonl"
    rows = [{"prompt": CLEAN[0], "response": CLEAN[1]},
            {"prompt": "身分證 A123456789", "response": "ok"},
            {"prompt": CLEAN[2], "response": "PatientID: 77"}]
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    hits = scan_file(f)
    assert [(h["line"], h["kind"]) for h in hits] == [(2, "tw_national_id"), (3, "dicom_tag")]
    assert "A123456789" not in json.dumps(hits)


def _manifest(tmp_path, **med):
    f = tmp_path / "pairs.jsonl"
    f.write_text("\n".join(json.dumps({"p": t}, ensure_ascii=False) for t in CLEAN) + "\n")
    g = tmp_path / "web.jsonl"
    g.write_text(json.dumps({"text": "客服電話 0912-345-678"}, ensure_ascii=False) + "\n")
    src = {"name": "pairs", "path": "pairs.jsonl", "domain": "medical",
           "deidentified": True, "license": "internal", "origin": "synthetic"}
    src.update(med)
    m = {"sources": [src, {"name": "web", "path": "web.jsonl", "domain": "general",
                           "license": "ODC-By", "origin": "crawl"}]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p


def test_clean_manifest_passes_general_not_scanned(tmp_path):
    rep = guard(load_manifest(_manifest(tmp_path)))
    assert rep["medical_sources"] == ["pairs"] and rep["scanned"] == ["pairs"]
    assert rep["hits"] == []   # the phone number in the general source is ignored


@pytest.mark.parametrize("bad,msg", [
    ({"deidentified": False}, "deidentified"),
    ({"deidentified": None}, "deidentified"),
    ({"license": ""}, "license"),
    ({"origin": None}, "origin"),
])
def test_medical_source_without_fields_refused(tmp_path, bad, msg):
    with pytest.raises(DataGuardError, match=msg):
        guard(load_manifest(_manifest(tmp_path, **bad)))


def test_research_only(tmp_path):
    for extra in ({"research_only": True}, {"license": "CC-BY-NC-4.0"},
                  {"name": "medical-bench-tw"}, {"license": "research use only"}):
        m = load_manifest(_manifest(tmp_path, **extra))
        assert research_only(m["sources"][0])
        with pytest.raises(DataGuardError, match="research-only"):
            check_manifest(m)
        assert check_manifest(m, release=False) == [m["sources"][0]["name"]]
    assert not research_only({"license": "Apache-2.0", "name": "x"})


def test_phi_in_medical_file_stops(tmp_path):
    p = _manifest(tmp_path)
    with open(tmp_path / "pairs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"p": "病歷號 99887766 姓名：林某某 2026/01/02"}, ensure_ascii=False) + "\n")
    with pytest.raises(DataGuardError, match=r"pairs:5 medical_record_no"):
        guard(load_manifest(p))


def test_missing_medical_file(tmp_path):
    p = _manifest(tmp_path)
    (tmp_path / "pairs.jsonl").unlink()
    with pytest.raises(DataGuardError, match="not found"):
        guard(load_manifest(p))


def test_teacher_endpoint(tmp_path):
    m = load_manifest(_manifest(tmp_path))
    check_teacher_endpoint(m, "http://localhost:8000/v1", "vllm")
    check_teacher_endpoint(m, "http://10.0.0.5:8000/v1", "vllm")
    for url, ep in (("https://api.deepseek.com/v1", "openai"),
                    ("http://localhost:8000/v1", "openai"),
                    ("https://teacher.example.com/v1", "vllm")):
        with pytest.raises(DataGuardError, match="external teacher"):
            check_teacher_endpoint(m, url, ep)
    general = {"sources": [m["sources"][1]]}
    check_teacher_endpoint(general, "https://api.deepseek.com/v1", "openai")
    assert is_local_url("http://127.0.0.1:1") and not is_local_url("https://8.8.8.8/")


def test_guard_records():
    guard_records(CLEAN)
    with pytest.raises(DataGuardError, match="tw_national_id"):
        guard_records([{"prompt": "A223456789"}], "prompts.jsonl")


def test_guard_cli(tmp_path):
    assert guard_cli(None) == (None, None)
    with pytest.raises(DataGuardError, match="--medical needs"):
        guard_cli(None, medical=True)
    p = tmp_path / "gen.json"
    p.write_text(json.dumps({"sources": [{"name": "w", "domain": "general"}]}))
    with pytest.raises(DataGuardError, match="declares no medical"):
        guard_cli(str(p), medical=True)
    m, rep = guard_cli(str(_manifest(tmp_path)), medical=True)
    assert rep["scanned"] == ["pairs"]


def test_sft_prepare_refuses(tmp_path):
    import sft_hf
    with pytest.raises(DataGuardError, match="--medical needs"):
        sft_hf.main(["prepare", "--records", "x.jsonl", "--tokenizer", "t",
                     "--out", str(tmp_path / "o"), "--medical"])
    p = _manifest(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text("{}\n")
    with pytest.raises(DataGuardError, match="not a medical source"):
        sft_hf.main(["prepare", "--records", str(other), "--tokenizer", "t",
                     "--out", str(tmp_path / "o"), "--data-manifest", str(p)])
    with pytest.raises(DataGuardError, match="research-only"):
        sft_hf.main(["train", "--model", "m", "--data", "d", "--out", "o",
                     "--data-manifest", str(_manifest(tmp_path, research_only=True))])


def test_distill_refuses_external(tmp_path):
    import distill_generate
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "1", "prompt": CLEAN[0]}, ensure_ascii=False) + "\n")
    with pytest.raises(DataGuardError, match="external teacher"):
        distill_generate.main(["generate", "--prompts", str(prompts), "--out",
                               str(tmp_path / "a.jsonl"), "--endpoint", "openai",
                               "--base-url", "https://api.deepseek.com/v1",
                               "--data-manifest", str(_manifest(tmp_path))])
    with pytest.raises(DataGuardError, match="--medical needs"):
        distill_generate.main(["generate", "--prompts", str(prompts), "--out",
                               str(tmp_path / "a.jsonl"), "--medical"])


def test_cpt_refuses():
    pytest.importorskip("torch")
    import cpt_hf
    with pytest.raises(DataGuardError, match="--medical needs"):
        cpt_hf.main(["--model", "m", "--data", "d", "--tokens", "1", "--out", "o",
                     "--medical"])
