"""T120 — distillation teacher licence audit (v5.82), offline.

Pins: every teacher has a reviewed reading with a verdict from VERDICTS
and at least one condition; a fetched licence text whose hash differs
from the reviewed one is refused (a changed licence must be re-read); an
unreviewed teacher is refused; and the committed live audit (hashes equal
the reviewed ones, only unreadable licences left unclear, no closed
model among the teachers, scope notes); the OS4 rows pinned from the
2026-10-23 live audit; the three MIT rows pinned from the 2026-10-24 one.

Run from repo root: python3 evastar/tests/test_teacher_license_audit.py
"""
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval import teacher_license_audit as ta  # noqa: E402

_ART = _REPO / "benchmarks" / "teacher_license_audit_2026-10-20.json"
_ART_OS4 = _REPO / "benchmarks" / "teacher_license_audit_2026-10-23.json"
_ART_1024 = _REPO / "benchmarks" / "teacher_license_audit_2026-10-24.json"
OS4 = ("tencent/Hy3", "XiaomiMiMo/MiMo-V2.5-Pro",
       "deepseek-ai/DeepSeek-V4-Flash", "zai-org/GLM-5.2")
NEW_1024 = ("deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "deepseek-ai/DeepSeek-V4-Pro-0813", "zai-org/GLM-5.3-Flash")


def test_reviews_complete():
    assert len(ta.TEACHERS) == 17 and len(ta.REVIEWS) == 17
    for rid, r in ta.REVIEWS.items():
        assert r["verdict"] in ta.VERDICTS, rid
        assert r["conditions"] and r["license"], rid
        assert "reviewed_sha256" in r, rid
    print("PASS test_reviews_complete")


def _fake(texts):
    def fetch(rid):
        return {"card_license": "x", "card_license_name": None, "sha": "s",
                "license_file": "LICENSE" if texts[rid] else None,
                "license_text": texts[rid]}
    return fetch


def test_changed_or_unreviewed_licence_is_refused():
    rid = "deepseek-ai/DeepSeek-V4.1-Flash"
    try:
        ta.audit([rid], _fake({rid: "MIT License, but edited"}))
    except ValueError as e:
        assert "changed since it was reviewed" in str(e)
    else:
        raise AssertionError("a changed licence was accepted")
    try:
        ta.audit(["someone/new-model"], _fake({"someone/new-model": "x"}))
    except KeyError:
        pass
    else:
        raise AssertionError("an unreviewed teacher was accepted")
    # the MiMo row has no LICENSE file: reviewed hash None matches None
    mimo = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
    row = ta.audit([mimo], _fake({mimo: None}))[0]
    assert row["license_sha256"] is None and row["verdict"] == "allowed"
    print("PASS test_changed_or_unreviewed_licence_is_refused")


def test_pending_rows_are_not_usable_until_pinned():
    # OS4: the five models of the 2026-10-22 review, pinned 2026-10-23
    assert ta.PENDING_TEACHERS == ()
    assert "MiniMaxAI/MiniMax-M3" in ta.TEACHERS  # already pinned
    for rid in OS4:
        assert rid in ta.TEACHERS and ta.usable(rid), rid
    # V4-Flash carries the same MIT text as V4.1-Flash
    assert ta.REVIEWS["deepseek-ai/DeepSeek-V4-Flash"]["reviewed_sha256"] \
        == ta.REVIEWS["deepseek-ai/DeepSeek-V4.1-Flash"]["reviewed_sha256"]
    assert ta.usable("deepseek-ai/DeepSeek-V4.1-Flash")
    # Nemotron 3 Ultra: OpenMDW-1.1 read from its canonical text (v5.128)
    nemo = ta.REVIEWS["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"]
    assert ta.usable("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    assert len(nemo["license_text_sha256"]) == 64
    assert not any(r["verdict"] == "unclear" for r in ta.REVIEWS.values())
    assert not ta.usable("someone/new-model")
    # a live audit of a pending row records the hash without refusing, and
    # pin_pending hands back entries with the reading unchanged
    rid = "someone/pending-model"
    ta.REVIEWS[rid] = {"reviewed_sha256": ta.PENDING, "license": "MIT",
                       "verdict": "allowed", "conditions": ["keep notice"]}
    try:
        assert not ta.usable(rid)
        texts = {rid: f"licence of {rid}"}
        rows = ta.audit([rid], _fake(texts))
        assert all(not r["pinned"] for r in rows)
        pins = ta.pin_pending(rows)
    finally:
        del ta.REVIEWS[rid]
    for rid, e in pins.items():
        assert e["reviewed_sha256"] == hashlib.sha256(texts[rid].encode()).hexdigest()
        assert e["verdict"] == "allowed"
    print("PASS test_pending_rows_are_not_usable_until_pinned")


def test_committed_audit():
    art = json.loads(_ART.read_text())
    rows = art["rows"]
    assert [r["id"] for r in rows] == [t for t in ta.TEACHERS
                                       if t not in OS4 + NEW_1024]
    for r in rows:
        assert r["license_sha256"] == ta.REVIEWS[r["id"]]["reviewed_sha256"]
        assert r["verdict"] in ta.VERDICTS
    # only a licence whose text could not be read is left unclear
    for r in rows:
        if r["verdict"] == "unclear":
            assert r["license_sha256"] is None, r["id"]
    # open-weight releases only: OpenAI's gpt-oss and Google's Gemma are
    # fine, their closed models (whose terms forbid this use) are not
    closed = ("claude", "gpt-5", "gpt-6", "gemini")
    assert not any(c in r["id"].lower() for r in rows for c in closed)
    assert any("terms of service" in n for n in art["scope_notes"])
    assert art["rows_sha256"] == hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    # the OS4 rows' live audit (2026-10-23): hashes equal the pins
    art = json.loads(_ART_OS4.read_text())
    assert [r["id"] for r in art["rows"]] == list(OS4)
    for r in art["rows"]:
        assert r["pinned"] and len(r["sha"]) == 40, r["id"]
        assert r["license_sha256"] == ta.REVIEWS[r["id"]]["reviewed_sha256"]
    assert art["rows_sha256"] == hashlib.sha256(
        json.dumps(art["rows"], sort_keys=True).encode()).hexdigest()
    print("PASS test_committed_audit")


def test_2026_10_24_rows():
    # the newer MIT releases found by the 2026-10-24 re-survey
    v41 = ta.REVIEWS["deepseek-ai/DeepSeek-V4.1-Flash"]["reviewed_sha256"]
    for rid in NEW_1024:
        assert rid in ta.TEACHERS and ta.usable(rid), rid
        assert ta.REVIEWS[rid]["verdict"] == "allowed", rid
    for rid in NEW_1024[:2]:
        assert ta.REVIEWS[rid]["reviewed_sha256"] == v41, rid
    assert ta.REVIEWS["zai-org/GLM-5.3-Flash"]["license"].startswith("MIT")
    art = json.loads(_ART_1024.read_text())
    assert [r["id"] for r in art["rows"]] == list(NEW_1024)
    for r in art["rows"]:
        assert r["pinned"] and len(r["sha"]) == 40, r["id"]
        assert r["license_sha256"] == ta.REVIEWS[r["id"]]["reviewed_sha256"]
    assert art["rows_sha256"] == hashlib.sha256(
        json.dumps(art["rows"], sort_keys=True).encode()).hexdigest()
    print("PASS test_2026_10_24_rows")


if __name__ == "__main__":
    test_reviews_complete()
    test_changed_or_unreviewed_licence_is_refused()
    test_pending_rows_are_not_usable_until_pinned()
    test_committed_audit()
    test_2026_10_24_rows()
