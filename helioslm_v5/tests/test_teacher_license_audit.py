"""T120 — distillation teacher licence audit (v5.82), offline.

Pins: every teacher has a reviewed reading with a verdict from VERDICTS
and at least one condition; a fetched licence text whose hash differs
from the reviewed one is refused (a changed licence must be re-read); an
unreviewed teacher is refused; and the committed live audit (hashes equal
the reviewed ones, only unreadable licences left unclear, no closed
model among the teachers, scope notes).

Run from repo root: python3 helioslm_v5/tests/test_teacher_license_audit.py
"""
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import teacher_license_audit as ta  # noqa: E402

_ART = _REPO / "benchmarks" / "teacher_license_audit_2026-10-20.json"


def test_reviews_complete():
    assert len(ta.TEACHERS) == 10 and len(ta.REVIEWS) == 14
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
    # OS4: the five models of the 2026-10-22 review
    assert set(ta.PENDING_TEACHERS) == {
        "tencent/Hy3", "XiaomiMiMo/MiMo-V2.5-Pro",
        "deepseek-ai/DeepSeek-V4-Flash", "zai-org/GLM-5.2"}
    assert "MiniMaxAI/MiniMax-M3" in ta.TEACHERS  # already pinned
    assert not set(ta.PENDING_TEACHERS) & set(ta.TEACHERS)
    for rid in ta.PENDING_TEACHERS:
        assert not ta.usable(rid), rid
    assert ta.usable("deepseek-ai/DeepSeek-V4.1-Flash")
    # Nemotron 3 Ultra: OpenMDW-1.1 read from its canonical text (v5.128)
    nemo = ta.REVIEWS["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"]
    assert ta.usable("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    assert len(nemo["license_text_sha256"]) == 64
    assert not any(r["verdict"] == "unclear" for r in ta.REVIEWS.values())
    assert not ta.usable("someone/new-model")
    # a live audit of pending rows records the hash without refusing, and
    # pin_pending hands back entries with the reading unchanged
    texts = {r: f"licence of {r}" for r in ta.PENDING_TEACHERS}
    rows = ta.audit(ta.PENDING_TEACHERS, _fake(texts))
    assert all(not r["pinned"] for r in rows)
    pins = ta.pin_pending(rows)
    for rid, e in pins.items():
        assert e["reviewed_sha256"] == hashlib.sha256(texts[rid].encode()).hexdigest()
        assert e["verdict"] == ta.REVIEWS[rid]["verdict"]
    print("PASS test_pending_rows_are_not_usable_until_pinned")


def test_committed_audit():
    art = json.loads(_ART.read_text())
    rows = art["rows"]
    assert [r["id"] for r in rows] == list(ta.TEACHERS)
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
    print("PASS test_committed_audit")


if __name__ == "__main__":
    test_reviews_complete()
    test_changed_or_unreviewed_licence_is_refused()
    test_pending_rows_are_not_usable_until_pinned()
    test_committed_audit()
