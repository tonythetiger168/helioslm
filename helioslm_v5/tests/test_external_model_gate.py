"""External reference model gate (#185): source pin, per-use licence
reading, L128/L256/L512 memory fit. Offline: no token, GPU or download."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.eval import external_model_gate as g  # noqa: E402
from helioslm_v5.eval import teacher_license_audit as ta  # noqa: E402

RID = "mistralai/Mistral-Large-4-1T-A52B"
C = g.CANDIDATES[RID]


def _live(**over):
    row = {"sha": C["revision"], "card_license": C["card_license"],
           "license_sha256": C["license_sha256"]}
    row.update(over)
    return lambda rid: row


def test_pinned_source_passes():
    assert g.check_source(RID, _live()) == {
        "id": RID, "status": "pending_review", "pinned": True}


@pytest.mark.parametrize("over", [
    {"sha": "0" * 40},
    {"card_license": "mit"},
    {"license_sha256": hashlib.sha256(b"edited licence").hexdigest()},
    {"license_sha256": None},
])
def test_drift_fails_closed(over):
    with pytest.raises(g.GateError, match="re-read"):
        g.check_source(RID, _live(**over))


def test_unknown_model_is_refused():
    with pytest.raises(g.GateError, match="no pinned review"):
        g.check_source("someone/new-1T", _live())
    with pytest.raises(g.GateError):
        g.use_allowed("someone/new-1T", "internal_research")


def test_per_use_readings():
    assert set(C["uses"]) == set(g.USES)
    for use, (verdict, why) in C["uses"].items():
        assert verdict in g.USE_VERDICTS and why, use
    assert g.use_allowed(RID, "internal_research")
    assert not g.use_allowed(RID, "redistribute_weights")
    assert g.use_allowed(RID, "redistribute_weights", accept_conditions=True)
    for use in ("commercial_maas", "official_api", "train_on_outputs"):
        assert not g.use_allowed(RID, use, accept_conditions=True), use
    with pytest.raises(g.GateError, match="needs_license"):
        g.require_use(RID, "commercial_maas")
    with pytest.raises(g.GateError, match="unknown use"):
        g.use_allowed(RID, "deploy")


def test_not_a_distillation_teacher():
    assert C["status"] == "pending_review"
    assert RID not in ta.REVIEWS and not ta.usable(RID)
    from helioslm_v5.src.training import distill_gen as dg
    assert all(t["repo"] != RID for t in dg.TEACHERS.values())


def test_kv_cache_formula():
    # 72 layers x (512 latent + 64 rope) x 2 bytes
    assert g.kv_bytes_per_token(C["arch"]) == 72 * 576 * 2 == 82944


def test_fit_table_l128_l256_l512():
    rows = {(r["tier"], r["quant"]): r for r in g.fit_table(RID)}
    assert len(rows) == 15
    for r in rows.values():
        assert r["tested_runtime"] == "not run"
        assert not (r["fits_with_reserve"] and not r["fits_static"])
    # nothing fits 128 or 256 GB, not even IQ2_XXS (277 GiB)
    assert not any(r["fits_static"] for (t, _), r in rows.items()
                   if t in ("L128", "L256"))
    assert rows[("L512", "IQ2_XXS")]["fits_with_reserve"]
    assert rows[("L512", "Q2_K")]["fits_with_reserve"]
    q4 = rows[("L512", "Q4_K_M")]
    assert not q4["fits_static"] and q4["weights_gib"] == 591.2


def test_reserve_and_context_bounds():
    r = g.fit(RID, "Q2_K", "L512", context=131072)
    assert r["kv_gib"] == 10.1 and r["budget_gib"] == 384.0
    assert r["total_gib"] == pytest.approx(
        r["weights_gib"] + r["mmproj_gib"] + r["kv_gib"] + 4.0, abs=0.2)
    # 1M context: the latent cache alone is ~81 GiB and Q2_K no longer fits
    big = g.fit(RID, "Q2_K", "L512", context=1048576)
    assert big["fits_static"] and not big["fits_with_reserve"]
    with pytest.raises(ValueError):
        g.fit(RID, "Q2_K", "L512", context=2 * 1048576)
    with pytest.raises(KeyError):
        g.fit(RID, "Q2_K", "L1024")


def test_conclusion_is_never_feasible_without_a_run():
    c = g.conclusion(RID)
    assert c["status"] == "unverified"
    assert c["fits_on_paper"] == [("L512", "IQ2_XXS"), ("L512", "Q2_K")]
    assert g.conclusion(RID, tiers=("L128", "L256"))["status"] == "blocked"
