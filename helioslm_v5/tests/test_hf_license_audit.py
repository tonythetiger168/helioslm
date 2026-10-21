"""T119 — roadmap TW1: dataset licence audit, offline (no network, no token).

Pins the licence classification (CC BY / BY-SA / CC0 / ODC-By / OGDL /
MIT / Apache commercial; any NC term noncommercial; ND flagged;
missing / "other" unknown; a list takes its most restrictive member), the
audit rows from an injected fetcher (order, sha, gated, a failing fetch
becomes an error row instead of aborting), the summary, that the module
imports without huggingface_hub, the dataset lists, and the committed
live audit (every id resolved; classes follow from the recorded licences).

Run from repo root: python3 helioslm_v5/tests/test_hf_license_audit.py
"""
import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import hf_license_audit as la  # noqa: E402


def test_classify():
    cases = {
        "cc-by-4.0": "commercial", "cc-by-sa-4.0": "commercial",
        "cc-by-sa-3.0": "commercial", "CC0-1.0": "commercial",
        "odc-by": "commercial", "odbl": "commercial", "mit": "commercial",
        "apache-2.0": "commercial", "bsd-3-clause": "commercial",
        "ogdl-taiwan-1.0": "commercial", "gfdl": "commercial",
        "cc-by-nc-4.0": "noncommercial", "cc-by-nc-sa-4.0": "noncommercial",
        "cc-by-nc-nd-4.0": "noncommercial",
        "cc-by-nd-4.0": "no_derivatives",
        "other": "unknown", "unknown": "unknown", None: "unknown",
        "": "unknown", "llama2": "unknown",
    }
    for lic, want in cases.items():
        assert la.classify(lic) == want, (lic, la.classify(lic))
    assert la.classify(["mit", "cc-by-nc-4.0"]) == "noncommercial"
    assert la.classify(["mit", "other"]) == "unknown"
    assert la.classify(["mit", "cc-by-4.0"]) == "commercial"
    assert la.classify(["cc-by-sa-3.0", "gfdl"]) == "commercial"  # Wikipedia
    assert la.classify([]) == "unknown"
    print("PASS test_classify")


def test_audit_with_fake_fetch():
    cards = {"a/open": {"license": "cc-by-4.0", "gated": False, "sha": "s1"},
             "b/nc": {"license": "cc-by-nc-sa-4.0", "gated": "auto",
                      "sha": "s2"},
             "c/none": {"license": None, "gated": False, "sha": "s3"}}

    def fetch(rid):
        if rid not in cards:
            raise FileNotFoundError(f"404 {rid}")
        return cards[rid]
    rows = la.audit(["a/open", "missing/x", "b/nc", "c/none"], fetch)
    assert [r["id"] for r in rows] == ["a/open", "missing/x", "b/nc", "c/none"]
    assert [r["class"] for r in rows] == ["commercial", "unknown",
                                          "noncommercial", "unknown"]
    assert rows[0]["sha"] == "s1" and rows[2]["gated"] == "auto"
    assert rows[1]["error"].startswith("FileNotFoundError") \
        and rows[1]["sha"] is None
    assert all(r["error"] is None for r in rows if r["id"] != "missing/x")
    s = la.summary(rows)
    assert s["commercial"] == ["a/open"] and s["noncommercial"] == ["b/nc"]
    assert s["unknown"] == ["missing/x", "c/none"]
    assert s["errors"] == ["missing/x"]
    print("PASS test_audit_with_fake_fetch")


def test_no_network_dependency_on_import():
    saved = sys.modules.pop("huggingface_hub", None)
    sys.modules["huggingface_hub"] = None          # import would fail
    try:
        mod = importlib.reload(la)
        assert mod.classify("mit") == "commercial"
    finally:
        del sys.modules["huggingface_hub"]
        if saved is not None:
            sys.modules["huggingface_hub"] = saved
        importlib.reload(la)
    print("PASS test_no_network_dependency_on_import")


def test_dataset_lists():
    ids = la.ALL
    assert len(ids) == len(set(ids))
    assert all(i.count("/") == 1 for i in ids)
    assert la.DATASETS_UNVERIFIED <= set(ids)
    print("PASS test_dataset_lists")


def test_committed_audit():
    """The live 2026-10-20 audit: every id resolved, every class follows
    from the recorded licence, hash."""
    import hashlib
    import json
    art = json.loads((_REPO / "benchmarks" /
                      "hf_license_audit_2026-10-20.json").read_text())
    rows = art["rows"]
    assert [r["id"] for r in rows] == list(la.ALL)
    assert all(r["error"] is None and r["sha"] for r in rows)
    assert all(r["class"] == la.classify(r["license"]) for r in rows)
    assert art["summary"] == la.summary(rows)
    assert art["rows_sha256"] == hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    print("PASS test_committed_audit " + json.dumps(
        {k: len(v) for k, v in art["summary"].items()}))


if __name__ == "__main__":
    test_classify()
    test_audit_with_fake_fetch()
    test_no_network_dependency_on_import()
    test_dataset_lists()
    test_committed_audit()
