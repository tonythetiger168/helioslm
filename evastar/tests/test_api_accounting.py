"""Offline oracles: cache buckets, rate applicability and provenance."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evastar.eval.api_accounting import build_report, estimate_cost, normalize_usage


def response():
    return {"model": "fixture-snapshot", "service_tier": "default",
            "system_fingerprint": "fixture-fingerprint", "created": 123,
            "usage": {"input_tokens": 12000, "output_tokens": 100,
                      "input_tokens_details": {
                          "cached_tokens": 4000, "cache_write_tokens": 7000,
                          "cache_write_tokens_details": {
                              "ttl_30m_tokens": 3000, "ttl_12h_tokens": 4000}}}}


def rates():
    # Synthetic oracle prices; not a claim about any actual model's rates.
    return {"model": "fixture-snapshot", "provider": "openai",
            "as_of": "2026-10-25", "source": "fixture://synthetic-rates",
            "service_tier": "default", "max_input_tokens": 20000,
            "rates": {"input": "1", "cache_read": "0.1",
                      "cache_write_30m": "1.25", "cache_write_12h": "2.5",
                      "output": "10"}}


def test_mixed_ttl_cost_matches_hand_invoice_without_double_counting():
    result = estimate_cost(response(), "openai", rates())
    # Input: (1000 + 400 + 3750 + 10000) / 1M; output: 1000 / 1M.
    assert result["buckets"]["input"] == 1000
    assert result["cost_usd"] == "0.01615"
    assert result["cost_status"] == "estimated"
    assert "cache_write" not in result["buckets"]


def test_anthropic_input_is_exclusive_of_cached_tokens():
    r = {"model": "fixture-snapshot", "usage": {
        "input_tokens": 1000, "output_tokens": 100,
        "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 7000,
        "cache_creation": {"ephemeral_5m_input_tokens": 3000,
                           "ephemeral_1h_input_tokens": 4000}}}
    card = rates()
    card.pop("service_tier")
    card["provider"] = "anthropic"
    card["rates"].update(cache_write_5m="1.25", cache_write_1h="2")
    result = estimate_cost(r, "anthropic", card)
    assert result["input_tokens_total"] == 12000
    assert result["buckets"]["input"] == 1000
    assert result["cost_usd"] == "0.01415"


def test_chat_completions_explicit_zero_cache_and_free_rates():
    r = {"model": "fixture-snapshot", "service_tier": "default",
         "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                   "prompt_tokens_details": {"cached_tokens": 0,
                                             "cache_write_tokens": 0}}}
    card = rates()
    card["rates"] = {"input": 0, "output": 0}
    assert estimate_cost(r, "openai", card)["cost_usd"] == "0"


@pytest.mark.parametrize("fault", ["usage", "details", "write", "rates",
                                   "rate", "model", "tier", "context"])
def test_incomplete_or_mismatched_data_has_no_total(fault):
    r, card = response(), rates()
    if fault == "usage":
        r.pop("usage")
    elif fault == "details":
        r["usage"].pop("input_tokens_details")
    elif fault == "write":
        r["usage"]["input_tokens_details"].pop("cache_write_tokens")
    elif fault == "rates":
        card = None
    elif fault == "rate":
        card["rates"].pop("cache_write_12h")
    elif fault == "model":
        r["model"] = "different-snapshot"
    elif fault == "tier":
        r["service_tier"] = "different-tier"
    else:
        card["max_input_tokens"] = 100
    result = estimate_cost(r, "openai", card)
    assert result["cost_usd"] is None and result["reasons"]


@pytest.mark.parametrize("value", [-1, True, 0.5, "100"])
def test_invalid_token_counts_are_rejected(value):
    r = response()
    r["usage"]["input_tokens"] = value
    with pytest.raises(ValueError):
        normalize_usage(r, "openai")


def test_impossible_totals_and_partial_ttl_breakdowns():
    r = response()
    r["usage"]["input_tokens"] = 100
    with pytest.raises(ValueError, match="exceed"):
        normalize_usage(r, "openai")
    r = response()
    split = r["usage"]["input_tokens_details"]["cache_write_tokens_details"]
    split["ttl_12h_tokens"] = 1
    with pytest.raises(ValueError, match="breakdown"):
        normalize_usage(r, "openai")
    split.pop("ttl_12h_tokens")
    assert estimate_cost(r, "openai", rates())["cost_usd"] is None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", True])
def test_invalid_rates_rejected(value):
    card = rates()
    card["rates"]["input"] = value
    with pytest.raises(ValueError, match="invalid rate"):
        estimate_cost(response(), "openai", card)


def test_empty_rate_table_and_legacy_aggregates_remain_unknown():
    card = rates()
    card["rates"] = {}
    assert estimate_cost(response(), "openai", card)["cost_usd"] is None
    legacy = {"model": "fixture-snapshot", "usage": {
        "prompt_tokens": 12000, "completion_tokens": 100}}
    assert estimate_cost(legacy, "openai", rates())["cost_status"] == "unknown"


def test_provenance_and_outcomes_are_per_task_not_per_call():
    record = {"provider": "openai", "requested_model": "fixture-alias",
              "response": response(), "task_id": "task-1"}
    original = copy.deepcopy(record)
    corpus = "臺灣測試\n".encode()
    report = build_report([record, record], corpus=corpus,
                          harness_revision="fixture-commit", tokenizer="tok@revision",
                          rate_card=rates(), outcomes={"task-1": True},
                          evaluation_settings={"seed": 7, "shots": 0, "samples": 2})
    assert record == original
    assert report["cost_usd"] == report["cost_per_correct_usd"] == "0.03230"
    assert report["n_calls"] == 2 and report["n_successful_tasks"] == 1
    assert report["provenance"]["corpus_sha256"] == hashlib.sha256(corpus).hexdigest()
    assert report["provenance"]["corpus_utf8_bytes"] == len(corpus)
    assert report["calls"][0]["reported_model"] == "fixture-snapshot"
    assert not report["provenance"]["snapshot_verified"]
    assert report["provenance"]["evaluation_settings"]["samples"] == 2
    for bad in ({"task-1": "yes"}, {}, {"extra": True}):
        with pytest.raises(ValueError, match="outcomes"):
            build_report([record], corpus=corpus, harness_revision="fixture",
                         outcomes=bad)
    failed = build_report([record], corpus=corpus, harness_revision="fixture",
                          rate_card=rates(), outcomes={"task-1": False})
    assert failed["cost_per_correct_usd"] is None


def test_cli_reads_saved_responses_without_credentials_or_network(tmp_path):
    root = Path(__file__).resolve().parents[2]
    records = [{"provider": "openai", "response": response()}]
    for name, obj in (("records", records), ("rates", rates())):
        (tmp_path / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    (tmp_path / "corpus.txt").write_text("臺灣", encoding="utf-8")
    code = (
        "import runpy,socket,sys; "
        "socket.socket=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network')); "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    r = subprocess.run([
        sys.executable, "-I", "-c", code, str(root / "examples/analyze_api_usage.py"),
        "--records", str(tmp_path / "records.json"), "--corpus", str(tmp_path / "corpus.txt"),
        "--harness-revision", "fixture-commit", "--rates", str(tmp_path / "rates.json"),
        "--out", str(tmp_path / "report.json")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["cost_usd"] == "0.01615"
    assert report["provenance"]["tokenizer"] is None
