"""v5.155 (2026-10-26): fold upstream's pytest-style sidecar suites into
the daily TESTS registry.

Gap: upstream lands new subsystems with standalone pytest files
(test_zhtw_corpus.py / T148, test_api_accounting.py — both v5.154) that
CI's pytest step collects but the mandated daily gate
(`python -m helioslm_v5.tests.test_v5`) never executes. A regression in
either subsystem would pass the daily gate unnoticed.

Each wrapper below calls the sidecar file's test functions directly
(parametrized ones get their full parameter lists; the tmp_path fixture
gets a real TemporaryDirectory) and — loud-error — pins the exact set of
test_* functions it covers, so the day upstream ADDS a test function the
daily suite fails with a clear "register it" message instead of silently
under-covering.
"""

import tempfile
from pathlib import Path

from evastar.tests.case_common import _pass


def _drift_guard(module, expected):
    got = {n for n in vars(module) if n.startswith("test_")}
    assert got == expected, (
        f"{module.__name__} test functions drifted: "
        f"new={sorted(got - expected)} gone={sorted(expected - got)} — "
        "register the new ones in case_upstream_suites.py")


def test_zhtw_corpus_suite():
    """TW2 zh-TW corpus pipeline (T148): exact+MinHash dedup, attribution
    rules, converter-before-variant ordering, and the CLI end to end on
    local files. test_opencc self-skips (prints SKIP, returns) when
    OpenCC is absent."""
    from evastar.tests import test_zhtw_corpus as m
    expected = {"test_keys_and_minhash", "test_deduper", "test_process",
                "test_cli", "test_opencc"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_zhtw_corpus_suite",
          "T148 zh-TW corpus: all 5 sidecar tests executed under the "
          "daily gate (opencc case self-skips without OpenCC)")


def test_api_accounting_suite():
    """Offline API cost/provenance audit (v5.154): cache-bucket
    accounting against a hand-computed invoice, invalid-usage and
    invalid-rate rejection, per-task provenance, and the no-network CLI.
    The sidecar file runs under the v5.155 pytest shim when pytest is
    absent; parametrized cases are called with their full lists."""
    from evastar.tests import test_api_accounting as m
    expected = {
        "test_mixed_ttl_cost_matches_hand_invoice_without_double_counting",
        "test_anthropic_input_is_exclusive_of_cached_tokens",
        "test_chat_completions_explicit_zero_cache_and_free_rates",
        "test_incomplete_or_mismatched_data_has_no_total",
        "test_invalid_token_counts_are_rejected",
        "test_impossible_totals_and_partial_ttl_breakdowns",
        "test_invalid_rates_rejected",
        "test_empty_rate_table_and_legacy_aggregates_remain_unknown",
        "test_provenance_and_outcomes_are_per_task_not_per_call",
        "test_cli_reads_saved_responses_without_credentials_or_network",
    }
    _drift_guard(m, expected)
    m.test_mixed_ttl_cost_matches_hand_invoice_without_double_counting()
    m.test_anthropic_input_is_exclusive_of_cached_tokens()
    m.test_chat_completions_explicit_zero_cache_and_free_rates()
    for fault in ["usage", "details", "write", "rates",
                  "rate", "model", "tier", "context"]:
        m.test_incomplete_or_mismatched_data_has_no_total(fault)
    for value in [-1, True, 0.5, "100"]:
        m.test_invalid_token_counts_are_rejected(value)
    m.test_impossible_totals_and_partial_ttl_breakdowns()
    for value in ["NaN", "Infinity", "-1", True]:
        m.test_invalid_rates_rejected(value)
    m.test_empty_rate_table_and_legacy_aggregates_remain_unknown()
    m.test_provenance_and_outcomes_are_per_task_not_per_call()
    with tempfile.TemporaryDirectory() as t:
        m.test_cli_reads_saved_responses_without_credentials_or_network(
            Path(t))
    _pass("test_api_accounting_suite",
          "v5.154 API accounting: all 10 sidecar tests (incl. 8+4+4 "
          "parametrized cases and the no-network CLI) under the daily "
          "gate")
