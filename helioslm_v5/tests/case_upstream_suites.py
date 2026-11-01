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

from helioslm_v5.tests.case_common import _pass


def _drift_guard(module, expected):
    got = {n for n in vars(module) if n.startswith("test_")}
    assert got == expected, (
        f"{module.__name__} test functions drifted: "
        f"new={sorted(got - expected)} gone={sorted(expected - got)} — "
        "register the new ones in case_upstream_suites.py")


_MISSING = object()


class _MonkeyPatch:
    """Subset of pytest's monkeypatch fixture used by the wrapped
    sidecar suites (limitations round 2026-10-29): ``setattr`` /
    ``setitem`` with LIFO ``undo``. The suites run sequentially in one
    process, so every user must undo before returning control."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append(("attr", obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, mapping.get(key, _MISSING)))
        mapping[key] = value

    def undo(self):
        while self._undo:
            kind, obj, name, old = self._undo.pop()
            if kind == "attr":
                setattr(obj, name, old)
            elif old is _MISSING:
                obj.pop(name, None)
            else:
                obj[name] = old


def test_zhtw_corpus_suite():
    """TW2 zh-TW corpus pipeline (T148): exact+MinHash dedup, attribution
    rules, converter-before-variant ordering, and the CLI end to end on
    local files. test_opencc self-skips (prints SKIP, returns) when
    OpenCC is absent."""
    from helioslm_v5.tests import test_zhtw_corpus as m
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
    from helioslm_v5.tests import test_api_accounting as m
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
        "test_report_model_identity_policy",
        "test_missing_or_mixed_reported_models_fail_identity_policy",
        "test_invalid_identity_allowlist_is_rejected",
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
    for expected in ["fixture-snapshot", "unexpected-target"]:
        with tempfile.TemporaryDirectory() as t:
            m.test_cli_reads_saved_responses_without_credentials_or_network(
                Path(t), expected)
    for allowed, status, eligible in [
            (None, "unverified", True),
            (["fixture-snapshot"], "matched", True),
            (["fixture-alias", "fixture-snapshot"], "matched", True),
            (["different-target"], "mismatch", False)]:
        m.test_report_model_identity_policy(allowed, status, eligible)
    m.test_missing_or_mixed_reported_models_fail_identity_policy()
    for bad in [[], "model", [""], [None], [1]]:
        m.test_invalid_identity_allowlist_is_rejected(bad)
    _pass("test_api_accounting_suite",
          "v5.154 API accounting: all 13 sidecar tests (incl. 8+4+4 "
          "parametrized cases, the v5.222 model-identity policy and the "
          "no-network CLI) under the daily gate")


def test_moe_dispatch_suite():
    """v5.106/v5.149 MoE dispatch acceptance (batched + grouped) under
    the daily gate.

    Gap (limitations round 2026-10-26): the pins that hold "batched" and
    "grouped" dispatch to the blocked reference — outputs and gradients,
    skewed routing with idle experts, blocked fallback, whole model,
    bf16 autocast — lived only in the pytest sidecar
    test_moe_batched_dispatch.py, which the mandated daily gate
    (`python -m helioslm_v5.tests.test_v5`) never executed. A regression
    in the grouped path would pass the daily gate unnoticed. This
    wrapper calls every sidecar test with its full parameter lists and
    drift-guards the function set, same pattern as the v5.155 wrappers
    above. The sidecar file runs under the pytest-optional shim when
    pytest is absent.
    """
    from helioslm_v5.tests import test_moe_batched_dispatch as m
    expected = {"test_config_default_and_validation",
                "test_batched_matches_blocked",
                "test_situ_zero_centered_norm",
                "test_skewed_routing_and_idle_experts",
                "test_fallback_when_expert_is_converted",
                "test_grouped_without_grouped_mm",
                "test_model_level",
                "test_bf16_autocast"}
    _drift_guard(m, expected)
    m.test_config_default_and_validation()
    for mode in m.MODES:
        for name in sorted(m.CASES):
            m.test_batched_matches_blocked(name, mode)
        m.test_situ_zero_centered_norm(mode)
        m.test_skewed_routing_and_idle_experts(mode)
        m.test_fallback_when_expert_is_converted(mode)
        m.test_bf16_autocast(mode)
    m.test_grouped_without_grouped_mm()
    m.test_model_level()
    _pass("test_moe_dispatch_suite",
          "v5.106/v5.149/v5.156 MoE dispatch: all 8 sidecar tests (incl. 5 "
          "cases x 2 modes parametrized matrix) under the daily gate")


def test_kda_recompute_suite():
    """v5.157 KDA recompute acceptance (T151) under the daily gate —
    another pytest-only sidecar the mandated gate never ran (same gap
    class as v5.155; limitations round 2026-10-29). Pins: config
    validation, outputs/state/gradients equal the saved path for both
    decay styles with and without packed documents, lower backward
    memory, unchanged eval/loop paths."""
    from helioslm_v5.tests import test_kda_recompute as m
    expected = {"test_config", "test_same_outputs_and_gradients",
                "test_saves_less", "test_no_grad_and_loop_unchanged"}
    _drift_guard(m, expected)
    m.test_config()
    for style in sorted(m.STYLES):
        for packed in (False, True):
            m.test_same_outputs_and_gradients(style, packed)
    m.test_saves_less()
    m.test_no_grad_and_loop_unchanged()
    _pass("test_kda_recompute_suite",
          "T151 KDA recompute: all 4 sidecar tests (2 styles x packed "
          "matrix) under the daily gate")


def test_moe_compile_suite():
    """v5.158/v5.159/v5.162 MoE-under-torch.compile acceptance (T152)
    under the daily gate (same sidecar gap; limitations round
    2026-10-29). Pins: the dispatch methods carry the dynamo-disable
    marker; a compiled (eager backend) model gives the eager loss and
    gradients for all three dispatch modes and under bf16 autocast."""
    from helioslm_v5.tests import test_moe_compile as m
    expected = {"test_dispatch_is_not_compiled", "test_compiled_equals_eager",
                "test_compiled_under_bf16_autocast"}
    _drift_guard(m, expected)
    m.test_dispatch_is_not_compiled()
    for dispatch in ("blocked", "batched", "grouped"):
        m.test_compiled_equals_eager(dispatch)
    m.test_compiled_under_bf16_autocast()
    _pass("test_moe_compile_suite",
          "T152 MoE compile: all 3 sidecar tests (3 dispatch modes) "
          "under the daily gate")


def test_kda_fla_suite():
    """v5.160 FLA chunk_kda acceptance (T153) under the daily gate
    (same sidecar gap; limitations round 2026-10-29). CPU suite: a
    stand-in chunk_kda in FLA's argument layout pins the handoff, the
    autocast dtype and the torch-kernel fallbacks. The monkeypatch
    fixture gets a _MonkeyPatch stand-in (undo in finally, so a failure
    cannot leak patched modules into the rest of the suite)."""
    from helioslm_v5.tests import test_kda_fla as m
    expected = {"test_config", "test_fla_matches_torch",
                "test_fla_runs_in_autocast_dtype",
                "test_torch_kernel_for_packed_documents",
                "test_torch_kernel_off_cuda_and_without_fla"}
    _drift_guard(m, expected)
    m.test_config()
    # Real pytest wraps fixture functions so direct calls fail; unwrap
    # (_fixture_function) in that case. Under the shim the decorator is
    # the identity and there is nothing to unwrap.
    fake_fla_fn = getattr(m.fake_fla, "_fixture_function", m.fake_fla)
    for fn in (m.test_fla_matches_torch, m.test_fla_runs_in_autocast_dtype,
               m.test_torch_kernel_for_packed_documents):
        mp = _MonkeyPatch()
        try:
            fn(fake_fla_fn(mp))
        finally:
            mp.undo()
    mp = _MonkeyPatch()
    try:
        m.test_torch_kernel_off_cuda_and_without_fla(mp)
    finally:
        mp.undo()
    _pass("test_kda_fla_suite",
          "T153 KDA FLA kernel: all 5 sidecar tests (stand-in chunk_kda, "
          "monkeypatch stand-in) under the daily gate")


def test_adaptive_batch_suite():
    """v5.161 adaptive micro-batch + grouped-matmul fallback acceptance
    (T154) under the daily gate (same sidecar gap; limitations round
    2026-10-29). The grouped_mm parametrization is pre-checked on
    HAS_GROUPED_MM so no skip exception crosses the shim boundary; the
    pilot probe step runs on CPU."""
    from helioslm_v5.tests import test_adaptive_batch_fallback as m
    expected = {"test_adaptive_batch_size", "test_fit_micro_batch",
                "test_pilot_probe_step", "test_safe_grouped_matmul",
                "test_safe_grouped_matmul_shapes"}
    _drift_guard(m, expected)
    for gb, exp in [(22, 4), (24, 4), (48, 8), (80, 8)]:
        mp = _MonkeyPatch()
        try:
            m.test_adaptive_batch_size(mp, gb, exp)
        finally:
            mp.undo()
    m.test_fit_micro_batch()
    m.test_pilot_probe_step()
    m.test_safe_grouped_matmul(False)
    skipped = ""
    if m.moe_fallback.HAS_GROUPED_MM:
        m.test_safe_grouped_matmul(True)
    else:
        skipped = " (grouped_mm absent on this torch: 1 case skipped)"
    m.test_safe_grouped_matmul_shapes()
    _pass("test_adaptive_batch_suite",
          f"T154 adaptive batch + grouped fallback: all 5 sidecar tests "
          f"(4 size rows) under the daily gate{skipped}")


def test_compile_graph_breaks_suite():
    """v5.165/v5.174 compile graph-break acceptance (T155) under the
    daily gate (same sidecar gap; limitations round 2026-10-29). This
    file imports no pytest at all — pure direct calls. Pins: RoPE and
    the dense pilot compile break-free, the MoE pilot breaks only at its
    disabled dispatch, compiled RoPE matches eager bitwise, the
    scatter_add load count equals bincount, and the KDA layer has no
    as_subclass breaks."""
    from helioslm_v5.tests import test_compile_graph_breaks as m
    expected = {"test_rope_compiles_without_breaks",
                "test_rope_eager_checks_kept",
                "test_dense_pilot_has_no_breaks",
                "test_moe_pilot_breaks_only_at_dispatch",
                "test_load_count_matches_bincount",
                "test_k3_pilot_no_subclass_breaks"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_compile_graph_breaks_suite",
          "T155 compile graph breaks: all 6 sidecar tests under the "
          "daily gate")
