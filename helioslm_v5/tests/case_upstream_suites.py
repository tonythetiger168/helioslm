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
    process, so every user must undo before returning control.
    2026-11-04: ``setenv`` / ``delenv`` for the OpenAI-adapter wire test
    in test_hsh_loop (environment variables go through the same
    item-undo machinery)."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append(("attr", obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, mapping.get(key, _MISSING)))
        mapping[key] = value

    def setenv(self, name, value):
        import os
        self.setitem(os.environ, name, value)

    def delenv(self, name):
        import os
        self._undo.append(("item", os.environ, name,
                           os.environ.get(name, _MISSING)))
        del os.environ[name]

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


# ---------------------------------------------------------------------------
# v5.225 (2026-11-02): fold the next sidecar batch into the daily gate —
# handover-resume (T87/v5.54 contract), knowledge rephrasing (v5.206),
# verifiable MCQ reward (v5.207), grounded-report grader (v5.214),
# report abstention (v5.216) and the PulmoGuard pre-registered bake-off
# gate (v5.218). Same gap class as v5.155: all six landed as pytest-only
# sidecars that the mandated `python -m helioslm_v5.tests.test_v5` never
# executed, so a regression in any of them would pass the daily gate
# unnoticed. The five pytest-importing files carry the v5.225 in-file
# shim (raises / raises(match) / approx / mark.parametrize — exactly the
# features each file uses); test_handover_resume.py imports no pytest at
# all. tmp_path fixtures get real TemporaryDirectory instances.
# ---------------------------------------------------------------------------


def test_handover_resume_suite():
    """T87 handover-resume faithfulness (v5.54 contract) under the daily
    gate. Pins: a handoff is accepted only for a bitwise-replayed strict
    oracle-plan prefix; a fresh AgentLoop resumes from the rendered
    packet and repeats zero completed work; stale state / goal tampering
    / truncated plans abort loudly; a restarting policy scores as goal
    drift; the committed artifact regenerates byte-for-byte."""
    from helioslm_v5.tests import test_handover_resume as m
    expected = {"test_packet_build_validate_roundtrip",
                "test_fresh_resume_executes_suffix_without_rework",
                "test_stale_state_goal_and_truncation_abort",
                "test_restart_policy_records_goal_drift",
                "test_committed_artifact_regenerates_and_registry"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_handover_resume_suite",
          "T87 handover-resume: all 5 sidecar tests under the daily gate")


def test_rephrase_suite():
    """v5.206 knowledge rephrasing under the daily gate. Pins: bounded
    ordered chunks, one stable-id prompt per (style, chunk) with the
    previous chunk's tail as context, fidelity refusals (Simplified,
    lost numbers, off-topic, near-copy, length ratio, truncation),
    assemble joins in order with licence+teacher attribution and drops
    counted, CLI round trip."""
    from helioslm_v5.tests import test_rephrase as m
    expected = {"test_split_chunks_bounds_and_order",
                "test_prompts_ids_context_and_normalize",
                "test_fidelity_pass_and_failures",
                "test_clean_output_drops_preamble",
                "test_assemble_joins_and_drops",
                "test_assemble_multichunk_order",
                "test_cli_round_trip",
                "test_plan_tokens"}
    _drift_guard(m, expected)
    m.test_split_chunks_bounds_and_order()
    m.test_prompts_ids_context_and_normalize()
    m.test_fidelity_pass_and_failures()
    m.test_clean_output_drops_preamble()
    m.test_assemble_joins_and_drops()
    m.test_assemble_multichunk_order()
    with tempfile.TemporaryDirectory() as t:
        m.test_cli_round_trip(Path(t))
    m.test_plan_tokens()
    _pass("test_rephrase_suite",
          "v5.206 rephrase: all 8 sidecar tests (incl. the CLI round "
          "trip) under the daily gate")


def test_mcq_reward_suite():
    """v5.207 verifiable multiple-choice reward for GRPO under the daily
    gate. Pins: task building refusals, last-answer-wins letter parsing
    (full-width letters, last-line fallback), reward 1/0/no-answer with
    the Simplified-zh penalty scoped to zh-TW tasks, teacher-majority
    task extraction from distillation records, eval-set decontamination,
    and the prepare-mcq CLI."""
    from helioslm_v5.tests import test_mcq_reward as m
    expected = {"test_make_task", "test_final_letter", "test_reward",
                "test_tasks_from_distill", "test_drop_contaminated",
                "test_prepare_mcq_cli"}
    _drift_guard(m, expected)
    m.test_make_task()
    m.test_final_letter()
    m.test_reward()
    m.test_tasks_from_distill()
    m.test_drop_contaminated()
    with tempfile.TemporaryDirectory() as t:
        m.test_prepare_mcq_cli(Path(t))
    _pass("test_mcq_reward_suite",
          "v5.207 MCQ reward: all 6 sidecar tests (incl. the prepare-mcq "
          "CLI) under the daily gate")


def test_grounded_report_suite():
    """v5.214 JSON -> report grounding grader under the daily gate.
    Pins: the 24 zh-TW/English fixtures' unverified/contradiction/
    omission counts, bound-vs-unbound sentence semantics, null-field
    abstention, byte-identical determinism, schema refusals, and the
    stubbed no-network runner."""
    from helioslm_v5.tests import test_grounded_report as m
    expected = {"test_fixture_coverage", "test_fixture",
                "test_example_schema_matches_data_file",
                "test_bound_sentence_uses_its_nodule",
                "test_null_field_and_abstention",
                "test_extract_units_and_categories",
                "test_deterministic", "test_schema_rejects",
                "test_summarize", "test_runner_stub"}
    _drift_guard(m, expected)
    m.test_fixture_coverage()
    for case in m.CASES:
        m.test_fixture(case)
    m.test_example_schema_matches_data_file()
    m.test_bound_sentence_uses_its_nodule()
    m.test_null_field_and_abstention()
    m.test_extract_units_and_categories()
    m.test_deterministic()
    for bad in [
            {"nodules": [{"lobe": "XYZ"}]},
            {"nodules": [{"lobe": "RUL", "type": "calcified-ish"}]},
            {"lung_rads": "5", "nodules": []},
            {"nodules": [{"id": "N1"}, {"id": "N1"}]},
            {"lung_rads": "1"}]:
        m.test_schema_rejects(bad)
    m.test_summarize()
    m.test_runner_stub()
    _pass("test_grounded_report_suite",
          f"v5.214 grounded report: all 11 sidecar tests (incl. "
          f"{len(m.CASES)} fixture cases and 5 schema refusals) under "
          f"the daily gate")


def test_report_abstain_suite():
    """v5.216 field-level abstention reward under the daily gate.
    Parametrized cases are called with their full lists, rebuilt from
    the module's own ZH_OK/EN_OK constants with the same replacements
    the sidecar's decorators use."""
    from helioslm_v5.tests import test_report_abstain as m
    expected = {"test_abstain_on_missing_is_positive",
                "test_number_on_missing_is_negative",
                "test_present_and_correct_is_positive",
                "test_hedge_is_an_answer",
                "test_silent_and_attribution",
                "test_contradiction_on_present_field_is_negative",
                "test_env_deterministic_rate_and_batch",
                "test_synth_findings_valid",
                "test_sft_records_and_cli"}
    _drift_guard(m, expected)
    for text in [m.ZH_OK, m.EN_OK]:
        m.test_abstain_on_missing_is_positive(text)
    for text in [m.ZH_OK.replace("體積未測量", "體積 435 立方公釐"),
                 m.EN_OK.replace("volume not measured", "volume 435 mm3")]:
        m.test_number_on_missing_is_negative(text)
        m.test_present_and_correct_is_positive(text)
    for text in [m.ZH_OK.replace("體積未測量", "體積未測量（約 435 立方公釐）"),
                 m.EN_OK.replace("volume not measured",
                                 "volume not measured, about 435 mm3")]:
        m.test_hedge_is_an_answer(text)
    m.test_silent_and_attribution()
    m.test_contradiction_on_present_field_is_negative()
    m.test_env_deterministic_rate_and_batch()
    m.test_synth_findings_valid()
    with tempfile.TemporaryDirectory() as t:
        m.test_sft_records_and_cli(Path(t))
    _pass("test_report_abstain_suite",
          "v5.216 report abstain: all 9 sidecar tests (incl. 4x2 "
          "parametrized cases and the SFT CLI) under the daily gate")


def test_pulmoguard_gate_suite():
    """v5.218 PulmoGuard pre-registered bake-off gate under the daily
    gate. Pins: the committed plan validates/locks and stays a draft,
    the full synthetic gate computation (3 seeds, paired bootstrap)
    passes a grounded model and fails fabrication / low ratings, the
    grader-derived metrics, and the missing-data refusals. All findings
    and ratings are synthetic."""
    from helioslm_v5.tests import test_pulmoguard_gate as m
    expected = {"test_committed_plan", "test_full_gate_passes",
                "test_one_fabrication_fails",
                "test_low_rating_fails_against_best_open_model",
                "test_report_metrics_counts_omissions_and_abstention",
                "test_refusals"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_pulmoguard_gate_suite",
          "v5.218 PulmoGuard gate: all 6 sidecar tests (synthetic "
          "findings/ratings only) under the daily gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-02): fold the five sidecars the 2026-11-02
# daily report named as the next EXEMPT-backlog triage candidates into
# the daily gate — the long-horizon agent bench (#174, v5.221), the
# external reference model gate (v5.220), LoRA SFT (v5.217), the hsh
# clinical preset (v5.223) and the medical-data PHI/licence guard
# (v5.215). Same gap class as v5.155: all five were pytest-only sidecars
# that the mandated `python -m helioslm_v5.tests.test_v5` never
# executed, so a regression in any of them would pass the daily gate
# unnoticed. Each file now carries the in-file pytest shim covering
# exactly the features it uses (fixture / raises(match) /
# mark.parametrize / approx / importorskip). tmp_path fixtures get real
# TemporaryDirectory instances; the monkeypatch fixture gets the
# _MonkeyPatch stand-in; the transformers-dependent tiny-HF LoRA cases
# are pre-checked on find_spec so no skip exception crosses the shim
# boundary.
# ---------------------------------------------------------------------------


def test_long_horizon_bench_suite():
    """#174 long-horizon agent / secure-coding bench (v5.221) under the
    daily gate. Pins: the correct scripted policy solves all five
    sandboxed tasks (2-10 steps) replay-clean with no violations; the
    seven negative controls are each caught; run_suite is deterministic
    and matches the committed artifact byte-for-byte; breaking the
    correct policy or un-catching a control fails loudly;
    evaluate_policy reports rates. The module-scoped `suite` fixture is
    called once here and its return value passed in; monkeypatch gets
    the _MonkeyPatch stand-in (undo in finally)."""
    from helioslm_v5.tests import test_long_horizon_bench as m
    expected = {"test_correct_policy_solves_everything",
                "test_negative_controls_are_caught",
                "test_deterministic_and_matches_artifact",
                "test_suite_fails_when_correct_policy_breaks",
                "test_suite_fails_when_a_control_is_not_caught",
                "test_evaluate_policy_reports_rates"}
    _drift_guard(m, expected)
    # Real pytest wraps fixture functions so direct calls fail; unwrap
    # (_fixture_function) in that case. Under the shim the decorator is
    # the identity and there is nothing to unwrap.
    suite_fn = getattr(m.suite, "_fixture_function", m.suite)
    suite = suite_fn()
    m.test_correct_policy_solves_everything(suite)
    m.test_negative_controls_are_caught(suite)
    m.test_deterministic_and_matches_artifact(suite)
    for fn in (m.test_suite_fails_when_correct_policy_breaks,
               m.test_suite_fails_when_a_control_is_not_caught):
        mp = _MonkeyPatch()
        try:
            fn(mp)
        finally:
            mp.undo()
    m.test_evaluate_policy_reports_rates()
    _pass("test_long_horizon_bench_suite",
          "#174 long-horizon bench: all 6 sidecar tests (module fixture "
          "called once, monkeypatch stand-in) under the daily gate")


def test_external_model_gate_suite():
    """v5.220 external reference model gate (Mistral Large 4) under the
    daily gate. Pins: the pinned source passes; any drift in sha / card
    licence / licence hash fails closed; an unknown model is refused;
    per-use licence readings; the candidate is not a distillation
    teacher; the KV-cache formula; the L128/L256/L512 fit table;
    reserve/context bounds; the conclusion never claims feasibility
    without a run. Parametrized drift cases are called with their full
    literal list (the same rows the decorator carries)."""
    import hashlib
    from helioslm_v5.tests import test_external_model_gate as m
    expected = {"test_pinned_source_passes", "test_drift_fails_closed",
                "test_unknown_model_is_refused", "test_per_use_readings",
                "test_not_a_distillation_teacher", "test_kv_cache_formula",
                "test_fit_table_l128_l256_l512",
                "test_reserve_and_context_bounds",
                "test_conclusion_is_never_feasible_without_a_run"}
    _drift_guard(m, expected)
    m.test_pinned_source_passes()
    for over in [{"sha": "0" * 40},
                 {"card_license": "mit"},
                 {"license_sha256": hashlib.sha256(
                     b"edited licence").hexdigest()},
                 {"license_sha256": None}]:
        m.test_drift_fails_closed(over)
    m.test_unknown_model_is_refused()
    m.test_per_use_readings()
    m.test_not_a_distillation_teacher()
    m.test_kv_cache_formula()
    m.test_fit_table_l128_l256_l512()
    m.test_reserve_and_context_bounds()
    m.test_conclusion_is_never_feasible_without_a_run()
    _pass("test_external_model_gate_suite",
          "v5.220 external model gate: all 9 sidecar tests (incl. 4 "
          "parametrized drift cases) under the daily gate")


def test_lora_suite():
    """v5.217 LoRA SFT under the daily gate. Pins: apply_lora wraps only
    the targeted projections (vision/audio excluded) and freezes
    everything else (step 0 equals the base model); two masked-SFT steps
    through sft_hf.train change only adapter tensors (base weights
    bit-identical); the merge oracle matches base+adapter and the
    adapter round-trips into a fresh wrapped copy; peft key names; rank
    validation. The transformers-dependent tiny-HF cases (llama/qwen2)
    are pre-checked on find_spec so no skip crosses the shim boundary;
    tmp_path gets a real TemporaryDirectory."""
    import importlib.util
    from helioslm_v5.tests import test_lora as m
    expected = {"test_apply_targets_and_identity",
                "test_two_steps_only_adapter_changes_and_merge_oracle",
                "test_peft_key_names", "test_rank_must_be_positive",
                "test_tiny_hf_models",
                "test_merge_lora_folds_in_place_without_copying_the_weight"}
    _drift_guard(m, expected)
    m.test_apply_targets_and_identity()
    m.test_merge_lora_folds_in_place_without_copying_the_weight()
    with tempfile.TemporaryDirectory() as t:
        m.test_two_steps_only_adapter_changes_and_merge_oracle(Path(t))
    m.test_peft_key_names()
    m.test_rank_must_be_positive()
    skipped = ""
    if importlib.util.find_spec("transformers") is not None:
        for arch in ("llama", "qwen2"):
            m.test_tiny_hf_models(arch)
    else:
        skipped = " (transformers not installed: 2 tiny-HF cases skipped)"
    _pass("test_lora_suite",
          f"v5.217 LoRA: all 6 sidecar tests under the daily gate{skipped}")


def test_hsh_clinical_suite():
    """v5.223 hsh clinical preset under the daily gate. Pins: no builtin
    tools mounted (bash is refused as unknown-tool and audited); global
    tools hidden; every call needs approval (7-row matrix, incl. a host
    auto-allow listener still being asked); an unlisted host tool is
    denied; the preset is locked (compose / create / live patch refused,
    policy/refused logged); egress is deny with an empty allow list; the
    audit table rebuilds from the log alone (incl. a save/load round
    trip and tamper detection) while requests still rebuild identically;
    other presets unchanged. tmp_path gets a real TemporaryDirectory;
    the parametrized approval matrix is called with its full literal
    rows (same rows as the decorator)."""
    from helioslm_v5.tests import test_hsh_clinical as m
    expected = {"test_no_builtin_tools_and_bash_refused",
                "test_global_tools_hidden",
                "test_every_call_needs_approval",
                "test_unlisted_host_tool_denied", "test_locked",
                "test_egress_deny", "test_end_to_end_audit_and_replay",
                "test_other_presets_unchanged"}
    _drift_guard(m, expected)
    with tempfile.TemporaryDirectory() as t:
        m.test_no_builtin_tools_and_bash_refused(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_global_tools_hidden(Path(t))
    for approval, auto, expect in [
            (None, False, False),                 # no approval service
            (lambda e, q: False, False, False),   # refused
            (lambda e, q: "yes", False, False),   # only True/dict approves
            (lambda e, q: False, True, False),    # host allow: still asks
            (None, True, False),                  # auto mode, no clinician
            (lambda e, q: True, False, True),
            (lambda e, q: {"approved": True, "by": "dr-lin"}, True, True)]:
        with tempfile.TemporaryDirectory() as t:
            m.test_every_call_needs_approval(Path(t), approval, auto, expect)
    with tempfile.TemporaryDirectory() as t:
        m.test_unlisted_host_tool_denied(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_locked(Path(t))
    m.test_egress_deny()
    with tempfile.TemporaryDirectory() as t:
        m.test_end_to_end_audit_and_replay(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_other_presets_unchanged(Path(t))
    _pass("test_hsh_clinical_suite",
          "v5.223 hsh clinical: all 8 sidecar tests (incl. the 7-row "
          "approval matrix) under the daily gate")


def test_data_guard_suite():
    """v5.215 PHI / licence guard for medical training data under the
    daily gate. Pins: the PHI scanner flags fake TW IDs / record numbers
    / phones / e-mail / date+name / DICOM tags (values masked, line
    numbers reported) while clean synthetic text passes; medical
    manifests missing deidentified / licence / origin are refused;
    research-only data is refused for releasable weights; a PHI hit in a
    medical source file stops guard(); external teachers are refused for
    medical data; the sft/cpt/distill CLIs refuse --medical without a
    manifest. All data is fake. tmp_path gets a real TemporaryDirectory;
    parametrized cases are called with their full lists (CLEAN/DIRTY
    tables from the module, the manifest-refusal rows literal)."""
    from helioslm_v5.tests import test_data_guard as m
    expected = {"test_clean_passes", "test_dirty_flagged",
                "test_masked_and_line_numbers",
                "test_clean_manifest_passes_general_not_scanned",
                "test_medical_source_without_fields_refused",
                "test_research_only", "test_phi_in_medical_file_stops",
                "test_missing_medical_file", "test_teacher_endpoint",
                "test_guard_records", "test_guard_cli",
                "test_sft_prepare_refuses", "test_distill_refuses_external",
                "test_cpt_refuses"}
    _drift_guard(m, expected)
    for text in m.CLEAN:
        m.test_clean_passes(text)
    for kind, text in m.DIRTY:
        m.test_dirty_flagged(kind, text)
    with tempfile.TemporaryDirectory() as t:
        m.test_masked_and_line_numbers(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_clean_manifest_passes_general_not_scanned(Path(t))
    for bad, msg in [({"deidentified": False}, "deidentified"),
                     ({"deidentified": None}, "deidentified"),
                     ({"license": ""}, "license"),
                     ({"origin": None}, "origin")]:
        with tempfile.TemporaryDirectory() as t:
            m.test_medical_source_without_fields_refused(Path(t), bad, msg)
    with tempfile.TemporaryDirectory() as t:
        m.test_research_only(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_phi_in_medical_file_stops(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_missing_medical_file(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_teacher_endpoint(Path(t))
    m.test_guard_records()
    with tempfile.TemporaryDirectory() as t:
        m.test_guard_cli(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_sft_prepare_refuses(Path(t))
    with tempfile.TemporaryDirectory() as t:
        m.test_distill_refuses_external(Path(t))
    m.test_cpt_refuses()
    _pass("test_data_guard_suite",
          "v5.215 data guard: all 14 sidecar tests (incl. 4+10+4 "
          "parametrized cases) under the daily gate")



def test_ot_bugfix_bench_suite():
    """v5.228 OpenTitan real-IP bug-fix / completion benchmark under the
    daily gate. Pins mutation sites (never comments or the port header),
    seeded single-site bugs, line-preserving masked blocks, the prompt,
    candidate filtering, the H5-manifest seen split and the summary; the
    tool-backed end-to-end case returns early without yosys / sv2v."""
    from helioslm_v5.tests import test_ot_bugfix_bench as m
    expected = {"test_mask_and_body_span", "test_bugfix_mutation_sites",
                "test_mutate_is_seeded_and_single_site",
                "test_complete_masks_one_block", "test_always_end_follows_else",
                "test_prompt_asks_for_whole_file", "test_candidates_and_seen_ips",
                "test_summarize_groups", "test_build_and_score_with_tools",
                "test_splice_snippet_answers", "test_edit_pairs_exclude_eval_ips"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_ot_bugfix_bench_suite",
          "v5.228 OpenTitan bug-fix bench: all 11 sidecar tests under the "
          "daily gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-04): fold the hsh harness core sidecars —
# the session log (H2), the tool registry/pipeline (H3), the LLM seam
# and turn driver (H4), and built-in tools / PTC mode / presets (H5) —
# into the daily gate. Same gap class as v5.155: four pytest-only
# sidecars the mandated `python -m helioslm_v5.tests.test_v5` never
# executed, while the README's flagship "Using the harness" surface and
# the v5.223 clinical preset (folded 2026-11-02) all build on this core.
# The four files carry the in-file pytest shim (raises /
# mark.parametrize — exactly the features each uses); tmp_path fixtures
# get real TemporaryDirectory instances; the OpenAI-adapter wire test's
# monkeypatch gets the _MonkeyPatch stand-in (setenv/delenv added for
# it). test_workspace_confinement's symlink probe is capability-gated in
# the sidecar itself (Windows without the symlink privilege still runs
# the parent-dir and absolute-path probes).
# ---------------------------------------------------------------------------


def test_hsh_session_suite():
    """hsh H2 session log under the daily gate. Pins: contiguous
    validated append (NaN / set / unknown type refused, data detached),
    derive_messages shape with system-head replace/clear, keep_thinking
    none/turn/all semantics, model-visible-means-logged (rebuilt
    requests hash-equal the hand-driven loop's, incl. a retry), unlogged
    input detection, header requirement, JSONL round trip + version
    refusal, interrupted-tail repair, turn-boundary fork, usage_summary,
    and the sessions service broadcast/unwind. The keep_thinking matrix
    is called with its full mode list; tmp_path gets a real
    TemporaryDirectory."""
    from helioslm_v5.tests import test_hsh_session as m
    expected = {"test_append_contiguous_and_validated",
                "test_register_event_type_log_only",
                "test_derive_history_shape",
                "test_system_head_replaced_and_cleared",
                "test_keep_thinking_modes",
                "test_model_visible_means_logged",
                "test_unlogged_input_is_detected",
                "test_request_needs_header",
                "test_jsonl_roundtrip_and_version",
                "test_repair_interrupted_tail",
                "test_fork_at_turn_boundary", "test_usage_summary",
                "test_sessions_service_broadcasts_and_unwinds"}
    _drift_guard(m, expected)
    for name in sorted(expected - {"test_model_visible_means_logged",
                                   "test_jsonl_roundtrip_and_version"}):
        getattr(m, name)()
    for mode in ("none", "turn", "all"):
        m.test_model_visible_means_logged(mode)
    with tempfile.TemporaryDirectory() as t:
        m.test_jsonl_roundtrip_and_version(Path(t))
    _pass("test_hsh_session_suite",
          "hsh H2 session log: all 13 sidecar tests (incl. the 3-mode "
          "keep_thinking matrix) under the daily gate")


def test_hsh_tools_suite():
    """hsh H3 tool registry / execution pipeline under the daily gate.
    Pins: the schema subset is enforced at registration, the validator's
    rules (bool is not an integer), schemas() shape, the pipeline order
    (pre-execute -> guards -> execute -> project -> post-execute ->
    finalize -> tools/result), deny/ask semantics incl. only True
    approving, guard monotonicity, post-execute replacement, input
    failures still finalizing, body/output failure codes, listener
    exception containment, deferred context + immutable args, scoped
    shadow/restrict/dispatch, plugin unwind, and the session payload.
    Both parametrized matrices are called with their full literal
    rows."""
    from helioslm_v5.tests import test_hsh_tools as m
    expected = {"test_schema_subset_enforced", "test_validator_rules",
                "test_schemas_only_model_fields_sorted",
                "test_happy_path_and_order", "test_deny_and_ask",
                "test_guards_are_monotonic", "test_post_execute_replaces",
                "test_input_failures_still_finalize",
                "test_body_and_output_failures",
                "test_listener_exception_is_contained",
                "test_deferred_context_and_args_not_mutable",
                "test_scoped_shadow_restrict_and_dispatch",
                "test_registrations_unwind_with_plugin",
                "test_session_payload"}
    _drift_guard(m, expected)
    for name in sorted(expected - {"test_deny_and_ask",
                                   "test_input_failures_still_finalize"}):
        getattr(m, name)()
    for decision, approval, expect_ran in [
            ({"action": "deny", "reason": "no"}, None, False),
            ({"action": "ask"}, None, False),
            ({"action": "ask"}, lambda e, q: False, False),
            ({"action": "ask"}, lambda e, q: "yes", False),
            ({"action": "ask"}, lambda e, q: True, True)]:
        m.test_deny_and_ask(decision, approval, expect_ran)
    for name, args, code in [
            ("add", "{bad json", "invalid-arguments"),
            ("add", '{"a": 1}', "invalid-arguments"),
            ("add", None, "invalid-arguments"),
            ("nope", "{}", "unknown-tool")]:
        m.test_input_failures_still_finalize(name, args, code)
    _pass("test_hsh_tools_suite",
          "hsh H3 tools pipeline: all 14 sidecar tests (incl. 5+4 "
          "parametrized rows) under the daily gate")


def test_hsh_loop_suite():
    """hsh H4 LLM seam / turn-step driver under the daily gate. Pins:
    the tool-turn event sequence with replay (model-visible means
    logged) across turns, retries, injected context and a system-prompt
    change; pre-step reject/rewrite; request config changes logged;
    turn-stopping continuation and max-steps; request-error retry/fail
    paths; scoped tools and deferred context; unknown provider; the
    llm/stream waterfall; the TextModelAdapter @@tool@@ protocol (incl.
    malformed blocks as text); render_transcript shape; and the
    OpenAICompatAdapter wire mapping offline (monkeypatch setenv/delenv
    via the _MonkeyPatch stand-in, undo in finally)."""
    from helioslm_v5.tests import test_hsh_loop as m
    expected = {"test_tool_turn_sequence_and_replay",
                "test_pre_step_reject_and_rewrite",
                "test_request_config_change_is_logged",
                "test_turn_stopping_continues_and_max_steps",
                "test_request_error_retry_and_fail",
                "test_scoped_tools_and_deferred_context",
                "test_unknown_provider_fails_turn",
                "test_llm_stream_waterfall_wraps_calls",
                "test_text_model_adapter_protocol",
                "test_text_model_malformed_block_is_text",
                "test_render_transcript_shape",
                "test_openai_wire_mapping_offline"}
    _drift_guard(m, expected)
    for name in sorted(expected - {"test_openai_wire_mapping_offline"}):
        getattr(m, name)()
    mp = _MonkeyPatch()
    try:
        m.test_openai_wire_mapping_offline(mp)
    finally:
        mp.undo()
    _pass("test_hsh_loop_suite",
          "hsh H4 loop/LLM seam: all 12 sidecar tests (monkeypatch "
          "stand-in for the wire test) under the daily gate")


def test_hsh_presets_suite():
    """hsh H5 built-in tools / PTC mode / presets under the daily gate.
    Pins: workspace confinement (parent-dir / absolute / symlink probes
    — the symlink probe is capability-gated in the sidecar for Windows
    hosts without the symlink privilege), bash exit/output/workdir/
    timeout/scrubbed env, the str_replace_editor and fs/todo tools, PTC
    code mode (only run_code visible, guards apply inside the program),
    preset tool isolation + persona, creator inspect, patches
    insert/replace/remove/disable and refusals, preset unwind with the
    Agents service, the minimal-preset end-to-end edit, and bash
    background jobs. tmp_path gets a real TemporaryDirectory."""
    from helioslm_v5.tests import test_hsh_presets as m
    expected = {"test_workspace_confinement", "test_bash", "test_editor",
                "test_fs_and_todo", "test_ptc_code_mode",
                "test_presets_isolate_agents_and_persona",
                "test_creator_inspect", "test_patches",
                "test_preset_unwinds_with_agents_service",
                "test_minimal_agent_edits_file_end_to_end",
                "test_bash_jobs"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        with tempfile.TemporaryDirectory() as t:
            getattr(m, name)(Path(t))
    _pass("test_hsh_presets_suite",
          "hsh H5 presets/builtins: all 11 sidecar tests under the daily "
          "gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-04): fold the remaining offline hsh core
# sidecars — the plugin kernel (H1) and the streaming / HTTP-error /
# request-retry pins in hsh/llm.py — into the daily gate. Same gap class
# and the same drift-wrapper as the H2–H5 fold above. Both files carry
# the in-file pytest shim; raises yields an ExceptionInfo stand-in
# because these cases read `.value`. The two llm wire tests take the
# _MonkeyPatch stand-in (setenv) and inject an opener, so they never
# touch the network. test_hsh_web stays in EXEMPT_SIDECARS: it binds
# 127.0.0.1 through ThreadingHTTPServer (ephemeral port), runs each turn
# on a background thread and sleeps on the SSE / cancel path, so it is
# not a clean offline fold.
# ---------------------------------------------------------------------------


def test_hsh_kernel_suite():
    """hsh H1 plugin kernel under the daily gate. Pins: provide / lookup /
    duplicate refused / withdrawn on dispose; inject waits, activates,
    unwinds and reactivates; effects unwind LIFO exactly once, including
    nested plugins; a failing apply rolls back and leaves the fork
    'failed'; settle failures are recorded not raised; Service subclass
    lifecycle and dependents seeing a started service; dispatch modes are
    contracts (undeclared / wrong mode raises); emit swallows and records;
    serial / parallel / bail; waterfall delegate, replace and
    short-circuit, including no listeners; scoped dispatch filter and
    unwind; a chain of services cascades. No fixtures: every case is a
    plain function."""
    from helioslm_v5.tests import test_hsh_kernel as m
    expected = {"test_provide_lookup_and_duplicate",
                "test_service_withdrawn_when_plugin_disposed",
                "test_inject_waits_activates_and_reactivates",
                "test_effects_unwind_lifo_once_and_nested",
                "test_disposer_runs_once_and_leaves_stack",
                "test_failed_apply_rolls_back",
                "test_failure_during_settle_is_recorded_not_raised",
                "test_service_subclass_lifecycle",
                "test_dependents_see_started_service",
                "test_modes_are_contracts",
                "test_emit_swallows_and_records",
                "test_serial_parallel_bail",
                "test_waterfall_delegate_replace_shortcircuit",
                "test_waterfall_without_listeners_runs_final",
                "test_scoped_dispatch_filter_and_unwind",
                "test_chain_of_services_cascades"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_hsh_kernel_suite",
          "hsh H1 plugin kernel: all 16 sidecar tests under the daily gate")


def test_hsh_llm_suite():
    """hsh streaming / HTTP errors / request retry under the daily gate.
    Pins: SSE deltas fold into one message (text, reasoning, tool-call
    fragments, usage incl. cache hits) and the loop re-emits them live
    while the log keeps the merged stream; HTTP 429/5xx are transient
    LLMErrors with Retry-After, 400 is not, URLError is transient;
    request-retry retries transient failures in the same step, gives up
    after max_retries, never retries a permanent error, and every attempt
    is logged. The wire tests inject an opener and never touch the
    network. monkeypatch.setenv goes through the _MonkeyPatch stand-in
    (undo in finally)."""
    from helioslm_v5.tests import test_hsh_llm as m
    expected = {"test_sse_fold", "test_streaming_through_the_loop",
                "test_http_errors", "test_request_retry", "test_sse_without_events"}
    _drift_guard(m, expected)
    for name in sorted(expected - {"test_streaming_through_the_loop",
                                   "test_http_errors"}):
        getattr(m, name)()
    for name in ("test_streaming_through_the_loop", "test_http_errors"):
        mp = _MonkeyPatch()
        try:
            getattr(m, name)(mp)
        finally:
            mp.undo()
    _pass("test_hsh_llm_suite",
          "hsh llm streaming/retry: all 4 sidecar tests (monkeypatch "
          "stand-in, injected opener, no network) under the daily gate")


def test_kv_cache_analytic_suite():
    """Published full-config KV-cache row (131072) under the daily gate.

    Folded the day it landed, same as test_ot_bugfix_bench: EXEMPT is
    the pre-guard backlog pending triage, and a one-function analytic
    pin (no fixtures, no network, no timing) is the shape these
    wrappers call. Leaving it exempt would let a cache-layout change
    pass the mandated gate while the README's 99.2% row went stale.
    """
    from helioslm_v5.tests import test_kv_cache_analytic as m
    expected = {"test_full_kv_cache_savings_at_128k"}
    _drift_guard(m, expected)
    m.test_full_kv_cache_savings_at_128k()
    _pass("test_kv_cache_analytic_suite",
          "full-config KV cache at 131072: published 1999.6 MB vs "
          "257698.0 MB (99.2%) locked via bench_cpu.cache_bytes")


# ---------------------------------------------------------------------------
# v5.225: sidecar coverage meta-guard. The folds above (and the earlier
# v5.155 / 2026-10-29 rounds) were each discovered by HAND, days after
# the sidecar landed. This guard makes the gap structurally impossible
# to miss: every helioslm_v5/tests/test_*.py file must be either FOLDED
# (executed through a wrapper in this module) or explicitly EXEMPT (the
# pre-guard legacy backlog, listed verbatim below — 189 files present on
# 2026-11-02 that have not yet been triaged for folding; shrinking that
# backlog is the daily maintenance job's standing candidate pool). The
# day upstream ADDS a sidecar, the daily gate fails loudly here naming
# the new file, instead of the file silently never running.
# 2026-11-02 limitations round: the five daily-report triage candidates
# (test_long_horizon_bench / test_external_model_gate / test_lora /
# test_hsh_clinical / test_data_guard) moved from EXEMPT to FOLDED —
# the backlog is now 184 files.
# 2026-11-04 limitations round: the four hsh harness core sidecars
# (test_hsh_session / test_hsh_tools / test_hsh_loop / test_hsh_presets)
# moved from EXEMPT to FOLDED — the backlog is now 180 files.
# 2026-11-04 limitations round: test_hsh_kernel (H1) and test_hsh_llm
# (streaming / HTTP errors / request retry) moved from EXEMPT to FOLDED —
# the backlog is now 178 files. test_hsh_web stays exempt: its cases bind
# 127.0.0.1 through ThreadingHTTPServer (ephemeral port), run the turn on
# a background thread and sleep on the SSE / cancel path.
# 2026-11-06 limitations round: test_hub (HF Hub offline round-trip),
# test_ruler_lite (OS7 RULER-lite tasks + probe arms) and
# test_review_fixes (T87 repo-review fix pins) moved from EXEMPT to
# FOLDED — the backlog is now 173 files.
# 2026-11-07 limitations round: test_prefix_pool_chained (chained block
# keys), test_pool_ledger (Phase 3.1 ledger oracles) and
# test_stop_calibration (T98 AC5 stop-decision metrics) moved from
# EXEMPT to FOLDED — the backlog is now 167 files.
# 2026-11-08 limitations round: test_package_tiers (T142/T166 packaging)
# and test_prereg (Plan E5 pre-registered evaluation) moved from EXEMPT
# to FOLDED — the backlog is now 160 files.
# OpenAI-compatible stream smoke (serving/openai_stream_smoke.py):
# test_openai_stream_smoke is exempt for the same reason as test_hsh_web.
# It binds 127.0.0.1 through ThreadingHTTPServer on an ephemeral port and
# reads a live SSE response (including a bounded timeout), so it is not a
# clean in-process fold for the daily TESTS list. pytest collects
# helioslm_v5/tests/test_openai_stream_smoke.py, which is what CI runs.
# This entry is an extra exemption on top of the 178-file legacy backlog,
# not a file from that snapshot.
# Worker launch env (serving/worker.env.example): test_worker_launch_env
# is folded the day it landed. It only reads checked-in text (no socket,
# no sleep, no network). Leaving it exempt would let a bad served name,
# a quantization flag, or a pasted token pass the daily gate.
# test_openai_stream_smoke stays exempt because it binds 127.0.0.1.
# ---------------------------------------------------------------------------

FOLDED_SIDECARS = {
    "test_zhtw_corpus": "test_zhtw_corpus_suite",
    "test_api_accounting": "test_api_accounting_suite",
    "test_moe_batched_dispatch": "test_moe_dispatch_suite",
    "test_kda_recompute": "test_kda_recompute_suite",
    "test_moe_compile": "test_moe_compile_suite",
    "test_kda_fla": "test_kda_fla_suite",
    "test_adaptive_batch_fallback": "test_adaptive_batch_suite",
    "test_compile_graph_breaks": "test_compile_graph_breaks_suite",
    "test_handover_resume": "test_handover_resume_suite",
    "test_rephrase": "test_rephrase_suite",
    "test_mcq_reward": "test_mcq_reward_suite",
    "test_grounded_report": "test_grounded_report_suite",
    "test_report_abstain": "test_report_abstain_suite",
    "test_pulmoguard_gate": "test_pulmoguard_gate_suite",
    # limitations round 2026-11-02 (daily-report triage candidates)
    "test_long_horizon_bench": "test_long_horizon_bench_suite",
    "test_external_model_gate": "test_external_model_gate_suite",
    "test_lora": "test_lora_suite",
    "test_hsh_clinical": "test_hsh_clinical_suite",
    "test_data_guard": "test_data_guard_suite",
    # v5.228 (2026-11-03): new sidecar, folded the day it landed
    "test_ot_bugfix_bench": "test_ot_bugfix_bench_suite",
    # limitations round 2026-11-04 (hsh harness core)
    "test_hsh_session": "test_hsh_session_suite",
    "test_hsh_tools": "test_hsh_tools_suite",
    "test_hsh_loop": "test_hsh_loop_suite",
    "test_hsh_presets": "test_hsh_presets_suite",
    # limitations round 2026-11-04 (hsh kernel + offline llm pins)
    "test_hsh_kernel": "test_hsh_kernel_suite",
    "test_hsh_llm": "test_hsh_llm_suite",
    # published KV-cache row: analytic pin, folded the day it landed
    "test_kv_cache_analytic": "test_kv_cache_analytic_suite",
    # worker launch env: file parse, folded the day it landed
    "test_worker_launch_env": "test_worker_launch_env_suite",
    # Mac MLX zips + page section (v5.245): tmp dirs and one CLI subprocess
    "test_mlx_mac": "test_mlx_mac_suite",
    # serving load test + monitoring stack (v5.249): file parse + pure math
    "test_serving_monitoring": "test_serving_monitoring_suite",
    # v5.251 (2026-11-06): daily-maintenance backlog triage
    "test_teacher_license_audit": "test_teacher_license_audit_suite",
    "test_pretrain_pilot": "test_pretrain_pilot_suite",
    "test_review_regressions": "test_review_regressions_suite",
    "test_cloudflare_bundles": "test_cloudflare_bundles_suite",
    # limitations round 2026-11-06 (hub / ruler-lite / review fixes)
    "test_hub": "test_hub_suite",
    "test_ruler_lite": "test_ruler_lite_suite",
    "test_review_fixes": "test_review_fixes_suite",
    # daily-maintenance round 2026-11-07 (v5.255): merge / MHC / BPE
    "test_merge": "test_merge_suite",
    "test_mhc": "test_mhc_suite",
    "test_mhc_sinkhorn": "test_mhc_sinkhorn_suite",
    "test_bpe": "test_bpe_suite",
    # limitations round 2026-11-07 (prefix pool / ledger / stop metrics)
    "test_prefix_pool_chained": "test_prefix_pool_chained_suite",
    "test_pool_ledger": "test_pool_ledger_suite",
    "test_stop_calibration": "test_stop_calibration_suite",
    # daily-maintenance round 2026-11-08 (v5.258): guard / think / docs
    "test_oracle_guard": "test_oracle_guard_suite",
    "test_think": "test_think_suite",
    "test_thinking_budget": "test_thinking_budget_suite",
    "test_three_modes": "test_three_modes_suite",
    "test_skill_docs": "test_skill_docs_suite",
    # limitations round 2026-11-08 (package tiers / prereg plans)
    "test_package_tiers": "test_package_tiers_suite",
    "test_prereg": "test_prereg_suite",
}

# Pre-guard legacy backlog (2026-11-02 snapshot), NOT yet triaged for
# folding. Being listed here is not an endorsement that a file should
# stay out of the gate; it only means "known, pending triage".
EXEMPT_SIDECARS = {
    'test_abstain', 'test_agent', 'test_agent_workflow', 'test_api_distill',
    'test_artifact_tiers', 'test_async_grpo', 'test_attn_res', 'test_base_gap',
    'test_bench_quant', 'test_benchmark', 'test_calibrated_prefetcher',
    'test_calibrated_router', 'test_calibration_board', 'test_chat', 'test_chat_tuned',
    'test_chunked_ce', 'test_clamped_swiglu', 'test_copy_curriculum',
    'test_cost_per_task', 'test_cpt_chunked_ce', 'test_cpt_pilot', 'test_cpt_plan',
    'test_csa', 'test_csa_learned', 'test_csa_learned_16x', 'test_csa_learned_h64',
    'test_csa_learned_r3', 'test_cuda_device_paths', 'test_cuda_smoke', 'test_custom_attention_notebook',
    'test_cvdp_bench', 'test_data_quality', 'test_decision',
    'test_decision_data', 'test_decision_head', 'test_decision_head_integration', 'test_decontam',
    'test_device_matrix', 'test_disagg', 'test_disagg_pareto', 'test_distill_agent',
    'test_distill_filter', 'test_distill_gen', 'test_distill_sft', 'test_dpo',
    'test_dsa', 'test_envs', 'test_file_env',
    'test_finetune_data', 'test_first_k_dense', 'test_gated_conv', 'test_gguf_quant',
    'test_global_prope_kv', 'test_gpu_batch_run', 'test_grounded_replay', 'test_grounding',
    'test_grpo_length', 'test_grpo_u1', 'test_harness', 'test_harness_deep',
    'test_hca_rescue', 'test_hdl_bench', 'test_hdl_data', 'test_hdl_git_sources',
    'test_hdl_projects', 'test_hdl_reward', 'test_hdl_train_eval', 'test_hf_license_audit',
    'test_hsh_egress', 'test_hsh_extras', 'test_hsh_interaction',
    'test_hsh_mcp',
    'test_hsh_skills',
    'test_hsh_web', 'test_indexer_share', 'test_k3_block',
    'test_k3_llamacpp_parts', 'test_kda_chunked', 'test_kda_k3_style', 'test_kda_subchunk',
    'test_kv_compress', 'test_kv_share', 'test_llamacpp_export', 'test_llamacpp_kimi_k3',
    'test_llamacpp_tokenizer', 'test_lm_decision_head', 'test_lm_harness', 'test_logit_kd',
    'test_longctx', 'test_longctx_harness',
    'test_midtrain',
    'test_mimo_hybrid', 'test_mini_bench', 'test_mla_every', 'test_mla_expanded_training',
    'test_moe_config_table', 'test_mopd', 'test_mtp_light', 'test_mtp_rollout',
    'test_multiseed', 'test_nope_temperature',     'test_noul_continuous', 'test_nvfp4_qat_e2e',
    # Binds 127.0.0.1 (ephemeral port); same exemption as test_hsh_web.
    'test_openai_stream_smoke',
    # v5.253: one case binds 127.0.0.1 (a llama-server stand-in); same
    # exemption as test_openai_stream_smoke. pytest still runs it in CI.
    'test_distill_local_3070',
    'test_option_matrix', 'test_ple',
    'test_probe_api', 'test_prune_kd_gemma4', 'test_quant_calib', 'test_quantile_balancing', 'test_qwen_next',
    'test_repo_longctx', 'test_rlcd_head',
    'test_rtllm_bench', 'test_selfgen_meta',
    'test_smebu_sandwich', 'test_std_bench', 'test_swarm_parallel',
    'test_swe_agent_rl', 'test_swe_bench', 'test_task_grammar',
    'test_term_bench',
    'test_tool_recovery', 'test_toy_bpe', 'test_trust_gate', 'test_trust_gate_clarify',
    'test_trust_gate_v2', 'test_tw_bench', 'test_uvm_bench', 'test_v110_github_pg',
    'test_v111_office', 'test_v112_niche', 'test_v113_media', 'test_v114_blender_omi',
    'test_v115_benchmark', 'test_v116_robot', 'test_v117_swarm', 'test_v118_robot_slam_vision',
    'test_v119_voice_ui_drive', 'test_v120_dna', 'test_v121_science', 'test_v122_crypto_cloud',
    'test_v123_spatial', 'test_v13_mid_model', 'test_v14_qwen', 'test_v15_math',
    'test_v16_code', 'test_v17_ic', 'test_v18_agent_plugins', 'test_v19_common',
    'test_v19_extended', 'test_v533_mid', 'test_v5_cases', 'test_v5_stage_a',
    'test_v5_stage_b', 'test_web_ui', 'test_web_ui_v2', 'test_zh_tokenizer_coverage',
    'test_zh_train_speed',
}


def test_worker_launch_env_suite():
    """Checked-in worker env (served name, MTP, lengths, no secrets).

    Folded the day it landed, same shape as test_kv_cache_analytic:
    one in-process function, no fixtures, no socket. EXEMPT_SIDECARS is
    the pre-guard backlog plus port-binding sidecars such as
    test_openai_stream_smoke; this file is neither.
    """
    from helioslm_v5.tests import test_worker_launch_env as m
    expected = {"test_worker_launch_env"}
    _drift_guard(m, expected)
    m.test_worker_launch_env()
    _pass("test_worker_launch_env_suite",
          "worker.env.example: helioslm, MTP x4, len 16384, "
          "16 seqs, no quant, no secret")


def test_serving_monitoring_suite():
    """Serving load-test summary math and monitoring config wiring (v5.249).

    Folded the day it landed: plain functions that read checked-in text,
    no socket, no sleep. Pins alert / dashboard metric names to vLLM 0.33.0.
    """
    from helioslm_v5.tests import test_serving_monitoring as m
    expected = {"test_summarize_math", "test_alert_and_dashboard_metrics_exist",
                "test_stack_wiring", "test_serving_monitoring"}
    _drift_guard(m, expected)
    m.test_serving_monitoring()
    _pass("test_serving_monitoring_suite",
          "loadtest summary, alerts/dashboard metrics in vLLM 0.33.0, stack wiring")


def test_mlx_mac_suite():
    """Mac (Apple Silicon) MLX download zips and page section (v5.245).

    Folded the day it landed: plain functions; the tmp_path ones get a
    real TemporaryDirectory each. No mlx install needed (the CLI test
    uses ready MLX directories)."""
    from helioslm_v5.tests import test_mlx_mac as m
    plain = ["test_mac_rows_cover_both_models", "test_quant_recipes"]
    with_tmp = ["test_bundle_mlx_zip", "test_bundle_mlx_checks_bits_and_weights",
                "test_mac_section_order_and_commands", "test_build_site_with_mac",
                "test_add_mac_to_built_site_replaces_earlier_section",
                "test_package_mlx_cli_with_ready_dirs",
                "test_mlx_config_fixes_from_per_layer_config"]
    _drift_guard(m, set(plain + with_tmp))
    for n in plain:
        getattr(m, n)()
    for n in with_tmp:
        with tempfile.TemporaryDirectory() as t:
            getattr(m, n)(Path(t))
    _pass("test_mlx_mac_suite",
          "MLX zip layout, Mac rows by memory, site section, CLI, config fix")
# ---------------------------------------------------------------------------
# v5.240 (2026-11-05): daily-maintenance fold — four more exempt-backlog
# sidecars whose subsystems moved in the last three days (teacher licence
# audit gained the Kimi-K3.1-Pro row in v5.239; the pretrain pilot's WSD
# schedule and the review-regression oracles were touched in the same
# window; Cloudflare bundles gained the local-run guide and measured-speed
# sections in v5.238). Same wrapper pattern as v5.225: drift-guarded
# function sets, parametrized cases called with their full lists,
# tmp_path as real TemporaryDirectory, monkeypatch via the stand-in.
# test_distill_gen stays exempt for now: its CLI test binds 127.0.0.1
# (same exemption class as test_hsh_web / test_openai_stream_smoke).
# ---------------------------------------------------------------------------


def test_teacher_license_audit_suite():
    """T120 distillation teacher licence audit (v5.82 + the v5.239
    Kimi-K3.1-Pro row) under the daily gate. Pins: every teacher has a
    reviewed reading with a verdict and conditions; a changed or
    unreviewed licence is refused; PENDING rows are unusable until
    pinned; the committed live audits (2026-10-20/23/24, OS4 rows, MIT
    rows) and the dated Kimi/Qwen rows. Offline: no pytest import, no
    network."""
    from helioslm_v5.tests import test_teacher_license_audit as m
    expected = {"test_reviews_complete",
                "test_changed_or_unreviewed_licence_is_refused",
                "test_pending_rows_are_not_usable_until_pinned",
                "test_committed_audit", "test_2026_10_24_rows",
                "test_2026_10_30_kimi_rows", "test_2026_11_03_qwen_row",
                "test_2026_11_05_kimi_k31_row", "test_2026_11_06_vendor_rows"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_teacher_license_audit_suite",
          "T120 teacher licence audit: all 9 sidecar tests (incl. the "
          "v5.239 Kimi-K3.1-Pro row and the v5.254 Gemma 4 / Grok 2 "
          "rows) under the daily gate")


def test_pretrain_pilot_suite():
    """T122 100M pilot pretraining loop (v5.83) under the daily gate —
    tiny and on CPU. Pins: every architecture builds with the routed
    arms within 25% of the dense active parameters; shard round-trip
    and uint16 overflow refusal; LR warmup/cosine and the WSD decay
    shapes; exact resume (bitwise) and a non-finite stop that resumes
    bitwise; the prepare-data script."""
    from helioslm_v5.tests import test_pretrain_pilot as m
    expected = {"test_configs", "test_shards", "test_lr_schedule",
                "test_wsd_schedule", "test_wsd_decay_data_switch",
                "test_exact_resume_and_learning",
                "test_nonfinite_stop_then_resume_is_bitwise",
                "test_prepare_data_script"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_pretrain_pilot_suite",
          "T122 pretrain pilot: all 8 sidecar tests (tiny CPU loops, "
          "bitwise resume) under the daily gate")


def test_review_regressions_suite():
    """Project-review P1/P2 regression oracles under the daily gate.
    Parametrized cases run with their full lists; tmp_path gets a real
    TemporaryDirectory; monkeypatch gets the _MonkeyPatch stand-in
    (undo in finally, so a failure cannot leak patched modules)."""
    from helioslm_v5.tests import test_review_regressions as m
    expected = {"test_cpt_rejects_batch_before_optimizer",
                "test_pretrain_rejects_non_finite_before_step",
                "test_prefix_eviction_does_not_break_generation",
                "test_mixed_kd_preserves_unaligned_and_masked_tokens",
                "test_pretrain_without_teacher_targets_matches_plain_ce",
                "test_session_store_concurrent_updates_are_atomic",
                "test_session_append_and_each_listener_receive_detached_events",
                "test_release_selection_imports"}
    _drift_guard(m, expected)
    for fault in ["loss", "nan_grad", "inf_grad", "remote"]:
        mp = _MonkeyPatch()
        try:
            m.test_cpt_rejects_batch_before_optimizer(fault, mp)
        finally:
            mp.undo()
    for fault in ["loss", "nan_grad", "inf_grad"]:
        mp = _MonkeyPatch()
        try:
            with tempfile.TemporaryDirectory() as t:
                m.test_pretrain_rejects_non_finite_before_step(
                    fault, Path(t), mp)
        finally:
            mp.undo()
    for capacity in [0, 1, 2]:
        for length in [6, 7]:
            m.test_prefix_eviction_does_not_break_generation(capacity,
                                                             length)
    for alpha in [0.0, 0.5, 1.0]:
        m.test_mixed_kd_preserves_unaligned_and_masked_tokens(alpha)
    mp = _MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as t:
            m.test_pretrain_without_teacher_targets_matches_plain_ce(
                Path(t), mp)
    finally:
        mp.undo()
    for second in ["append", "rename", "params"]:
        with tempfile.TemporaryDirectory() as t:
            m.test_session_store_concurrent_updates_are_atomic(
                Path(t), second)
    m.test_session_append_and_each_listener_receive_detached_events()
    with tempfile.TemporaryDirectory() as t:
        m.test_release_selection_imports(Path(t))
    _pass("test_review_regressions_suite",
          "review P1/P2 regressions: all 8 sidecar tests (incl. 4+3 "
          "fault rows, 3x2 eviction matrix, 3 KD alphas, 3 concurrency "
          "modes) under the daily gate")


class _TmpPathFactory:
    """Subset of pytest's tmp_path_factory used by the wrapped sidecar
    suites (v5.240): ``mktemp(name)`` makes and returns a subdirectory
    of one TemporaryDirectory owned by the caller."""

    def __init__(self, root):
        self._root = Path(root)

    def mktemp(self, name):
        p = self._root / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def test_cloudflare_bundles_suite():
    """T163 Cloudflare download bundles (v5.188 + the v5.238 local-run
    guide / measured-speed sections) under the daily gate. The
    module-scoped `dist` fixture (a lite model packaged into two tiers)
    is built once through a _TmpPathFactory stand-in and shared across
    the six tests that take it, matching pytest's module scope. No
    network: uploads are dry runs and every URL is a string check."""
    from helioslm_v5.tests import test_cloudflare_bundles as m
    expected = {"test_bundle_per_quant",
                "test_modelfile_points_at_its_own_gguf",
                "test_site_lists_every_tier_and_links_r2",
                "test_base_url_refusals", "test_cli_and_dry_run_upload",
                "test_site_has_local_run_guide",
                "test_site_has_measured_speed_and_mtp_tip"}
    _drift_guard(m, expected)
    # Real pytest wraps fixture functions so direct calls fail; unwrap
    # (_fixture_function) in that case. Under the shim the decorator is
    # the identity and there is nothing to unwrap.
    dist_fn = getattr(m.dist, "_fixture_function", m.dist)
    with tempfile.TemporaryDirectory() as dist_root:
        dist = dist_fn(_TmpPathFactory(dist_root))
        for fn in (m.test_bundle_per_quant,
                   m.test_modelfile_points_at_its_own_gguf,
                   m.test_site_lists_every_tier_and_links_r2,
                   m.test_cli_and_dry_run_upload,
                   m.test_site_has_local_run_guide,
                   m.test_site_has_measured_speed_and_mtp_tip):
            with tempfile.TemporaryDirectory() as t:
                fn(dist, Path(t))
    for url in ["https://github.com/x/y/releases",
                "https://objects.githubusercontent.com/a",
                "https://x.github.io/dl", "http://dl.example.com",
                "dl.example.com"]:
        m.test_base_url_refusals(url)
    _pass("test_cloudflare_bundles_suite",
          "T163 Cloudflare bundles: all 7 sidecar tests (two packaged "
          "tiers, 5 base-url refusals, dry-run upload) under the daily "
          "gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-06): fold three more exempt-backlog sidecars
# into the daily gate — the HF Hub offline round-trip, the RULER-lite
# tasks + probe arms, and the T87 repo-review fix pins. Same wrapper
# pattern as v5.225: drift-guarded function sets, parametrized cases
# called with their full lists, tmp_path as a real TemporaryDirectory,
# monkeypatch via the _MonkeyPatch stand-in (undo in finally, so a
# failure cannot leak the patched hub backend). test_hub and
# test_ruler_lite gained the in-file pytest shim (raises /
# mark.parametrize only); test_review_fixes needs no shim — plain
# asserts, and it inserts helioslm_v5/ on sys.path itself for the
# harness package.
# ---------------------------------------------------------------------------


def test_hub_suite():
    """HF Hub integration (hub.py) under the daily gate — fully offline.
    Pins: the local save/load round-trip is bitwise (weights AND forward
    output); from_hub refuses an untrusted remote unless
    trust_checkpoint=True; push_to_hub/from_hub against a mock backend
    (the _MonkeyPatch stand-in rewrites hub._hub; no network, no token);
    config.json rebuilds every preset exactly (nested dataclasses, tuples
    that JSON turns into lists)."""
    from helioslm_v5.tests import test_hub as m
    expected = {"test_hub_local_roundtrip",
                "test_hub_remote_requires_explicit_trust",
                "test_hub_upload_download_with_mock_backend",
                "test_hub_config_roundtrip_every_preset"}
    _drift_guard(m, expected)
    with tempfile.TemporaryDirectory() as t:
        m.test_hub_local_roundtrip(Path(t))
    m.test_hub_remote_requires_explicit_trust()
    mp = _MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as t:
            m.test_hub_upload_download_with_mock_backend(Path(t), mp)
    finally:
        mp.undo()
    m.test_hub_config_roundtrip_every_preset()
    _pass("test_hub_suite",
          "HF Hub offline round-trip: all 4 sidecar tests (bitwise local "
          "round-trip, trust gate, mock-backend upload/download, config "
          "presets) under the daily gate")


def test_ruler_lite_suite():
    """OS7 RULER-lite (eval/ruler_lite.py) under the daily gate. The
    parametrized task test is called once per task with its full length
    list (32/100/257 plus the determinism and too-short-refusal probes);
    the probe-arms test builds all four longctx arms at hidden=64 and
    runs one forward/backward each on CPU."""
    from helioslm_v5.tests import test_ruler_lite as m
    expected = {"test_tasks", "test_probe_arms_run"}
    _drift_guard(m, expected)
    for task in m.rl.TASKS:
        m.test_tasks(task)
    m.test_probe_arms_run()
    _pass("test_ruler_lite_suite",
          "RULER-lite: both sidecar tests (3 tasks x 3 lengths, seed "
          "determinism, too-short refusal, 4 probe arms fwd/bwd) under "
          "the daily gate")


def test_review_fixes_suite():
    """T87 repo-review fixes under the daily gate. No pytest and no
    fixtures: plain functions, tempfile only. Pins: defer_validation with
    a plain gate records TOOL_ERROR instead of crashing and a malformed
    finish does not end the run (AgentLoop and ChatSession); missing +
    wrong-typed args inspect as invalid -> ESCALATE (not CLARIFY); calc
    overflow / 0**-1 are ToolCallError; harness side-effect plugins fail
    closed without a TrustGate and only run on Route.DIRECT."""
    from helioslm_v5.tests import test_review_fixes as m
    expected = {"test_deferred_validation_with_plain_gate_does_not_crash",
                "test_missing_plus_invalid_escalates",
                "test_calc_arithmetic_faults_are_tool_errors",
                "test_harness_gates_fail_closed_and_block_clarify"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_review_fixes_suite",
          "T87 review fixes: all 4 sidecar tests (deferred validation, "
          "invalid-vs-missing routing, calc faults, fail-closed harness "
          "gates) under the daily gate")


# ---------------------------------------------------------------------------
# daily-maintenance round (2026-11-07, v5.255): fold four more
# exempt-backlog sidecars into the daily gate — checkpoint merging
# (linear/task-arithmetic/TIES + the merge_checkpoints CLI), the MHC
# residual stack and its Sinkhorn-constrained variant, and the byte-level
# BPE tokenizer. All four are plain functions: no pytest import, no
# fixtures, no sockets, no CUDA. test_merge.test_merge_script shells out
# to examples/merge_checkpoints.py with capture_output in BYTES (no
# text=), so there is no locale decode path to pin.
# ---------------------------------------------------------------------------


def test_merge_suite():
    """Checkpoint merging (src/training/merge.py) under the daily gate.
    Pins: linear/task-arithmetic/TIES oracles, validation refusals, the
    merge_checkpoints.py CLI round-trip (run.json carried across), and
    the top-k union KL / MOPD selectors."""
    from helioslm_v5.tests import test_merge as m
    expected = {"test_linear", "test_task_arithmetic", "test_ties",
                "test_validation", "test_merge_script",
                "test_topk_union_kl", "test_mopd_topk_union"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_merge_suite",
          "checkpoint merging: all 7 sidecar tests (3 oracle methods, "
          "validation, CLI round-trip, 2 selectors) under the daily gate")


def test_mhc_suite():
    """Manifold-constrained hyper-connections (src/blocks/mhc.py) under
    the daily gate. Pins: residual-equivalent init, gradient flow to all
    paths, manifold non-expansion, and per-T capacity differences."""
    from helioslm_v5.tests import test_mhc as m
    expected = {"test_residual_equivalent_init",
                "test_gradient_flows_to_all_paths",
                "test_manifold_non_expansion", "test_t_capacity_differs"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_mhc_suite",
          "MHC stack: all 4 sidecar tests under the daily gate")


def test_mhc_sinkhorn_suite():
    """Sinkhorn-constrained MHC wiring in the full model under the daily
    gate. Pins: off-by-default identity at init, constraint under random
    and extreme inputs and after training steps, decode == full forward,
    the static-HC fold regression, and validation refusals."""
    from helioslm_v5.tests import test_mhc_sinkhorn as m
    expected = {"test_off_by_default_and_identity_at_init",
                "test_constraint_random_and_extreme",
                "test_constraint_survives_training",
                "test_decode_equals_full",
                "test_static_hc_fold_regression", "test_validation"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_mhc_sinkhorn_suite",
          "Sinkhorn MHC: all 6 sidecar tests under the daily gate")


def test_bpe_suite():
    """Byte-level BPE tokenizer (src/tokenizer/bpe.py) under the daily
    gate. Pins: round-trip fidelity on domain strings, separator/space
    fidelity, OOV byte robustness, specials/BOS handling, save/load
    exactness, determinism + vocab cap, and corpus-builder protocol
    coverage."""
    from helioslm_v5.tests import test_bpe as m
    expected = {"test_roundtrip_domain_strings",
                "test_separator_and_space_fidelity",
                "test_oov_bytes_robust", "test_specials_and_bos",
                "test_save_load_exact", "test_determinism_and_vocab_cap",
                "test_corpus_builder_covers_protocol"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_bpe_suite",
          "BPE tokenizer: all 7 sidecar tests under the daily gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-07): fold three more exempt-backlog sidecars
# into the daily gate — the prefix-pool chained-keys regression, the
# pool-ledger oracles, and the stop-decision calibration metrics. All
# three are plain functions with no pytest, no fixtures, no sockets and
# no CUDA (the pool tests build a lite model on CPU and generate a few
# tokens; stop_calibration is pure eval logic over scripted agents).
# Same drift-guarded wrapper pattern as v5.225.
# ---------------------------------------------------------------------------


def test_prefix_pool_chained_suite():
    """PrefixPool chained block keys (src/inference/prefix_pool.py) under
    the daily gate. Pins the regression where two prompts sharing a
    middle block collided on the un-chained key: with chaining the
    shared block stores 6 distinct entries, A's exact prefix still hits
    after C was stored, and pooled generate stays bit-exact with
    from-scratch generate after cross-prompt block reuse."""
    from helioslm_v5.tests import test_prefix_pool_chained as m
    expected = {"test_prefix_pool_chained_keys"}
    _drift_guard(m, expected)
    m.test_prefix_pool_chained_keys()
    _pass("test_prefix_pool_chained_suite",
          "prefix pool chained keys: 1 sidecar test (6 distinct keys, "
          "prefix hit, bit-exact pooled generate) under the daily gate")


def test_pool_ledger_suite():
    """v5.42 prefix-pool ledger oracles (src/inference/pool_ledger.py,
    Phase 3.1) under the daily gate. Pins: cold-miss/warm-hit reuse
    counts and the 0.5 effective discount over two identical requests;
    partial-prefix reuse counts (4 reused / 2 computed); the empty
    ledger reports discount 0.0 with its scope recorded (no claim
    without requests)."""
    from helioslm_v5.tests import test_pool_ledger as m
    expected = {"test_ledger_counts_reuse", "test_ledger_partial_prefix",
                "test_report_scopes_and_loud_empty"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_pool_ledger_suite",
          "pool ledger: all 3 sidecar tests (reuse counts, partial "
          "prefix, loud empty report) under the daily gate")


def test_stop_calibration_suite():
    """T98 stop-decision metrics (eval/stop_calibration.py, roadmap
    Phase AC / AC5 知止) under the daily gate. Pins: hand-built rows
    with known answers (false_done, confident subset, ran_out,
    tested-before-done share, accuracy split, AUROC); the test-command
    detector; the defining SWE case — a trap patch whose visible tests
    pass is still a FALSE DONE while the careful agent has none; rows
    missing the episode fields are refused, not guessed."""
    from helioslm_v5.tests import test_stop_calibration as m
    expected = {"test_known_rows", "test_test_command_detector",
                "test_trap_is_false_done", "test_refuses_old_rows"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_stop_calibration_suite",
          "stop calibration: all 4 sidecar tests (known rows, command "
          "detector, trap-is-false-done, old-row refusal) under the "
          "daily gate")


# ---------------------------------------------------------------------------
# daily-maintenance round (2026-11-08, v5.258): fold five more
# exempt-backlog sidecars into the daily gate — the oracle-answer guard
# (answer-key isolation), the tri-mode think parser + experience store,
# the thinking-budget state machine, the three-modes report oracles and
# the skill-docs validator. All five are plain functions: no pytest
# import, no fixtures, no sockets, no CUDA.
# ---------------------------------------------------------------------------


def test_oracle_guard_suite():
    """Oracle answer-key guard under the daily gate. Pins: writes and
    answer-key reads are blocked, prefix-sibling escapes are blocked,
    an agent loop cannot self-grade, and an empty protected set fails
    loudly."""
    from helioslm_v5.tests import test_oracle_guard as m
    expected = {"test_guard_blocks_write_and_preserves_oracle",
                "test_guard_blocks_answer_key_read",
                "test_prefix_sibling_escape_blocked",
                "test_agent_loop_self_grading_attempt",
                "test_empty_protected_set_is_loud"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_oracle_guard_suite",
          "oracle guard: all 5 sidecar tests under the daily gate")


def test_think_suite():
    """Tri-mode think parsing and experience store under the daily gate.
    Pins: parse tri-mode, the think session flow, experience recall and
    cross-turn reuse, and replay compatibility."""
    from helioslm_v5.tests import test_think as m
    expected = {"test_parse_tri_mode", "test_think_session_flow",
                "test_experience_store_recall",
                "test_experience_reuse_across_turns",
                "test_replay_compatible"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_think_suite",
          "think/experience store: all 5 sidecar tests under the daily "
          "gate")


def test_thinking_budget_suite():
    """Thinking-budget control under the daily gate. Pins: default
    behaviour unchanged, the budget oracle, the state machine,
    truncation helpers and the A/B artifact."""
    from helioslm_v5.tests import test_thinking_budget as m
    expected = {"test_default_unchanged", "test_budget_oracle",
                "test_state_machine", "test_truncation_helpers",
                "test_ab_artifact"}
    _drift_guard(m, expected)
    for name in sorted(expected):
        getattr(m, name)()
    _pass("test_thinking_budget_suite",
          "thinking budget: all 5 sidecar tests under the daily gate")


def test_three_modes_suite():
    """Three-modes report oracles (T19) under the daily gate: one plain
    function, folded as-is."""
    from helioslm_v5.tests import test_three_modes as m
    expected = {"test_t19_report_oracles"}
    _drift_guard(m, expected)
    m.test_t19_report_oracles()
    _pass("test_three_modes_suite",
          "T19 three-modes report oracles under the daily gate")


def test_skill_docs_suite():
    """Skill-docs validator under the daily gate: one plain function,
    folded as-is."""
    from helioslm_v5.tests import test_skill_docs as m
    expected = {"test_skill_docs_validate"}
    _drift_guard(m, expected)
    m.test_skill_docs_validate()
    _pass("test_skill_docs_suite",
          "skill docs validator under the daily gate")


# ---------------------------------------------------------------------------
# limitations round (2026-11-08): fold test_package_tiers (T142/T166
# packaging) and test_prereg (Plan E5 pre-registered evaluation) into the
# daily gate. Both files gained the in-file pytest shim. test_prereg's
# three parametrize lists were hoisted to module level
# (_VALIDATION_MUTATIONS / _REL_GATE_MUTATIONS / _COMMITTED_PLANS) so the
# wrapper can iterate the same rows pytest would collect; the shim adds
# approx (default rel=1e-6 / abs=1e-12) for the paired-bootstrap pin.
# test_package_tiers' tokenizer test self-skips under the gate because
# the optional 'tokenizers' package is absent from the managed runtime —
# the wrapper checks importlib.util.find_spec, mirroring pytest's
# importorskip on CI; the shim's importorskip fails loudly if reached.
# tmp_path gets a real TemporaryDirectory; monkeypatch (test_fit_flag)
# gets the _MonkeyPatch stand-in with undo in finally.
# ---------------------------------------------------------------------------


def test_package_tiers_suite():
    """T142/T166 tier packaging (src/export/package.py + examples/
    package_tiers.py / package_hf.py) under the daily gate. 16 of 17
    sidecar tests run unconditionally (lite-model GGUF exports on CPU,
    refusals, CLI subprocesses, imatrix / speculative-draft / KV-cache /
    MoE-offload / reasoning-off / Q3_K pins); the tokenizer-embedding
    test self-skips when the optional 'tokenizers' package is absent
    (same skip as pytest's importorskip on CI)."""
    import importlib.util
    from helioslm_v5.tests import test_package_tiers as m
    expected = {"test_tiers_follow_the_readme_table",
                "test_package_without_tokenizer",
                "test_package_with_tokenizer_writes_modelfile",
                "test_fit_flag", "test_kimi_k3_package", "test_refusals",
                "test_cli_packages_a_pilot_run", "test_unified_memory_tiers",
                "test_package_gguf_external_files",
                "test_package_gguf_quant_override",
                "test_package_hf_imatrix", "test_speculative_draft",
                "test_kv_cache_options", "test_moe_expert_offload_note",
                "test_unified_memory_laptops_on_tiers",
                "test_package_gguf_reasoning_off", "test_q3k_edge_package"}
    _drift_guard(m, expected)
    for name in ("test_tiers_follow_the_readme_table",
                 "test_unified_memory_tiers",
                 "test_unified_memory_laptops_on_tiers"):
        getattr(m, name)()
    for name in sorted(expected - {"test_tiers_follow_the_readme_table",
                                   "test_unified_memory_tiers",
                                   "test_unified_memory_laptops_on_tiers",
                                   "test_fit_flag"}):
        if name == "test_package_with_tokenizer_writes_modelfile" and \
                importlib.util.find_spec("tokenizers") is None:
            print("SKIP test_package_with_tokenizer_writes_modelfile: "
                  "'tokenizers' not installed (importorskip semantics)")
            continue
        with tempfile.TemporaryDirectory() as t:
            getattr(m, name)(Path(t))
    mp = _MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as t:
            m.test_fit_flag(Path(t), mp)
    finally:
        mp.undo()
    _pass("test_package_tiers_suite",
          "tier packaging: 16/17 sidecar tests under the daily gate "
          "(tokenizer-embedding test self-skips without the optional "
          "'tokenizers' package, as importorskip does on CI)")


def test_prereg_suite():
    """Plan E5 pre-registered evaluation (eval/prereg.py) under the
    daily gate. All 9 sidecar tests; the parametrized ones iterate the
    hoisted module-level lists (13 plan mutations, 4 relative-gate
    mutations, 2 committed plans) so the gate runs every row pytest
    would collect. Pins: canonical lock hash, validation refusals, gate
    verdicts, plan-citation and completeness refusals, relative gates
    (CI above margin), PENDING revisions, paired bootstrap, and the
    committed v1.0 / v2.0 plans."""
    from helioslm_v5.tests import test_prereg as m
    expected = {"test_lock_is_canonical_and_edit_sensitive",
                "test_validation_rejects", "test_gate_verdicts",
                "test_results_must_cite_this_plan_and_be_complete",
                "test_relative_gate_validation",
                "test_relative_gate_verdicts",
                "test_pending_revisions_lock_but_do_not_check",
                "test_paired_bootstrap", "test_committed_plans"}
    _drift_guard(m, expected)
    m.test_lock_is_canonical_and_edit_sensitive()
    for mutate in m._VALIDATION_MUTATIONS:
        m.test_validation_rejects(mutate)
    m.test_gate_verdicts()
    m.test_results_must_cite_this_plan_and_be_complete()
    for mutate in m._REL_GATE_MUTATIONS:
        m.test_relative_gate_validation(mutate)
    m.test_relative_gate_verdicts()
    m.test_pending_revisions_lock_but_do_not_check()
    m.test_paired_bootstrap()
    for release, ref, gated_tasks in m._COMMITTED_PLANS:
        m.test_committed_plans(release, ref, gated_tasks)
    _pass("test_prereg_suite",
          "prereg plans: all 9 sidecar tests (13+4 mutation rows, 2 "
          "committed plans, paired bootstrap) under the daily gate")


def test_sidecar_coverage_guard():
    """Every helioslm_v5/tests/test_*.py sidecar is accounted for: either
    explicitly EXEMPT (pre-guard legacy backlog, 2026-11-02 snapshot).
    Loud both ways: a NEW sidecar upstream lands must be folded or
    exempted the same day; a deleted/renamed file must be dropped from
    the registry. Also pins that every FOLDED entry really names a
    wrapper defined in this module."""
    tests_dir = Path(__file__).resolve().parent
    on_disk = {p.stem for p in tests_dir.glob("test_*.py")}
    on_disk.discard("test_v5")  # the gate itself, not a sidecar
    folded = set(FOLDED_SIDECARS)
    exempt = set(EXEMPT_SIDECARS)
    overlap = folded & exempt
    assert not overlap, f"listed as both folded and exempt: {sorted(overlap)}"
    import helioslm_v5.tests.case_upstream_suites as self_mod
    missing_wrappers = [w for w in FOLDED_SIDECARS.values()
                        if not callable(getattr(self_mod, w, None))]
    assert not missing_wrappers, \
        f"FOLDED entries without a wrapper function: {missing_wrappers}"
    new = sorted(on_disk - folded - exempt)
    gone = sorted((folded | exempt) - on_disk)
    assert not new and not gone, (
        f"sidecar coverage drift: new={new} (fold into a wrapper above "
        f"or add to EXEMPT_SIDECARS with a reason) gone={gone} (drop "
        f"from the registry)")
    _pass("test_sidecar_coverage_guard",
          f"sidecar coverage locked: {len(folded)} folded + "
          f"{len(exempt)} exempt legacy = {len(on_disk)} files on disk")
