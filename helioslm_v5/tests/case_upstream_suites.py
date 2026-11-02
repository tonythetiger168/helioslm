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
          f"v5.214 grounded report: all 10 sidecar tests (incl. "
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
}

# Pre-guard legacy backlog (2026-11-02 snapshot), NOT yet triaged for
# folding. Being listed here is not an endorsement that a file should
# stay out of the gate; it only means "known, pending triage".
EXEMPT_SIDECARS = {
    'test_abstain', 'test_agent', 'test_agent_workflow', 'test_api_distill',
    'test_artifact_tiers', 'test_async_grpo', 'test_attn_res', 'test_base_gap',
    'test_bench_quant', 'test_benchmark', 'test_bpe', 'test_calibrated_prefetcher',
    'test_calibrated_router', 'test_calibration_board', 'test_chat', 'test_chat_tuned',
    'test_chunked_ce', 'test_clamped_swiglu', 'test_cloudflare_bundles', 'test_copy_curriculum',
    'test_cost_per_task', 'test_cpt_chunked_ce', 'test_cpt_pilot', 'test_cpt_plan',
    'test_csa', 'test_csa_learned', 'test_csa_learned_16x', 'test_csa_learned_h64',
    'test_csa_learned_r3', 'test_cuda_device_paths', 'test_cuda_smoke', 'test_custom_attention_notebook',
    'test_cvdp_bench', 'test_data_guard', 'test_data_quality', 'test_decision',
    'test_decision_data', 'test_decision_head', 'test_decision_head_integration', 'test_decontam',
    'test_device_matrix', 'test_disagg', 'test_disagg_pareto', 'test_distill_agent',
    'test_distill_filter', 'test_distill_gen', 'test_distill_sft', 'test_dpo',
    'test_dsa', 'test_envs', 'test_external_model_gate', 'test_file_env',
    'test_finetune_data', 'test_first_k_dense', 'test_gated_conv', 'test_gguf_quant',
    'test_global_prope_kv', 'test_gpu_batch_run', 'test_grounded_replay', 'test_grounding',
    'test_grpo_length', 'test_grpo_u1', 'test_harness', 'test_harness_deep',
    'test_hca_rescue', 'test_hdl_bench', 'test_hdl_data', 'test_hdl_git_sources',
    'test_hdl_projects', 'test_hdl_reward', 'test_hdl_train_eval', 'test_hf_license_audit',
    'test_hsh_clinical', 'test_hsh_egress', 'test_hsh_extras', 'test_hsh_interaction',
    'test_hsh_kernel', 'test_hsh_llm', 'test_hsh_loop', 'test_hsh_mcp',
    'test_hsh_presets', 'test_hsh_session', 'test_hsh_skills', 'test_hsh_tools',
    'test_hsh_web', 'test_hub', 'test_indexer_share', 'test_k3_block',
    'test_k3_llamacpp_parts', 'test_kda_chunked', 'test_kda_k3_style', 'test_kda_subchunk',
    'test_kv_compress', 'test_kv_share', 'test_llamacpp_export', 'test_llamacpp_kimi_k3',
    'test_llamacpp_tokenizer', 'test_lm_decision_head', 'test_lm_harness', 'test_logit_kd',
    'test_long_horizon_bench', 'test_longctx', 'test_longctx_harness', 'test_lora',
    'test_merge', 'test_mhc', 'test_mhc_sinkhorn', 'test_midtrain',
    'test_mimo_hybrid', 'test_mini_bench', 'test_mla_every', 'test_mla_expanded_training',
    'test_moe_config_table', 'test_mopd', 'test_mtp_light', 'test_mtp_rollout',
    'test_multiseed', 'test_nope_temperature', 'test_noul_continuous', 'test_nvfp4_qat_e2e',
    'test_option_matrix', 'test_oracle_guard', 'test_package_tiers', 'test_ple',
    'test_pool_ledger', 'test_prefix_pool_chained', 'test_prereg', 'test_pretrain_pilot',
    'test_probe_api', 'test_quant_calib', 'test_quantile_balancing', 'test_qwen_next',
    'test_repo_longctx', 'test_review_fixes', 'test_review_regressions', 'test_rlcd_head',
    'test_rtllm_bench', 'test_ruler_lite', 'test_selfgen_meta', 'test_skill_docs',
    'test_smebu_sandwich', 'test_std_bench', 'test_stop_calibration', 'test_swarm_parallel',
    'test_swe_agent_rl', 'test_swe_bench', 'test_task_grammar', 'test_teacher_license_audit',
    'test_term_bench', 'test_think', 'test_thinking_budget', 'test_three_modes',
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


def test_sidecar_coverage_guard():
    """Every helioslm_v5/tests/test_*.py sidecar is accounted for: either
    FOLDED into the daily gate through a wrapper in this module, or
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
