"""HeliosLM v5.9 test suite (CPU, lite config).

Rewritten for v5.1 after the full code review and extended since (v5.2: true
GPTQ calibration, audio sliding window; v5.3: AWQ calibration repair;
v5.4: packed-position cross-document isolation, input-validation coverage;
v5.5: hybrid GatedDeltaAttention + state-cache rollback, LatentMoE,
quantile balancing, attention residuals, SiTU-GLU; v5.7: RoPE scaling,
FP8 KV cache, Hyper-Connections, QAT; v5.8: YaRN RoPE scaling, DSA sparse
top-k attention, per-head Muon, GPTQ act-order; v5.9: attention logit
soft-capping, per-head QK-norm, sliding-window attention with sinks,
final logit soft-capping; v5.21: per-channel (KDA-style) decay gate for
GatedDeltaAttention + MTP rollback compatibility; v5.22: NVFP4-format QAT
target + NoPE option for MLA; limitations round 2026-09-21: learned
lightning indexer for sparse top-k + opt-in sparse prefill):
every test exercises the real module with numerical assertions (not just
shapes), and any failure makes the process exit non-zero.

Usage (from the repository root, i.e. the directory that CONTAINS the
``helioslm_v5`` package):

    cd /mnt/agents/output
    python -m helioslm_v5.tests.test_v5

The project root is derived from this file's location, so direct execution
(``python helioslm_v5/tests/test_v5.py``) works as well. Exit code is 0 iff
all tests pass; failures print a FAIL line plus the full traceback and the
final exit code is 1.


Split layout (2026-10-11): the 90 test functions live in the
``helioslm_v5/tests/case_*.py`` modules (small, individually
reviewable/uploadable files) and share the helpers in
``case_common.py``; this runner imports them and executes the suite
in the historical order. Behavior and exit-code semantics are
unchanged. The 2026-10-12 merge with main's v5.48 limitations round
added case_harness_repair.py (harness plugin repair locks +
whole-tree import smoke) on top of the original 88.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helioslm_v5.tests import (  # noqa: E402
    case_audio_vision,
    case_dualpipe,
    case_engine,
    case_export_eval,
    case_harness_repair,
    case_hybrid,
    case_indexer,
    case_mla,
    case_model_mtp,
    case_moe,
    case_muon,
    case_quant,
    case_quant_fused,
    case_rope_cache,
    case_security,
    case_serving,
    case_sparse_attention,
    case_stability,
    case_sweeps,
    case_trainers,
    case_trust)
from helioslm_v5.tests.case_common import (  # noqa: E402
    RESULTS, _fail)

TESTS = [
    case_mla.test_mla,
    case_mla.test_mla_absorption,
    case_mla.test_mla_absorbed_left_padding,
    case_mla.test_packed_positions,
    case_moe.test_sigmoid_moe,
    case_model_mtp.test_v5_model,
    case_model_mtp.test_mtp,
    case_engine.test_paged_attention,
    case_engine.test_vllm_engine,
    case_dualpipe.test_dualpipe,
    case_trainers.test_fp8_trainer,
    case_trainers.test_grpo,
    case_trainers.test_env_grpo_loop,
    case_stability.test_topk_self_consistency,
    case_trust.test_quant_drift_trust_gate,
    case_stability.test_prefill_stability,
    case_trust.test_trust_calibration_from_report,
    case_trainers.test_async_env_grpo,
    case_trust.test_decision_head_noul_targets,
    case_trust.test_quant_calib_trust_gate_example,
    case_stability.test_indexer_stability_min_gate,
    case_sweeps.test_sparse_mtp_acceptance_sweep,
    case_sweeps.test_indexer_distill_learned_sweep,
    case_sweeps.test_engine_sparse_serving,
    case_muon.test_muon,
    case_audio_vision.test_streaming_audio,
    case_audio_vision.test_navit,
    case_audio_vision.test_navit_input_validation,
    case_quant.test_quantization,
    case_quant.test_gptq_calibration,
    case_quant.test_awq_calibration,
    case_quant.test_quant_weight_property,
    case_quant.test_quant_input_validation,
    case_quant.test_gptq_odd_dims,
    case_quant_fused.test_mxfp4,
    case_model_mtp.test_mtp_rebind_after_quantization,
    case_audio_vision.test_audio_sliding_window,
    case_quant_fused.test_nvfp4_qat,
    case_rope_cache.test_nope_attention,
    case_hybrid.test_per_channel_decay,
    case_model_mtp.test_mtp_per_channel_rollback,
    case_hybrid.test_linear_attention,
    case_hybrid.test_hybrid_model,
    case_moe.test_latent_moe,
    case_moe.test_quantile_balancing,
    case_hybrid.test_attention_residuals,
    case_moe.test_situ_glu,
    case_model_mtp.test_mtp_hybrid_rollback,
    case_moe.test_quantile_margin_boundary,
    case_moe.test_balance_state_resume,
    case_hybrid.test_hybrid_packed_positions,
    case_dualpipe.test_attention_residuals_dualpipe,
    case_hybrid.test_linear_attention_guards,
    case_rope_cache.test_rope_scaling,
    case_rope_cache.test_fp8_kv_cache,
    case_rope_cache.test_hyper_connections,
    case_quant_fused.test_qat,
    case_rope_cache.test_yarn_rope_scaling,
    # v5.49: YaRN factor-512 (2048 -> 2^20) position-encoding stress
    case_rope_cache.test_yarn_longctx_position_stress,
    case_sparse_attention.test_sparse_top_k_attention,
    case_indexer.test_learned_lightning_indexer,
    case_indexer.test_sparse_prefill_indexer,
    # limitations round 2026-10-13
    case_indexer.test_indexer_share_group_stress,
    case_muon.test_per_head_muon,
    case_quant.test_gptq_act_order,
    case_sparse_attention.test_attention_logit_soft_cap,
    case_sparse_attention.test_qk_norm,
    case_sparse_attention.test_sliding_window_attention,
    case_sparse_attention.test_final_logit_soft_cap,
    case_export_eval.test_gguf_export,
    case_trust.test_decision_layer_audit,
    case_serving.test_evolver_pareto_adaptation,
    case_serving.test_harness_evolver,
    case_serving.test_prefix_pool,
    case_serving.test_score_stream,
    case_serving.test_spec_breakeven,
    case_serving.test_expert_streaming,
    case_dualpipe.test_multi_process_dualpipe,
    case_serving.test_toy_checkpoint,
    case_engine.test_batched_prefill,
    case_quant_fused.test_fused_quant_kernels,
    case_export_eval.test_eval_harness,
    case_engine.test_vllm_engine_padded_batched_prefill,
    case_serving.test_prefix_pool_high_token_ids,
    case_dualpipe.test_multi_process_dualpipe_attention_residuals,
    case_trainers.test_fp8_trainer_cross_entropy,
    case_security.test_model_fingerprint,
    case_security.test_membership_inference_signal,
    case_security.test_distillation_leak_budget,
    case_security.test_prefix_pool_tenant_isolation,
    case_security.test_score_stream_leak_budget,
    # v5.48 merge (from main's 2026-10-10 limitations round): harness
    # plugin repair locks + whole-tree import smoke
    case_harness_repair.test_harness_memory_sqlite_fs_repair,
    case_harness_repair.test_import_smoke,
]

# CI thread guard (2026-10-13): PR #16's full-suite probe passes with
# threads pinned to 2; PR #15 preserves this setting and test order.
# The observed py3.11 failure was a distillation-loss assertion, not a
# confirmed resource exhaustion (see docs/PR16_CI_RESOLUTION.md).
# Pin on CI only; local runs keep machine defaults.
if os.environ.get("GITHUB_ACTIONS") == "true":
    import torch as _torch
    _torch.set_num_threads(2)


def main():
    print("HeliosLM v5.22 Test Suite (lite config, CPU)")
    print("=" * 72)
    for t in TESTS:
        try:
            t()
        except Exception:
            _fail(t.__name__)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("=" * 72)
    print(f"SUMMARY: {n_ok}/{len(RESULTS)} PASS")
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_ok == len(RESULTS) == len(TESTS) else 1)


if __name__ == "__main__":
    main()
