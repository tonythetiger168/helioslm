"""Review-plan B1 folds (2026-11-23): the last three review-plan
mechanisms whose oracles sat in the EXEMPT sidecar backlog.

- T76 IndexShare (B1-1): GLM-5.2/5.3-style shared lightning indexer,
  one learned indexer per ``indexer_share_every`` MLA layers.
- T79 --mla-every sweep (B1-5): hybrid interleave rule swept over
  every in {2, 4, 8} via examples/mla_every_sweep.py, gated by the
  HarnessEvolver-style bitwise double-run.
- K3 block order (v5.57, first_layer_full=False): the hybrid-layout
  counterpart pins.

Bookkeeping note: the wrappers live here, not in
case_upstream_suites.py, because that file is 160KB+ and the GitHub
API content path is transcription-risky at that size (two slip events
on record). The sidecars stay listed in EXEMPT_SIDECARS so the
coverage guard is unaffected; moving the registry entries to
FOLDED_SIDECARS is left to the next direct-edit (non-API) maintenance
round. Their tests run daily here regardless.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from helioslm_v5.tests.case_common import _pass  # noqa: E402
from helioslm_v5.tests.case_upstream_suites import _drift_guard  # noqa: E402


def test_indexer_share_suite():
    """T76 IndexShare (B1-1): all six sidecar tests, plain functions."""
    from helioslm_v5.tests import test_indexer_share as m
    expected = {"test_config_validation",
                "test_default_per_layer_unchanged",
                "test_share_groups_and_param_savings",
                "test_shared_sparse_decode_deterministic",
                "test_shared_distill_trains_leader",
                "test_shared_checkpoint_roundtrip"}
    _drift_guard(m, expected)
    m.test_config_validation()
    m.test_default_per_layer_unchanged()
    m.test_share_groups_and_param_savings()
    m.test_shared_sparse_decode_deterministic()
    m.test_shared_distill_trains_leader()
    m.test_shared_checkpoint_roundtrip()
    _pass("test_indexer_share_suite",
          "T76 IndexShare: 6/6 — share=1 regression locked, share=2 "
          "saves exactly 2 indexers, distill trains the leader, "
          "checkpoint roundtrip bit-identical")


def test_mla_every_suite():
    """T79 --mla-every sweep (B1-5): layer map, plumbing, bitwise gate,
    CLI smoke. steps=1 winner is init-noise by construction — the smoke
    checks plumbing only."""
    from helioslm_v5.tests import test_mla_every as m
    expected = {"test_layer_map_rule", "test_config_validation",
                "test_bitwise_gate_double_run", "test_sweep_cli_smoke"}
    _drift_guard(m, expected)
    m.test_layer_map_rule()
    m.test_config_validation()
    m.test_bitwise_gate_double_run()
    m.test_sweep_cli_smoke()
    _pass("test_mla_every_suite",
          "T79 mla-every: 4/4 — layer map {2,4,8} locked, bitwise gate "
          "non-vacuous, sweep CLI end-to-end with gate-passing arms")


def test_k3_block_suite():
    """K3 block order: defaults untouched, gate math bitwise, ungated
    checkpoint fails loudly."""
    from helioslm_v5.tests import test_k3_block as m
    expected = {"test_defaults_untouched", "test_k3_order_and_validation",
                "test_gate_math_bitwise", "test_gate_paths_agree",
                "test_k3_preset", "test_ungated_checkpoint_fails_loudly"}
    _drift_guard(m, expected)
    m.test_defaults_untouched()
    m.test_k3_order_and_validation()
    m.test_gate_math_bitwise()
    m.test_gate_paths_agree()
    m.test_k3_preset()
    m.test_ungated_checkpoint_fails_loudly()
    _pass("test_k3_block_suite",
          "K3 block order: 6/6 — defaults untouched, gate math bitwise, "
          "ungated checkpoint fails loudly")


if __name__ == "__main__":
    test_indexer_share_suite()
    test_mla_every_suite()
    test_k3_block_suite()
