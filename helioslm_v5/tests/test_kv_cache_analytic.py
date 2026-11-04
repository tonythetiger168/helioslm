"""Lock the published full-config KV-cache row at sequence length 131072.

The README quotes the analytic table from
``python benchmarks/bench_cpu.py --analytic-only --size full``: at
131072 tokens the ``full`` config holds 1999.6 MB of cache against an
MHA reference of 257698.0 MB, 99.2% smaller. Those three figures are
the one-decimal print of ``benchmarks.bench_cpu.cache_bytes``; this
test calls that function (it does not restate the byte formula) so a
config or cache-layout change fails here instead of leaving the
published row silently stale. No wall-clock timing.

Plain function, no pytest fixtures (folded into the daily gate).
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.bench_cpu import cache_bytes  # noqa: E402
from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402

# Published one-decimal row (bench_cpu's ``.1f`` print). The underlying
# counts are exact structural bytes, so a half-tenth window is the
# published figure: anything outside it prints a different tenth.
_SEQ_LEN = 131072
_HELIOS_MB = 1999.6
_MHA_MB = 257698.0
_SAVINGS_PCT = 99.2
_TOL = 0.05


def test_full_kv_cache_savings_at_128k():
    row = cache_bytes(HeliosLMv5Config(size="full"), _SEQ_LEN)
    helios_mb = row["helioslm_bytes"] / 1e6
    mha_mb = row["mha_reference_bytes"] / 1e6
    savings_pct = row["savings_vs_mha"] * 100.0
    assert abs(helios_mb - _HELIOS_MB) < _TOL, helios_mb
    assert abs(mha_mb - _MHA_MB) < _TOL, mha_mb
    assert abs(savings_pct - _SAVINGS_PCT) < _TOL, savings_pct
