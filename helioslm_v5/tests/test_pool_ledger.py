"""v5.42 — prefix-pool ledger oracles (Phase 3.1).

Run from repo root: python helioslm_v5/tests/test_pool_ledger.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.inference.pool_ledger import PoolLedger
from helioslm_v5.src.inference.prefix_pool import PrefixPool
from helioslm_v5.src.model_v5 import HeliosLMv5


def _setup():
    torch.manual_seed(0)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    pool = PrefixPool(model, block_size=4, max_blocks=64,
                      fingerprint="ledger-test")
    return PoolLedger(pool)


def test_ledger_counts_reuse():
    led = _setup()
    shared = [7, 8, 9, 10, 11, 12, 13, 14]
    led.generate(shared, max_new_tokens=2)                 # cold: miss
    led.generate(shared, max_new_tokens=2)                 # warm: full hit
    rep = led.report()
    assert rep["requests"] == 2
    assert rep["entries"][0]["reused_tokens"] == 0
    assert rep["entries"][1]["reused_tokens"] == len(shared), rep["entries"]
    assert rep["tokens_computed"] == len(shared)
    assert rep["tokens_reused"] == len(shared)
    # discount = reused/(reused+computed) = 8/16 = 0.5
    assert abs(rep["effective_discount"] - 0.5) < 1e-9, rep
    print(f"PASS test_ledger_counts_reuse discount="
          f"{rep['effective_discount']:.3f} over {rep['requests']} requests")


def test_ledger_partial_prefix():
    led = _setup()
    led.generate([7, 8, 9, 10, 11, 12, 13, 14], max_new_tokens=2)
    led.generate([7, 8, 9, 10, 99, 98], max_new_tokens=2)   # half shared
    e = led.report()["entries"][1]
    assert e["reused_tokens"] == 4, e
    assert e["computed_tokens"] == 2, e
    print(f"PASS test_ledger_partial_prefix reused={e['reused_tokens']} "
          f"computed={e['computed_tokens']}")


def test_report_scopes_and_loud_empty():
    led = _setup()
    rep = led.report()
    assert rep["effective_discount"] == 0.0     # no requests: no claim
    assert "prompt tokens only" in rep["scope"]
    assert rep["entries"] == []
    print("PASS test_report_scopes_and_loud_empty empty ledger -> "
          "discount 0.0, scope recorded")


if __name__ == "__main__":
    test_ledger_counts_reuse()
    test_ledger_partial_prefix()
    test_report_scopes_and_loud_empty()
    print("\n3/3 pool ledger tests passed")
