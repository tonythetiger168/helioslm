"""Regression: PrefixPool block keys must chain to the parent prefix.

Old _key() hashed (fingerprint + the block's OWN token ids) only, so two
prompts sharing a middle block collided: the second store() overwrote the
first prompt's entry, and a later lookup served KV computed under the
WRONG prefix (reproduced pre-fix: matched=8 tokens with KV max-diff 3.78,
next-token argmax 217 vs true 647 — a silent violation of the
"pooled == from-scratch, bitwise" contract that HarnessEvolver's gate
relies on).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.inference.prefix_pool import PrefixPool
from helioslm_v5.src.model_v5 import HeliosLMv5


def _lite_model(seed):
    torch.manual_seed(seed)
    return HeliosLMv5(HeliosLMv5Config(size="lite")).eval()


def test_prefix_pool_chained_keys():
    model = _lite_model(seed=123)
    pool = PrefixPool(model, block_size=4, max_blocks=32,
                      fingerprint="v5.20/fp32")
    A = [1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8]      # blocks A0 X A2
    C = [4, 3, 2, 1, 9, 9, 9, 9, 7, 7, 7, 7]      # blocks C0 X C2
    Q = [1, 2, 3, 4, 9, 9, 9, 9]                  # A0 X — ends on the
                                                  # once-shared block
    pool.store(A)
    pool.store(C)   # pre-fix this overwrote A's middle-block entry

    # with chaining, A's X-block and C's X-block are distinct keys
    assert len(pool._pool) == 6, \
        f"shared middle block must not collide: {len(pool._pool)} keys"

    hit = pool.lookup(Q)
    assert hit is not None and hit[0] == 8, \
        "A's exact prefix must still hit after C was stored"

    ref = model.generate(torch.tensor([Q]), max_new_tokens=8,
                         temperature=0)
    out = pool.generate(Q, max_new_tokens=8, temperature=0)
    assert torch.equal(out, ref), \
        "pooled generate after cross-prompt block reuse must be bit-exact"


if __name__ == "__main__":
    test_prefix_pool_chained_keys()
    print("PASS test_prefix_pool_chained_keys")
