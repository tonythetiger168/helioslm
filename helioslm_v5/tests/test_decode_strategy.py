"""Decode-strategy layer tests (roadmap 2026Q4-v2, improvement 5).

Oracle for the MTP / thinking-budget layering refactor: the mutual
exclusion between ``use_mtp`` (decode machinery selection) and
``thinking_budget`` (a per-row inference strategy) is owned by
``helioslm_v5.src.inference.decode_strategy``, not by the model. Before
the refactor this file fails at import time (the strategy layer does not
exist); after it, these tests pin:

1. the rule fires at strategy construction, with a message that names
   both knobs and the fix;
2. ``HeliosLMv5.generate`` (the compatibility facade) raises that same
   strategy-layer ``ValueError`` at the same boundary;
3. the default strategy is the plain path — bit-identical to calling
   ``generate`` with no strategy kwargs, greedy and sampled;
4. a valid thinking-budget strategy still reaches the plain path (the
   forced close fires after exactly ``budget`` thinking tokens).

Run from repo root: python3 helioslm_v5/tests/test_decode_strategy.py
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.inference.decode_strategy import (  # noqa: E402
    STRATEGY_CONFLICT_MSG, DecodeStrategy, validate_decode_strategy)
from helioslm_v5.src.inference.thinking_budget import ThinkingBudget  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

OPEN, CLOSE = 7, 8


def _model():
    torch.manual_seed(0)
    return HeliosLMv5(HeliosLMv5Config(size="lite")).eval()


def test_strategy_layer_owns_the_rule():
    tb = ThinkingBudget(OPEN, CLOSE, 1)
    # valid combinations construct fine
    DecodeStrategy()
    DecodeStrategy(thinking_budget=tb)
    DecodeStrategy(use_mtp=True)
    # the conflict is refused at strategy construction, in this layer
    for build in (lambda: DecodeStrategy(use_mtp=True, thinking_budget=tb),
                  lambda: validate_decode_strategy(True, tb)):
        try:
            build()
        except ValueError as e:
            msg = str(e)
            # the message must name both knobs and the fix
            assert "use_mtp" in msg, msg
            assert "thinking_budget" in msg, msg
            assert "plain" in msg, msg
        else:
            raise AssertionError("expected ValueError")
    print("PASS test_strategy_layer_owns_the_rule")


def test_generate_facade_delegates():
    m = _model()
    tb = ThinkingBudget(OPEN, CLOSE, 1)
    try:
        m.generate(torch.ones(1, 2, dtype=torch.long),
                   use_mtp=True, thinking_budget=tb)
    except ValueError as e:
        assert str(e) == STRATEGY_CONFLICT_MSG, str(e)
    else:
        raise AssertionError("expected ValueError")
    print("PASS test_generate_facade_delegates")


def test_default_strategy_bit_identical():
    m = _model()
    ids = torch.randint(10, 900, (2, 6))
    for temp in (0, 0.8):
        torch.manual_seed(1)
        ref = m.generate(ids, max_new_tokens=8, temperature=temp)
        torch.manual_seed(1)
        via_kwargs = m.generate(ids, max_new_tokens=8, temperature=temp,
                                use_mtp=False, thinking_budget=None)
        torch.manual_seed(1)
        s = DecodeStrategy()
        via_strategy = m.generate(ids, max_new_tokens=8, temperature=temp,
                                  use_mtp=s.use_mtp,
                                  thinking_budget=s.thinking_budget)
        assert torch.equal(ref, via_kwargs)
        assert torch.equal(ref, via_strategy)
    print("PASS test_default_strategy_bit_identical")


def test_budget_strategy_still_enforced():
    m = _model()
    # the prompt row sits inside a thinking span (OPEN present, no CLOSE)
    ids = torch.tensor([[5, OPEN, 11, 12]])
    s = DecodeStrategy(thinking_budget=ThinkingBudget(OPEN, CLOSE, 2))
    out = m.generate(ids, max_new_tokens=5, temperature=0,
                     use_mtp=s.use_mtp, thinking_budget=s.thinking_budget)
    new = out[0, ids.shape[1]:].tolist()
    # after exactly 2 thinking tokens the next token is forced to CLOSE
    assert new[2] == CLOSE, new
    print("PASS test_budget_strategy_still_enforced")


if __name__ == "__main__":
    test_strategy_layer_owns_the_rule()
    test_generate_facade_delegates()
    test_default_strategy_bit_identical()
    test_budget_strategy_still_enforced()
