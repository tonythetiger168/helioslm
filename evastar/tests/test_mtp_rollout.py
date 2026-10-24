"""OS5: MTP speculative rollouts in GRPO (grpo.rollout_use_mtp).

Pins:
1. default False: ``_sample_response`` calls ``generate`` exactly as
   before (no ``use_mtp`` argument);
2. True: rollouts go through ``generate(..., use_mtp=True)`` at
   temperature 1 / top_p 1, and the MTP path's output satisfies GRPO's
   full-sequence contract (prompt prefix + new tokens, finite log-prob);
3. True on a model without MTP modules is refused at construction, and
   the flag is validated as a bool.

Speed-up and the sampled-distribution gate are measured by
examples/mtp_rollout_bench.py (GPU run pending).

Run from repo root: python3 -m pytest evastar/tests/test_mtp_rollout.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.configs.config_v5 import (  # noqa: E402
    GRPOConfig, EvastarV5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.training.grpo import GRPOTrainer  # noqa: E402


def _cfg(mtp=True, **grpo):
    return EvastarV5Config(
        size="lite", num_hidden_layers=1, vocab_size=300, hidden_size=64,
        intermediate_size=128,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=mtp, num_modules=1),
        moe=MoEConfig(num_experts=2, num_activated_experts=1, expert_hidden_size=64),
        use_attention_residuals=False, use_hyper_connections=False,
        grpo=GRPOConfig(group_size=2, epsilon=0.2, kl_coef=0.0, lr=1e-2, **grpo))


def _trainer(mtp=True, **grpo):
    cfg = _cfg(mtp, **grpo)
    torch.manual_seed(0)
    m = EvastarV5(cfg)
    return GRPOTrainer(m, copy.deepcopy(m), cfg), m


def _spy(model):
    calls = []
    real = model.generate

    def gen(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    model.generate = gen
    return calls


@pytest.mark.parametrize("flag", [False, True])
def test_rollout_routing(flag):
    tr, m = _trainer(rollout_use_mtp=flag)
    calls = _spy(m)
    torch.manual_seed(1)
    text, old_lp, full, plen = tr._sample_response("hi")
    assert len(calls) == 1
    assert calls[0].get("use_mtp", False) is flag
    if not flag:
        assert "use_mtp" not in calls[0]
    assert calls[0]["temperature"] == 1.0 and calls[0]["top_p"] == 1.0
    assert full.shape[1] > plen and torch.isfinite(old_lp)


def test_refused_without_mtp_modules():
    cfg = _cfg(mtp=False, rollout_use_mtp=True)
    m = EvastarV5(cfg)
    with pytest.raises(ValueError, match="rollout_use_mtp"):
        GRPOTrainer(m, copy.deepcopy(m), cfg)
    with pytest.raises(ValueError):
        GRPOConfig(rollout_use_mtp="yes")
