"""OS3: abstention-aware rewards (src/training/abstain.py).

Pins:
1. abstention detection: explicit ANSWER: UNKNOWN, English and Chinese
   phrases; hedging that still states a number is an answer;
2. payoffs: answerable correct +1 / abstain 0 / wrong -penalty;
   unanswerable abstain +1 / answer -penalty; penalty 0 = plain accuracy;
3. the decision rule: answering beats abstaining in expectation exactly
   above answer_threshold(penalty);
4. AbstainMathEnv builds unanswerable tasks (operand withheld) at the
   requested rate, deterministically per seed, and routes through
   MultiEnvBatch;
5. abstain_metrics splits accuracy / over-caution / hallucination;
6. GRPOTrainer: abstain_penalty=None keeps the old rewards; set, it uses
   the abstention payoffs and accepts None answers.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_abstain.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    GRPOConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training.abstain import (  # noqa: E402
    UNANSWERABLE, AbstainMathEnv, abstain_metrics, abstain_reward,
    answer_threshold, is_abstention)
from helioslm_v5.src.training.envs import MultiEnvBatch  # noqa: E402
from helioslm_v5.src.training.grpo import GRPOTrainer  # noqa: E402

eq = lambda r, t: r.strip().split()[-1] == t  # noqa: E731


def test_detection():
    for r in ("ANSWER: UNKNOWN", "I don't know.", "This cannot be determined.",
              "Not enough information to say.", "<think>hmm 12?</think>我不知道",
              "資訊不足，無法回答"):
        assert is_abstention(r), r
    for r in ("42", "I don't know, maybe 42", "無法確定，大概是 7", "It is unknown"):
        assert not is_abstention(r), r


def test_payoffs():
    assert abstain_reward("7", "7", eq, 2.0) == 1.0
    assert abstain_reward("8", "7", eq, 2.0) == -2.0
    assert abstain_reward("ANSWER: UNKNOWN", "7", eq, 2.0) == 0.0
    assert abstain_reward("ANSWER: UNKNOWN", UNANSWERABLE, eq, 2.0) == 1.0
    assert abstain_reward("7", UNANSWERABLE, eq, 2.0) == -2.0
    assert abstain_reward("8", "7", eq, 0.0) == 0.0  # plain accuracy
    with pytest.raises(ValueError):
        abstain_reward("8", "7", eq, -1)


def test_threshold_is_the_break_even_confidence():
    for pen in (0.0, 0.5, 1.0, 3.0):
        t = answer_threshold(pen)
        ev = lambda p: p * 1.0 - (1 - p) * pen  # noqa: E731
        assert abs(ev(t)) < 1e-12           # break even vs abstain (0)
        assert ev(min(t + 0.01, 1.0)) > 0 or t == 1.0
        assert ev(max(t - 0.01, 0.0)) < 0 or t == 0.0
    assert answer_threshold(1.0) == 0.5


def test_env_and_batch():
    env = AbstainMathEnv(p_unanswerable=0.5)
    tasks = [env.task(seed=s) for s in range(400)]
    una = [t for t in tasks if t["target"] is UNANSWERABLE]
    assert 150 < len(una) < 250
    assert all("ANSWER: UNKNOWN" in t["prompt"] for t in tasks)
    assert all(not any(c.isdigit() for c in t["prompt"].split("+")[-1].split("-")[-1]
                       .split("*")[-1].split("?")[0]) for t in una)
    assert [env.task(seed=3) for _ in range(2)] == [env.task(seed=3)] * 2
    t = next(t for t in tasks if t["target"] is not UNANSWERABLE)
    assert env.reward(t["target"], t["target"]) == 1.0
    assert env.reward("ANSWER: UNKNOWN", una[0]["target"]) == 1.0
    assert env.reward("5", una[0]["target"]) == -1.0
    b = MultiEnvBatch({"math-abstain": 4, "math": 2},
                      envs={"math-abstain": AbstainMathEnv})
    bt = b.tasks(seed=0)
    assert len(bt) == 6
    assert b.rewards(bt, ["ANSWER: UNKNOWN"] * 6).count(0.0) >= 2  # math env


def test_metrics():
    tasks = [{"target": "1"}, {"target": "2"}, {"target": "3"}, {"target": "4"},
             {"target": None}, {"target": None}]
    resp = ["1", "9", "ANSWER: UNKNOWN", "4", "5", "I don't know"]
    m = abstain_metrics(tasks, resp, eq)
    assert m == {"n_answerable": 4, "n_unanswerable": 2, "accuracy": 0.5,
                 "abstain_rate": 0.25, "wrong_rate": 0.25,
                 "precision": 2 / 3, "hallucination_rate": 0.5}


def _trainer(**grpo):
    cfg = HeliosLMv5Config(
        size="lite", num_hidden_layers=1, vocab_size=16, hidden_size=64,
        intermediate_size=128,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=2, num_activated_experts=1, expert_hidden_size=64),
        use_attention_residuals=False, use_hyper_connections=False,
        grpo=GRPOConfig(group_size=2, epsilon=0.2, kl_coef=0.0, lr=1e-2, **grpo))
    torch.manual_seed(0)
    m = HeliosLMv5(cfg)
    return GRPOTrainer(m, copy.deepcopy(m), cfg)


def test_grpo_trainer_rewards():
    resp, ans = ["7", "8", "ANSWER: UNKNOWN"], ["7", "7", "7"]
    old = _trainer().compute_rewards(resp, ans)
    # before: wrong and abstain both get no accuracy credit; they differ
    # only by the length term (-0.1 * |len - 3| / 3)
    assert abs((old[1] + 0.1 * 2 / 3) - (old[2] + 0.1 * 12 / 3)) < 1e-6
    assert old[0] > old[1]
    new = _trainer(abstain_penalty=1.0).compute_rewards(resp + ["ANSWER: UNKNOWN", "3"],
                                                         ans + [None, None])
    assert new[0] > new[2] > new[1]               # correct > abstain > wrong
    assert new[3].item() == 1.0 and new[4].item() == -1.0
    with pytest.raises(ValueError):
        GRPOConfig(abstain_penalty=-0.5)
