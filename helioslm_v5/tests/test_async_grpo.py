"""T26 - v5.31: asynchronous GRPO oracles.

Core guarantee: decoupling generation from learning must NOT change the
mathematics. Single worker + single question reproduces the synchronous
train_step loss bitwise; multi-question runs process every group exactly
once in FIFO order.
Run from repo root: python3 helioslm_v5/tests/test_async_grpo.py
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.training.async_grpo import AsyncGRPO
from helioslm_v5.src.training.grpo import GRPOTrainer


def _mk(pair_seed=5):
    torch.manual_seed(pair_seed)
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    ref = copy.deepcopy(model)
    cfg_g = SimpleNamespace(group_size=4, epsilon=0.2, kl_coef=0.01,
                            max_new_tokens=16, lr=1e-4)
    config = SimpleNamespace(grpo=cfg_g)
    return model, ref, config


def test_single_group_parity():
    """1 worker, 1 question: async update == sync train_step bitwise.
    Uses the trainer's built-in reward (accuracy+format+length) — both
    paths must agree because samples and answers are identical."""
    torch.manual_seed(11)
    m1, r1, c1 = _mk()
    t1 = GRPOTrainer(m1, r1, c1, tokenizer=None)
    torch.manual_seed(123)
    sync_metrics = t1.train_step(["What is 2+2?"], ["4"])

    torch.manual_seed(11)
    m2, r2, c2 = _mk()
    t2 = GRPOTrainer(m2, r2, c2, tokenizer=None)
    torch.manual_seed(123)
    pipe = AsyncGRPO(t2, n_workers=1, seed=123)
    pipe.run(["What is 2+2?"], ["4"])

    assert len(pipe.metrics) == 1
    for k in ("loss", "policy_loss", "kl_penalty", "mean_reward"):
        assert abs(pipe.metrics[0][k] - sync_metrics[k]) < 1e-9, \
            f"{k}: async {pipe.metrics[0][k]} != sync {sync_metrics[k]}"
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "parameter drift after async update"
    print(f"PASS test_single_group_parity (loss={sync_metrics['loss']:.6f})")


def test_fifo_all_groups_once():
    torch.manual_seed(13)
    m, r, c = _mk()
    t = GRPOTrainer(m, r, c, tokenizer=None)
    questions = [f"Question {i}: compute {i}*2" for i in range(6)]
    answers = [str(i * 2) for i in range(6)]
    pipe = AsyncGRPO(t, n_workers=2, seed=7)
    pipe.run(questions, answers)
    assert len(pipe.metrics) == len(questions), \
        f"expected {len(questions)} updates, got {len(pipe.metrics)}"
    # no residual work
    assert pipe.q.empty()
    print(f"PASS test_fifo_all_groups_once "
          f"({len(pipe.metrics)} groups, 2 workers)")


def test_backpressure_and_stop():
    """Queue bounded: a slow learner never accumulates unbounded groups;
    stop() lets workers exit cleanly between questions."""
    import threading as _t
    torch.manual_seed(17)
    m, r, c = _mk()
    t = GRPOTrainer(m, r, c, tokenizer=None)
    pipe = AsyncGRPO(t, n_workers=1, max_pending=2, seed=3)
    questions = [f"q{i}" for i in range(10)]
    answers = ["x"] * 10
    worker = _t.Thread(target=pipe.rollout_worker, args=(0, questions, answers),
                       daemon=True)
    worker.start()
    import time as _time
    _time.sleep(1.0)          # learner is "slow" (asleep)
    assert pipe.q.qsize() <= 2, f"back-pressure violated: {pipe.q.qsize()}"
    pipe.stop()               # cooperative shutdown
    worker.join(timeout=10)
    assert not worker.is_alive(), "worker did not stop"
    print(f"PASS test_backpressure_and_stop (queue<=2, clean stop)")


if __name__ == "__main__":
    test_single_group_parity()
    test_fifo_all_groups_once()
    test_backpressure_and_stop()
    print("\n3/3 async-GRPO tests passed")
