"""HeliosLM v5 test cases — FP8 trainer + GRPO / env-GRPO (sync + async).

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math

import torch

from helioslm_v5.tests.case_common import (_lite_model, _pass)


# ----------------------------------------------------------------------
# FP8Trainer: finite losses, no grad accumulation, real quantization
# ----------------------------------------------------------------------
def test_fp8_trainer():
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer, FP8Linear

    model, config = _lite_model(seed=110)
    trainer = FP8Trainer(model, config)

    # Quantization is real (not an identity pass-through).
    fp8_layer = next(m for m in model.modules() if isinstance(m, FP8Linear))
    w = fp8_layer.weight.detach()
    wq = fp8_layer._quantize_to_fp8(w, fp8_layer.weight_scale.clamp(min=1e-8))
    q_diff = (w - wq).abs().max().item()
    q_rel = ((w - wq).norm() / w.norm().clamp(min=1e-12)).item()
    assert q_diff > 0, "FP8 quantization is an identity — not real quantization"
    assert q_rel < 0.2, f"FP8 quantization error implausibly large: {q_rel:.3f}"

    ids = torch.randint(3, config.vocab_size, (2, 10))
    batch = {"input_ids": ids}
    probe = model.embed_tokens.weight  # not a Linear -> survives FP8 conversion
    losses, grad_norms = [], []
    for _ in range(3):
        losses.append(trainer.train_step(batch))
        grad_norms.append(probe.grad.norm().item())

    assert all(math.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    assert grad_norms[0] > 0, "no gradient reached embed_tokens"
    # Same batch each step: with per-step zero_grad the grad norm stays ~flat
    # (weights drift only slightly); accumulation would roughly triple it.
    ratio = grad_norms[-1] / max(grad_norms[0], 1e-12)
    assert ratio < 1.10, \
        f"grad norm ratio {ratio:.2f} over 3 identical steps suggests accumulation"
    _pass("test_fp8_trainer",
          f"losses={[f'{l:.4f}' for l in losses]}, "
          f"grad-norm ratio={ratio:.3f}, fp8 rel-err={q_rel:.4f}")


def test_fp8_trainer_cross_entropy():
    """v5.20: train_step(labels=...) must be real token cross-entropy
    (regression: _extract_loss averaged raw logits, so FP8 training never
    saw a task loss; mean-of-logits is unbounded below)."""
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer

    model, config = _lite_model(seed=120)
    trainer = FP8Trainer(model, config)
    ids = torch.randint(3, config.vocab_size, (2, 10))
    labels = ids.clone()  # next-token targets aligned with input_ids
    loss = trainer.train_step({"input_ids": ids}, labels=labels)

    # random init: CE ~ ln(vocab); the old fallback can even go negative
    ln_vocab = math.log(config.vocab_size)
    assert 0.05 < loss < 3.0 * ln_vocab, \
        f"loss {loss:.4f} is not a plausible token CE (ln vocab={ln_vocab:.3f})"
    probe = model.embed_tokens.weight  # not a Linear -> survives FP8 conversion
    assert probe.grad is not None and probe.grad.norm().item() > 0, \
        "no gradient reached embed_tokens"
    _pass("test_fp8_trainer_cross_entropy",
          f"labels-aware loss={loss:.4f} vs ln(vocab)={ln_vocab:.4f}")


# ----------------------------------------------------------------------
# GRPO: one real train_step; reward/answer alignment; non-negative KL
# ----------------------------------------------------------------------
def test_grpo():
    from helioslm_v5.src.training.grpo import GRPOTrainer

    model, config = _lite_model(seed=111)
    ref_model, _ = _lite_model(seed=211)  # different init -> non-trivial KL
    G = 3
    config.grpo.group_size = G
    config.grpo.max_new_tokens = 4
    trainer = GRPOTrainer(model, ref_model, config)

    # Spy reward function: records the answers it receives and returns
    # varied deterministic rewards (non-zero advantage variance per group).
    seen = {}

    def spy_reward(responses, answers):
        seen["answers"] = list(answers)
        seen["n_responses"] = len(responses)
        return torch.tensor([0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
                            dtype=torch.float32, device=trainer.device)

    trainer.compute_rewards = spy_reward
    out = trainer.train_step(["q1?", "q2?"], ["a1", "a2"])

    # batch=2, G=3 -> answers must be repeat-interleaved per question.
    assert seen["answers"] == ["a1"] * G + ["a2"] * G, \
        f"reward answers misaligned: {seen['answers']}"
    assert seen["n_responses"] == 2 * G

    assert isinstance(out["loss"], float) and math.isfinite(out["loss"])
    assert out["kl_penalty"] >= -1e-6, \
        f"k3 KL estimator must be non-negative, got {out['kl_penalty']}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), \
        "no (finite) parameter gradients after train_step"
    assert model.embed_tokens.weight.grad is not None, \
        "embed_tokens must receive gradient via the grad-carrying logprob forward"
    _pass("test_grpo",
          f"loss={out['loss']:.4f} policy={out['policy_loss']:.4f} "
          f"kl={out['kl_penalty']:.6f}, answers aligned, "
          f"params_with_grad={len(grads)}")


# ----------------------------------------------------------------------
# v5.43 (1/3): MultiEnv GRPO end-to-end loop — envs produce tasks, the
# trainer samples, envs score, and the SHARED update math consumes
# injected rewards (compute_rewards must be bypassed, not reimplemented)
# ----------------------------------------------------------------------
def test_env_grpo_loop():
    from helioslm_v5.src.training.env_grpo import EnvGRPO
    from helioslm_v5.src.training.envs import MultiEnvBatch
    from helioslm_v5.src.training.grpo import GRPOTrainer

    model, config = _lite_model(seed=311)
    ref_model, _ = _lite_model(seed=411)  # different init -> non-trivial KL
    config.grpo.group_size = 2
    config.grpo.max_new_tokens = 4
    trainer = GRPOTrainer(model, ref_model, config)
    batch = MultiEnvBatch({"math": 2, "code": 1})
    loop = EnvGRPO(trainer, batch)

    out = loop.rollout(seed=0)
    assert len(out["tasks"]) == 3
    assert len(out["samples"]) == 3 * 2, "flat repeat-interleaved samples"
    assert out["rewards"].shape == (6,)

    # Env routing: independently recomputed env rewards must match the
    # loop's rewards in exactly the same order.
    recomputed = []
    i = 0
    for t in out["tasks"]:
        for _ in range(2):
            recomputed.append(batch.reward(t, out["samples"][i]["response"]))
            assert out["samples"][i]["env"] == t["env"]
            i += 1
    assert torch.allclose(out["rewards"],
                          torch.tensor(recomputed, dtype=torch.float32)), \
        "loop rewards disagree with independent env scoring"

    # Injected-reward path: env rewards must BYPASS compute_rewards (the
    # math-answer heuristic would 0-out code-env RESULT literals that the
    # env scores correctly — bypassing, not reimplementing, is the point).
    def _boom(responses, answers):
        raise AssertionError("compute_rewards must not run on the env path")

    trainer.compute_rewards = _boom
    stats = loop.train_step(seed=1)
    assert isinstance(stats["loss"], float) and \
        math.isfinite(stats["loss"])
    assert stats["kl_penalty"] >= -1e-6, \
        f"k3 KL estimator must be non-negative, got {stats['kl_penalty']}"
    assert len(stats["env_rewards"]) == 6
    assert set(stats["mean_reward_by_env"]) == {"math", "code"}
    for name, m in stats["mean_reward_by_env"].items():
        assert 0.0 <= m <= 1.0, f"{name} mean reward {m} out of [0, 1]"

    # Loud error: a reward tensor of the wrong flat length is rejected.
    try:
        trainer._learn_from_samples(out["samples"], answers=None,
                                    rewards=torch.zeros(5))
        assert False, "wrong-length injected rewards must fail loudly"
    except ValueError:
        pass
    _pass("test_env_grpo_loop",
          f"loss={stats['loss']:.4f}, env rewards routed (3 tasks x "
          f"G=2), compute_rewards bypassed, KL={stats['kl_penalty']:.6f}")


# ----------------------------------------------------------------------
# v5.44 (3/3): AsyncEnvGRPO — the env-routed reward path on the AsyncGRPO
# producer/consumer skeleton: env tags ride every metrics dict, the update
# math is the shared _learn_from_samples (compute_rewards bypassed), and
# every dispatched task is consumed exactly once
# ----------------------------------------------------------------------
def test_async_env_grpo():
    from helioslm_v5.src.training.async_grpo import AsyncGRPO
    from helioslm_v5.src.training.env_grpo import AsyncEnvGRPO
    from helioslm_v5.src.training.envs import MultiEnvBatch
    from helioslm_v5.src.training.grpo import GRPOTrainer

    model, config = _lite_model(seed=511)
    ref_model, _ = _lite_model(seed=611)
    config.grpo.group_size = 2
    config.grpo.max_new_tokens = 4
    trainer = GRPOTrainer(model, ref_model, config)
    batch = MultiEnvBatch({"math": 2, "code": 1})
    loop = AsyncEnvGRPO(trainer, batch, n_workers=1)
    assert isinstance(loop, AsyncGRPO), "must ride the async skeleton"

    # Loud construction errors mirror EnvGRPO's guards.
    for bad in (lambda: AsyncEnvGRPO("not-a-trainer", batch),
                lambda: AsyncEnvGRPO(trainer, {"math": 1})):
        try:
            bad()
            assert False, "bad construction must fail loudly"
        except ValueError:
            pass

    # Env rewards must bypass compute_rewards (same contract as the
    # synchronous loop — the math-answer heuristic would mis-score code).
    def _boom(responses, answers):
        raise AssertionError("compute_rewards must not run on the env path")

    trainer.compute_rewards = _boom
    metrics = loop.run(seed=0)
    assert len(metrics) == 3, "one metrics dict per dispatched task"
    for m in metrics:
        assert m["env"] in ("math", "code"), "every metrics dict env-tagged"
        assert isinstance(m["loss"], float) and math.isfinite(m["loss"])
        assert m["kl_penalty"] >= -1e-6
    assert {m["env"] for m in metrics} == {"math", "code"}

    # Determinism sanity: a second run with the same task seed reproduces
    # the same env sequence (tasks are seeded, workers reseed per task).
    loop2 = AsyncEnvGRPO(trainer, batch, n_workers=1)
    loop2.trainer.compute_rewards = _boom
    metrics2 = loop2.run(seed=0)
    assert [m["env"] for m in metrics2] == [m["env"] for m in metrics]
    loop.stop()
    loop2.stop()
    _pass("test_async_env_grpo",
          f"{len(metrics)} tasks consumed (G=2), env tags "
          f"{[m['env'] for m in metrics]}, compute_rewards bypassed, "
          f"losses finite")
