"""T102 — roadmap U1 (DAPO / Olmo 3 OlmoRL options for GRPO), v5.64.

Pins, on fresh on-policy samples (ratio == 1 before the update):

1. defaults reproduce a hand-written sequence-level GRPO update exactly
   (loss and updated parameters);
2. token-level loss: the policy loss equals -sum_i A_i * n_i / T (each
   sample's advantage counted once per policy token, divided by the total
   token count) and the gradient equals the hand-built token objective;
3. normalize_adv_std=False uses raw centered rewards;
4. filter_zero_adv_groups: flat groups are dropped from loss and
   normalization; an all-flat batch leaves the parameters untouched;
5. TIS: sampler log-probs equal to the old ones give weight 1 (same
   update as without TIS); a lower sampler log-prob is capped at tis_cap;
6. clip-higher: a sample whose ratio is 1.25 with a positive advantage
   is clipped (no gradient) at epsilon 0.2 but not at clip_eps_high 0.28;
7. validation.

Run from repo root: python3 helioslm_v5/tests/test_grpo_u1.py
"""
import copy
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    GRPOConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training.grpo import GRPOTrainer  # noqa: E402

G = 4


def _cfg(**grpo):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=1, vocab_size=16, hidden_size=64,
        intermediate_size=128,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=2, num_activated_experts=1,
                      expert_hidden_size=64),
        use_attention_residuals=False, use_hyper_connections=False,
        grpo=GRPOConfig(group_size=G, epsilon=0.2, kl_coef=0.0, lr=1e-2,
                        **grpo))


def _setup(**grpo):
    cfg = _cfg(**grpo)
    torch.manual_seed(0)
    model = HeliosLMv5(cfg)
    trainer = GRPOTrainer(model, copy.deepcopy(model), cfg)
    gen = torch.Generator().manual_seed(1)
    samples = []
    for n in (3, 5, 2, 4, 6, 3, 2, 5):            # varying response lengths
        ids = torch.randint(1, 16, (1, 1 + n), generator=gen)
        samples.append({"full_ids": ids, "prompt_len": 1})
    model.eval()
    with torch.no_grad():
        for s in samples:
            s["old_logprob"] = GRPOTrainer._sequence_logprob(
                model, s["full_ids"], 1).detach()
    return model, trainer, samples


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


REWARDS = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0])


def test_defaults_match_hand_written_update():
    model, trainer, samples = _setup()
    ref = copy.deepcopy(model)
    opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.01)
    r = REWARDS.view(-1, G)
    adv = ((r - r.mean(1, keepdim=True))
           / r.std(1, unbiased=False, keepdim=True).clamp(min=1e-8)).reshape(-1)
    ref.train()
    opt.zero_grad()
    for i, s in enumerate(samples):
        lp = GRPOTrainer._sequence_logprob(ref, s["full_ids"], 1)
        ratio = torch.exp((lp - s["old_logprob"]).clamp(-20, 20))
        pol = -torch.min(ratio * adv[i], torch.clamp(ratio, 0.8, 1.2) * adv[i])
        (pol / len(samples)).backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    opt.step()
    stats = trainer._learn_from_samples(samples, None, rewards=REWARDS)
    assert torch.equal(_params(model), _params(ref))
    assert stats["n_kept"] == len(samples)
    print("PASS test_defaults_match_hand_written_update")


def test_token_level_loss():
    model, trainer, samples = _setup(loss_agg="token")
    lens = torch.tensor([s["full_ids"].shape[1] - 1 for s in samples],
                        dtype=torch.float)
    r = REWARDS.view(-1, G)
    adv = ((r - r.mean(1, keepdim=True))
           / r.std(1, unbiased=False, keepdim=True).clamp(min=1e-8)).reshape(-1)
    T = lens.sum()
    want = -(adv * lens).sum().item() / T.item()

    ref = copy.deepcopy(model)
    ref.train()
    obj = sum(-adv[i] * GRPOTrainer._token_logprobs(ref, s["full_ids"], 1).sum()
              for i, s in enumerate(samples)) / T
    obj.backward()
    want_grad = torch.cat([p.grad.flatten() for p in ref.parameters()
                           if p.grad is not None])

    trainer.optimizer.step = lambda: None        # inspect gradients only
    stats = trainer._learn_from_samples(samples, None, rewards=REWARDS)
    assert abs(stats["policy_loss"] - want) < 1e-6, (stats, want)
    got_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                          if p.grad is not None])
    # trainer clips the grad norm to 1.0; compare directions and the clip
    scale = min(1.0, 1.0 / (want_grad.norm().item() + 1e-6))
    assert torch.allclose(got_grad, want_grad * scale, atol=1e-6)
    print("PASS test_token_level_loss")


def test_no_std_normalization():
    model, trainer, samples = _setup(normalize_adv_std=False)
    r = REWARDS.view(-1, G)
    adv = (r - r.mean(1, keepdim=True)).reshape(-1)
    trainer.optimizer.step = lambda: None
    stats = trainer._learn_from_samples(samples, None, rewards=REWARDS)
    # 1e-6 like test_token_level_loss: since v5.139 the rollout logprobs
    # (eval, absorbed MLA) and the training ones (expanded MLA) can differ
    # by one float32 ulp, so the importance ratio is 1 only to ~1e-7.
    assert abs(stats["policy_loss"] - (-adv.sum().item() / 8)) < 1e-6
    print("PASS test_no_std_normalization")


def test_zero_adv_group_filtering():
    model, trainer, samples = _setup(filter_zero_adv_groups=True)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    stats = trainer._learn_from_samples(samples, None, rewards=rewards)
    assert stats["n_kept"] == 4
    # filtered == running only the informative group unfiltered
    m2, t2, s2 = _setup()
    t2._learn_from_samples(s2[4:], None, rewards=rewards[4:])
    assert torch.equal(_params(model), _params(m2))
    m3, t3, s3 = _setup(filter_zero_adv_groups=True)
    before = _params(m3)
    st = t3._learn_from_samples(s3, None, rewards=torch.zeros(8))
    assert st["n_kept"] == 0 and torch.equal(before, _params(m3))
    print("PASS test_zero_adv_group_filtering")


def test_tis():
    m1, t1, s1 = _setup(tis_cap=2.0)
    for s in s1:
        s["sampler_logprob"] = s["old_logprob"].clone()   # weight exactly 1
    t1._learn_from_samples(s1, None, rewards=REWARDS)
    m0, t0, s0 = _setup()
    t0._learn_from_samples(s0, None, rewards=REWARDS)
    assert torch.equal(_params(m1), _params(m0))

    m2, t2, s2 = _setup(tis_cap=2.0)
    for s in s2:
        s["sampler_logprob"] = s["old_logprob"] - 5.0     # exp(5) -> cap 2
    t2.optimizer.step = lambda: None
    t0b = _setup()[1]
    t0b.optimizer.step = lambda: None
    a = t2._learn_from_samples(s2, None, rewards=REWARDS)["policy_loss"]
    b = t0b._learn_from_samples(_setup()[2], None, rewards=REWARDS)["policy_loss"]
    assert abs(a - 2.0 * b) < 1e-6, (a, b)
    print("PASS test_tis")


def test_clip_higher():
    def grad_for(**kw):
        model, trainer, samples = _setup(**kw)
        s = [samples[0], samples[1], samples[2], samples[3]]
        s[0] = dict(s[0], old_logprob=s[0]["old_logprob"] - math.log(1.25))
        trainer.optimizer.step = lambda: None
        trainer._learn_from_samples(
            s, None, rewards=torch.tensor([1.0, 0.0, 0.0, 0.0]))
        return torch.cat([p.grad.flatten() for p in model.parameters()
                          if p.grad is not None])
    g_clip = grad_for()
    g_high = grad_for(clip_eps_high=0.28)
    assert not torch.allclose(g_clip, g_high)
    print("PASS test_clip_higher")


def test_validation():
    for kw, needle in ((dict(loss_agg="mean"), "loss_agg"),
                       (dict(clip_eps_high=0.0), "clip_eps_high"),
                       (dict(normalize_adv_std=1), "normalize_adv_std"),
                       (dict(tis_cap=0.5), "tis_cap")):
        try:
            GRPOConfig(**kw)
        except ValueError as e:
            assert needle in str(e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


if __name__ == "__main__":
    test_defaults_match_hand_written_update()
    test_token_level_loss()
    test_no_std_normalization()
    test_zero_adv_group_filtering()
    test_tis()
    test_clip_higher()
    test_validation()
