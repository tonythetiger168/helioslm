"""T124 — LB6 RL length schedule and DAPO overlong handling (v5.117).

Pins:

1. overlong_penalty follows DAPO eq. 13 (0, linear ramp to -1, -1) and
   two_stage_lengths builds a valid short-then-long schedule;
2. the trainer's length limit follows grpo.length_stages as updates
   complete, and train_step passes the current limit to generate and
   records it on every sample;
3. overlong_buffer subtracts the ramp from the rewards of long samples
   only (mean_reward shows it); samples without a recorded limit are
   untouched;
4. overlong_filter drops truncated samples (limit reached without EOS)
   from the loss; a sample that ends in EOS at the limit is kept;
   truncated_frac and mean_response_tokens are reported;
5. with the defaults the update is bit-identical to a trainer that
   predates LB6 (no reward change, nothing filtered);
6. validation.

Run from repo root: python3 helioslm_v5/tests/test_grpo_length.py
"""
import copy
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    GRPOConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training.grpo import (  # noqa: E402
    GRPOTrainer, overlong_penalty, two_stage_lengths)

G = 4
EOS = 2


def _cfg(**grpo):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=1, vocab_size=16, hidden_size=64,
        intermediate_size=128,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=2, num_activated_experts=1,
                      expert_hidden_size=64),
        use_attention_residuals=False, use_hyper_connections=False,
        eos_token_id=EOS,
        grpo=GRPOConfig(group_size=G, epsilon=0.2, kl_coef=0.0, lr=1e-2,
                        **grpo))


def _trainer(**grpo):
    cfg = _cfg(**grpo)
    torch.manual_seed(0)
    model = HeliosLMv5(cfg)
    return model, GRPOTrainer(model, copy.deepcopy(model), cfg)


def _samples(model, spec):
    """spec: (n_response_tokens, limit or None, ends_with_eos)."""
    gen = torch.Generator().manual_seed(1)
    out = []
    for n, lim, eos in spec:
        ids = torch.randint(3, 16, (1, 1 + n), generator=gen)
        if eos:
            ids[0, -1] = EOS
        s = {"full_ids": ids, "prompt_len": 1}
        if lim is not None:
            s["max_new_tokens"] = lim
        out.append(s)
    model.eval()
    with torch.no_grad():
        for s in out:
            s["old_logprob"] = GRPOTrainer._sequence_logprob(
                model, s["full_ids"], 1).detach()
    return out


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


REWARDS = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0])
# 8 samples, limit 8: two truncated (8 tokens, no EOS), one ending in EOS
# at the limit, the rest short
SPEC = [(8, 8, False), (3, 8, False), (8, 8, True), (5, 8, False),
        (2, 8, False), (7, 8, False), (8, 8, False), (4, 8, False)]


def test_penalty_and_helper():
    assert overlong_penalty(4, 8, 4) == 0.0
    assert overlong_penalty(6, 8, 4) == -0.5
    assert overlong_penalty(8, 8, 4) == -1.0
    assert overlong_penalty(9, 8, 4) == -1.0
    st = two_stage_lengths(100, 512, 2048)
    assert st == [(0, 512), (50, 2048)]
    GRPOConfig(length_stages=st, overlong_buffer=128)        # valid
    for bad in ((0, 512, 512), (1.0, 4, 8), (0.5, 8, 4)):
        try:
            two_stage_lengths(10, bad[1], bad[2], bad[0])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("PASS test_penalty_and_helper")


def test_stages_follow_updates():
    model, tr = _trainer(length_stages=[(0, 3), (2, 6)])
    seen = []

    class Tok:
        def encode(self, text):
            return [5, 6]

        def decode(self, ids):
            return "x"

    tr.tokenizer = Tok()

    def fake_generate(prompt_ids, max_new_tokens, **kw):
        seen.append(max_new_tokens)
        return torch.cat([prompt_ids, torch.full(
            (1, max_new_tokens), 7, dtype=prompt_ids.dtype)], dim=1)

    model.generate = fake_generate
    for step in range(4):
        assert tr.current_max_new_tokens() == (3 if step < 2 else 6)
        st = tr.train_step(["q"], ["1"])
        assert st["mean_response_tokens"] == (3 if step < 2 else 6)
        assert st["truncated_frac"] == 1.0          # never ends in EOS
    assert seen == [3] * (2 * G) + [6] * (2 * G)
    assert tr.rl_step == 4
    print("PASS test_stages_follow_updates")


def test_overlong_buffer_rewards():
    model, tr = _trainer(overlong_buffer=4)
    st = tr._learn_from_samples(_samples(model, SPEC), None,
                                rewards=REWARDS.clone())
    pen = [overlong_penalty(n, 8, 4) for n, _, _ in SPEC]
    want = (REWARDS + torch.tensor(pen)).mean().item()
    assert abs(st["mean_reward"] - want) < 1e-6, (st["mean_reward"], want)
    # samples without a recorded limit get no penalty
    model2, tr2 = _trainer(overlong_buffer=4)
    st2 = tr2._learn_from_samples(
        _samples(model2, [(n, None, e) for n, _, e in SPEC]), None,
        rewards=REWARDS.clone())
    assert abs(st2["mean_reward"] - REWARDS.mean().item()) < 1e-6
    print("PASS test_overlong_buffer_rewards")


def test_overlong_filter():
    model, tr = _trainer(overlong_filter=True)
    st = tr._learn_from_samples(_samples(model, SPEC), None,
                                rewards=REWARDS.clone())
    assert st["n_kept"] == 6, st                   # samples 0 and 6 dropped
    assert st["truncated_frac"] == 2 / 8
    assert st["mean_response_tokens"] == sum(n for n, _, _ in SPEC) / 8
    print("PASS test_overlong_filter")


def test_defaults_unchanged():
    model_a, tr_a = _trainer()
    model_b, tr_b = _trainer()
    # b's samples carry limits (as train_step's now do); a's do not
    sa = _samples(model_a, [(n, None, e) for n, _, e in SPEC])
    sb = _samples(model_b, SPEC)
    st_a = tr_a._learn_from_samples(sa, None, rewards=REWARDS.clone())
    st_b = tr_b._learn_from_samples(sb, None, rewards=REWARDS.clone())
    assert st_a["loss"] == st_b["loss"] and st_b["n_kept"] == 8
    assert torch.equal(_params(model_a), _params(model_b))
    print("PASS test_defaults_unchanged")


def test_validation():
    for bad in (dict(length_stages=[]), dict(length_stages=[(1, 8)]),
                dict(length_stages=[(0, 8), (0, 16)]),
                dict(length_stages=[(0, 0)]),
                dict(length_stages=[(0, 8.0)]),
                dict(overlong_buffer=0), dict(overlong_buffer=True),
                dict(length_stages=[(0, 8)], overlong_buffer=8),
                dict(overlong_filter=1)):
        try:
            GRPOConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    GRPOConfig(length_stages=[[0, 8], [10, 16]], overlong_buffer=4)  # JSON
    print("PASS test_validation")


if __name__ == "__main__":
    test_penalty_and_helper()
    test_stages_follow_updates()
    test_overlong_buffer_rewards()
    test_overlong_filter()
    test_defaults_unchanged()
    test_validation()
