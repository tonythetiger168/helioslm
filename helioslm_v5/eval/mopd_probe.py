"""Two-domain MOPD probe (v5.76, roadmap B2-6).

Prompts are ``BOS DOMAIN d1 d2 d3 d4`` over ten digit tokens; the answer
is four digits then EOS:

  rev   the digits reversed;
  inc   every digit + 1 (mod 10).

Each teacher is the same tiny HeliosLMv5 trained with cross-entropy on ONE
domain only, so it answers the other domain with its own rule. The student
gets a short cross-entropy warm start on both domains (``warm_student``;
MOPD post-trains an SFT model) and is then trained only on its own samples
scored by a teacher (``training.mopd.MOPDTrainer``). Routed MOPD sends rev prompts to
the rev teacher and inc prompts to the inc teacher; the single-teacher
control sends everything to the rev teacher.
"""
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from helioslm_v5.configs.config_v5 import (
    GRPOConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.training.grpo import GRPOTrainer
from helioslm_v5.src.training.mopd import MOPDTrainer

BOS, EOS = 1, 2
DOMAIN = {"rev": 3, "inc": 4}
DIGIT0, N_DIGITS, N_IN = 10, 10, 4
VOCAB = 32
ANSWER_LEN = N_IN + 1                 # digits + EOS


def config(lr=3e-3):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=2, vocab_size=VOCAB, hidden_size=64,
        intermediate_size=128,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=2, num_activated_experts=1,
                      expert_hidden_size=64),
        use_attention_residuals=False, use_hyper_connections=False,
        grpo=GRPOConfig(group_size=2, epsilon=0.2, kl_coef=0.0, lr=lr))


def answer(env, digits):
    if env == "rev":
        return digits[::-1]
    if env == "inc":
        return [(d + 1) % N_DIGITS for d in digits]
    raise ValueError(env)


def make_batch(envs, gen):
    """(prompt ids [B, 2 + N_IN], target ids [B, ANSWER_LEN])."""
    digits = torch.randint(0, N_DIGITS, (len(envs), N_IN), generator=gen)
    prompts, targets = [], []
    for env, d in zip(envs, digits.tolist()):
        prompts.append([BOS, DOMAIN[env]] + [DIGIT0 + x for x in d])
        targets.append([DIGIT0 + x for x in answer(env, d)] + [EOS])
    return torch.tensor(prompts), torch.tensor(targets)


def mixed_envs(n, gen):
    pick = torch.randint(0, 2, (n,), generator=gen).tolist()
    return ["rev" if p == 0 else "inc" for p in pick]


def train_ce(model, envs_fn, steps, seed, batch=64, lr=3e-3):
    """Teacher-forced cross-entropy on the answer tokens."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        p, t = make_batch(envs_fn(batch, gen), gen)
        ids = torch.cat([p, t], 1)
        logits = model(ids)[0][:, p.shape[1] - 1:-1]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def train_teacher(env, steps, seed):
    torch.manual_seed(seed)
    return train_ce(HeliosLMv5(config()), lambda n, g: [env] * n, steps, seed)


def warm_student(steps, seed):
    """The student before distillation: a short CE warm start on both
    domains (MOPD post-trains an SFT model, not a random one)."""
    torch.manual_seed(1000 + seed)
    return train_ce(HeliosLMv5(config()), mixed_envs, steps, 3000 + seed)


@torch.no_grad()
def exact_match(model, env, n=256, seed=0):
    gen = torch.Generator().manual_seed(10_000 + seed)
    p, t = make_batch([env] * n, gen)
    prev = model.training
    out = model.generate(p, max_new_tokens=ANSWER_LEN, temperature=0.0)
    model.train(prev)
    got = out[:, p.shape[1]:p.shape[1] + ANSWER_LEN]
    if got.shape[1] < ANSWER_LEN:
        got = F.pad(got, (0, ANSWER_LEN - got.shape[1]))
    return (got == t).all(1).float().mean().item()


@torch.no_grad()
def reverse_kl(student, teacher, env, n=128, seed=0):
    """Mean per-position KL(student || teacher) along the teacher-correct
    answer of held-out prompts (a fixed reference path)."""
    gen = torch.Generator().manual_seed(20_000 + seed)
    p, t = make_batch([env] * n, gen)
    ids = torch.cat([p, t], 1)
    P = p.shape[1]
    ls = torch.log_softmax(student(ids)[0][:, P - 1:-1].float(), -1)
    lt = torch.log_softmax(teacher(ids)[0][:, P - 1:-1].float(), -1)
    return (ls.exp() * (ls - lt)).sum(-1).mean().item()


def run_arm(arm, teachers, steps, seed, warm_steps=100, batch=32, lr=3e-4,
            log=None, ckpt=None):
    """Distil the warm-started student. ``arm`` in ARMS.

    ``lr`` is a fine-tuning rate: a fresh AdamW moves every weight by about
    lr on its first steps, and at the warm start's 3e-3 a single noisy
    distillation step erased the inc skill. ``ckpt`` (a path) saves every
    50 steps and resumes exactly (student, optimizer, data and sampling
    RNG states)."""
    cfg = config(lr)
    student = warm_student(warm_steps, seed)
    trainer = GRPOTrainer(student, copy.deepcopy(student), cfg)
    kw = dict(teacher_by_env={"rev": "rev", "inc": "inc"})
    if arm == "single_teacher":
        kw = dict(teacher_by_env={}, default_teacher="rev")
    elif arm == "tropd":
        kw.update(alpha=0.2, subtract_global_baseline=True)
    elif arm == "full_vocab":
        kw.update(full_vocab=True)
    elif arm != "mopd":
        raise ValueError(arm)
    mopd = MOPDTrainer(trainer, teachers, **kw)
    gen = torch.Generator().manual_seed(2000 + seed)
    torch.manual_seed(4000 + seed)            # sampling RNG
    curve, start = [], 0
    if ckpt is not None and Path(ckpt).exists():
        state = torch.load(ckpt, weights_only=False)
        student.load_state_dict(state["student"])
        trainer.optimizer.load_state_dict(state["opt"])
        gen.set_state(state["gen"])
        torch.set_rng_state(state["torch_rng"])
        curve, start = state["curve"], state["step"]
        if log:
            log(f"{arm}: resumed at step {start}")
    for step in range(start, steps):
        envs = mixed_envs(batch, gen)
        p, _ = make_batch(envs, gen)
        st = mopd.train_step(p, envs, ANSWER_LEN)
        if (step + 1) % 50 == 0:
            curve.append({"step": step + 1, "gap_mean": round(st["gap_mean"], 4)})
            if log:
                log(f"{arm} step {step + 1}: gap {st['gap_mean']:.3f}")
            if ckpt is not None:
                Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(str(ckpt) + ".tmp")
                torch.save({"student": student.state_dict(),
                            "opt": trainer.optimizer.state_dict(),
                            "gen": gen.get_state(),
                            "torch_rng": torch.get_rng_state(),
                            "curve": curve, "step": step + 1}, tmp)
                tmp.replace(ckpt)
    student.eval()
    return student, curve


ARMS = ("mopd", "tropd", "full_vocab", "single_teacher")
