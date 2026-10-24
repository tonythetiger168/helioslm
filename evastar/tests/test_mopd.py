"""T115 — roadmap B2-6: multi-teacher on-policy distillation (MOPD), v5.76.

Pins:

1. the estimator: for a one-token response, the sampled-token loss
   -A * ratio (A = log pi_T - log pi_S, ratio 1 on fresh samples) has an
   expected gradient (exact enumeration over the vocabulary, weighted by
   the student's probabilities) equal to the gradient of the exact reverse
   KL(student || teacher); the full-vocabulary loss has that gradient
   directly;
2. self-distillation (teacher == student): every advantage is 0, and both
   losses give zero gradient;
3. TROPD: alpha 1 equals plain MOPD; alpha < 1 bounds every advantage
   below by log(1 - alpha); the global baseline centres the advantages;
4. routing: each sample uses its env's teacher, unmapped envs use the
   default or raise in strict mode; bad configurations raise;
5. the action mask: environment tokens get no advantage and no loss;
6. the committed probe artifact (two domains, one teacher each): routed
   arms keep and sharpen the student's inc skill and move it toward the
   rev teacher (without learning exact reversal in 600 steps); one
   teacher for everything erases inc and is worse on rev too.

Run from repo root: python3 evastar/tests/test_mopd.py
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval.mopd_probe import ARMS, config  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.src.training.grpo import GRPOTrainer  # noqa: E402
from evastar.src.training.mopd import MOPDTrainer  # noqa: E402

_ART = _REPO / "benchmarks" / "mopd_probe_2026-10-20.json"
V = 32


def _models(seed_s=0, seed_t=1):
    cfg = config()
    torch.manual_seed(seed_s)
    student = EvastarV5(cfg).eval()
    torch.manual_seed(seed_t)
    teacher = EvastarV5(cfg).eval()
    return cfg, student, teacher


def _mopd(student, teachers, cfg, **kw):
    tr = GRPOTrainer(student, copy.deepcopy(student), cfg)
    kw.setdefault("teacher_by_env", {k: k for k in teachers})
    return MOPDTrainer(tr, teachers, **kw)


def _grads(model):
    return torch.cat([(p.grad if p.grad is not None
                       else torch.zeros_like(p)).flatten()
                      for p in model.parameters()])


def test_expected_gradient_is_reverse_kl():
    cfg, student, teacher = _models()
    prompt = torch.tensor([[1, 3, 12, 15, 11, 19]])
    P = prompt.shape[1]
    # exact reverse KL at the next position and its gradient
    student.zero_grad()
    ls = torch.log_softmax(student(prompt)[0][0, -1].float(), -1)
    with torch.no_grad():
        lt = torch.log_softmax(teacher(prompt)[0][0, -1].float(), -1)
    (ls.exp() * (ls - lt)).sum().backward()
    g_kl = _grads(student).clone()
    p = ls.detach().exp()
    # sampled-token estimator, enumerated over every token y
    m = _mopd(student, {"a": teacher}, cfg)
    g_est = torch.zeros_like(g_kl)
    for y in range(V):
        s = {"full_ids": torch.cat([prompt, torch.tensor([[y]])], 1),
             "prompt_len": P, "env": "a"}
        adv, old, gap = m.advantages([s])
        assert torch.allclose(adv[0], lt[y:y + 1] - ls[y:y + 1].detach(),
                              atol=1e-5)
        student.zero_grad()
        m.sample_loss(s, adv[0], old[0]).backward()
        g_est += p[y] * _grads(student)
    assert torch.allclose(g_est, g_kl, atol=1e-5, rtol=1e-4), \
        (g_est - g_kl).abs().max()
    # full-vocabulary loss: the same gradient from any one sample
    f = _mopd(student, {"a": teacher}, cfg, full_vocab=True)
    s = {"full_ids": torch.cat([prompt, torch.tensor([[12]])], 1),
         "prompt_len": P, "env": "a"}
    adv, old, _ = f.advantages([s])
    student.zero_grad()
    f.sample_loss(s, adv[0], old[0]).backward()
    assert torch.allclose(_grads(student), g_kl, atol=1e-6)
    print("PASS test_expected_gradient_is_reverse_kl")


def test_self_distillation_is_zero():
    cfg, student, _ = _models()
    teacher = copy.deepcopy(student)
    ids = torch.tensor([[1, 4, 10, 11, 12, 13, 11, 12, 13, 14, 2]])
    s = {"full_ids": ids, "prompt_len": 6, "env": "a"}
    for full in (False, True):
        m = _mopd(student, {"a": teacher}, cfg, full_vocab=full)
        adv, old, gap = m.advantages([dict(s)])
        assert gap[0].abs().max().item() < 1e-5
        student.zero_grad()
        m.sample_loss(s, adv[0], old[0]).backward()
        assert _grads(student).abs().max().item() < 1e-5, full
    print("PASS test_self_distillation_is_zero")


def test_tropd_and_baseline():
    cfg, student, teacher = _models()
    ids = torch.tensor([[1, 3, 10, 11, 12, 13, 17, 5, 30, 9, 2]])
    s = {"full_ids": ids, "prompt_len": 6, "env": "a"}
    plain = _mopd(student, {"a": teacher}, cfg).advantages([dict(s)])[0][0]
    one = _mopd(student, {"a": teacher}, cfg,
                alpha=1.0).advantages([dict(s)])[0][0]
    assert torch.equal(plain, one)
    for a in (0.2, 0.5, 0.9):
        adv = _mopd(student, {"a": teacher}, cfg,
                    alpha=a).advantages([dict(s)])[0][0]
        assert (adv >= math.log(1 - a) - 1e-6).all(), (a, adv)
        assert (adv.abs() <= plain.abs() + 1e-6).all()
    s2 = {"full_ids": ids[:, :8], "prompt_len": 6, "env": "a"}
    advs = _mopd(student, {"a": teacher}, cfg, alpha=0.2,
                 subtract_global_baseline=True).advantages(
        [dict(s), dict(s2)])[0]
    assert abs(torch.cat(advs).mean().item()) < 1e-6
    print("PASS test_tropd_and_baseline")


def test_routing_and_validation():
    cfg, student, t1 = _models()
    t2 = copy.deepcopy(t1)
    with torch.no_grad():
        for p in t2.parameters():
            p.add_(0.05)
    ids = torch.tensor([[1, 3, 10, 11, 12, 13, 14, 2]])
    m = _mopd(student, {"x": t1, "y": t2}, cfg,
              teacher_by_env={"rev": "x", "inc": "y"}, default_teacher="x")
    assert [m.teacher_for({"env": e}) for e in ("rev", "inc", "other")] \
        == ["x", "y", "x"]
    gx = m.advantages([{"full_ids": ids, "prompt_len": 6, "env": "rev"}])[2][0]
    gy = m.advantages([{"full_ids": ids, "prompt_len": 6, "env": "inc"}])[2][0]
    old = GRPOTrainer._token_logprobs(student, ids, 6)
    assert torch.allclose(gx, GRPOTrainer._token_logprobs(t1, ids, 6) - old,
                          atol=1e-6)
    assert torch.allclose(gy, GRPOTrainer._token_logprobs(t2, ids, 6) - old,
                          atol=1e-6)
    strict = _mopd(student, {"x": t1}, cfg, teacher_by_env={"rev": "x"},
                   default_teacher="x", strict=True)
    nodefault = _mopd(student, {"x": t1}, cfg, teacher_by_env={"rev": "x"})
    for mm in (strict, nodefault):
        try:
            mm.teacher_for({"env": "inc"})
        except KeyError:
            pass
        else:
            raise AssertionError("unmapped env accepted")
    tr = GRPOTrainer(student, copy.deepcopy(student), cfg)
    for bad in (dict(teacher_by_env={"rev": "zz"}),
                dict(default_teacher="zz"), dict(alpha=0.0),
                dict(alpha=1.5), dict(full_vocab=True, alpha=0.5),
                dict(full_vocab=True, subtract_global_baseline=True)):
        try:
            MOPDTrainer(tr, {"x": t1}, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    try:
        MOPDTrainer(student, {"x": t1})
    except ValueError:
        pass
    else:
        raise AssertionError("non-GRPOTrainer accepted")
    print("PASS test_routing_and_validation")


def test_action_mask():
    cfg, student, teacher = _models()
    ids = torch.tensor([[1, 3, 10, 11, 12, 13, 20, 21, 22, 14, 2]])
    mask = torch.tensor([True, False, False, False, True])
    s = {"full_ids": ids, "prompt_len": 6, "env": "a", "action_mask": mask}
    m = _mopd(student, {"a": teacher}, cfg)
    adv, old, _ = m.advantages([s])
    assert adv[0].numel() == 2 and old[0].numel() == 2
    # changing an environment token AFTER the last policy token changes
    # nothing; the loss only sees the policy positions
    full = _mopd(student, {"a": teacher}, cfg, full_vocab=True)
    kl = full._full_vocab_kl(s)
    unmasked = full._full_vocab_kl({k: v for k, v in s.items()
                                    if k != "action_mask"})
    assert kl.item() < unmasked.item()
    print("PASS test_action_mask")


def test_probe_artifact():
    art = json.loads(_ART.read_text())
    res = {r["arm"]: r for r in art["results"]}
    assert set(res) == set(ARMS)
    for env in ("rev", "inc"):
        t = art["teachers"][env]
        assert t["exact_match"][env] == 1.0
        assert t["exact_match"]["inc" if env == "rev" else "rev"] == 0.0
    warm = art["warm_start"]
    assert warm["exact_match"] == {"rev": 0.0, "inc": 1.0}
    routed = ("mopd", "tropd", "full_vocab")
    for arm in routed:
        r = res[arm]
        # the skill the student has is kept and sharpened ...
        assert r["exact_match"]["inc"] >= 0.99, arm
        assert r["reverse_kl"]["inc"] < warm["reverse_kl"]["inc"] / 2, arm
        # ... and it moves toward the rev teacher, but 600 on-policy steps
        # do not teach this 2-layer student exact reversal
        assert r["reverse_kl"]["rev"] < 0.6 * warm["reverse_kl"]["rev"], arm
        assert r["exact_match"]["rev"] < 0.05, arm
    # one teacher for everything erases inc, and is worse on rev too
    st = res["single_teacher"]
    assert st["exact_match"]["inc"] < 0.05, st
    assert st["reverse_kl"]["inc"] > 10 * warm["reverse_kl"]["inc"]
    assert all(st["reverse_kl"]["rev"] > res[a]["reverse_kl"]["rev"]
               for a in routed)
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_probe_artifact " + json.dumps(
        {a: [r["exact_match"], r["reverse_kl"]] for a, r in res.items()}))


if __name__ == "__main__":
    test_expected_gradient_is_reverse_kl()
    test_self_distillation_is_zero()
    test_tropd_and_baseline()
    test_routing_and_validation()
    test_action_mask()
    test_probe_artifact()
