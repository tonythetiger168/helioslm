"""T127 — LB7 checkpoint merging and top-k-union on-policy distillation
(v5.121).

Pins:

1. merge_states linear: equal weights average, weights renormalise, one
   model is returned unchanged; integer buffers are copied, not averaged;
2. task_arithmetic: base + scale * sum w_i (theta_i - base); with one
   model, weight 1 and scale 1 it is that model;
3. ties: a hand-checked tensor (trim by magnitude, sign election, mean of
   the agreeing entries); density 1 with one model is task arithmetic;
4. validation: key / shape mismatch, missing base, bad weights / density
   / method are refused;
5. merge_checkpoints.py merges two pretrain checkpoints into one that
   loads into HeliosLMv5 and runs, and copies run.json;
6. topk_union_reverse_kl: k = V equals the exact reverse KL; on a
   hand-built case the support is the union and both sides are
   renormalised; gradients reach the student only;
7. MOPDTrainer(full_vocab=True, topk_union=k): the per-sample loss uses
   it (k = V matches full_vocab), it trains, and bad settings raise.

Run from repo root: python3 helioslm_v5/tests/test_merge.py
"""
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.mopd_probe import config  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training.grpo import GRPOTrainer  # noqa: E402
from helioslm_v5.src.training.merge import (  # noqa: E402
    merge_states, ties_merge)
from helioslm_v5.src.training.mopd import (  # noqa: E402
    MOPDTrainer, topk_union_reverse_kl)


def _st(*vals):
    return {"w": torch.tensor(vals, dtype=torch.float32),
            "step": torch.tensor(7)}


def test_linear():
    a, b = _st(1.0, 2.0), _st(3.0, 6.0)
    m = merge_states([a, b])
    assert torch.equal(m["w"], torch.tensor([2.0, 4.0]))
    assert m["step"].item() == 7 and m["step"].dtype == torch.int64
    m = merge_states([a, b], weights=[3, 1])
    assert torch.allclose(m["w"], torch.tensor([1.5, 3.0]))
    assert torch.equal(merge_states([a])["w"], a["w"])
    h = {"w": a["w"].half(), "step": a["step"]}
    assert merge_states([h, h])["w"].dtype == torch.float16
    print("PASS test_linear")


def test_task_arithmetic():
    base, a, b = _st(1.0, 1.0), _st(2.0, 0.0), _st(1.0, 3.0)
    m = merge_states([a, b], "task_arithmetic", base=base, scale=0.5)
    # deltas (1, -1) and (0, 2): base + 0.5 * (1, 1)
    assert torch.allclose(m["w"], torch.tensor([1.5, 1.5]))
    m = merge_states([a], "task_arithmetic", base=base)
    assert torch.equal(m["w"], a["w"])
    print("PASS test_task_arithmetic")


def test_ties():
    d1 = torch.tensor([0.9, -0.1, 0.5, 0.0])
    d2 = torch.tensor([0.7, 0.3, -0.8, 0.2])
    # density 0.5 keeps 2 of 4 entries per vector:
    #   d1 -> (0.9, 0, 0.5, 0), d2 -> (0.7, 0, -0.8, 0)
    # signs from the sum: (+, 0, -, 0); agreeing means: (0.8, 0, -0.8, 0)
    m = ties_merge([d1, d2], [1.0, 1.0], 0.5)
    assert torch.allclose(m, torch.tensor([0.8, 0.0, -0.8, 0.0])), m
    base, a = _st(1.0, 2.0), _st(0.5, 4.0)
    t = merge_states([a], "ties", base=base, density=1.0)
    assert torch.allclose(t["w"], a["w"])
    print("PASS test_ties")


def test_validation():
    a = _st(1.0, 2.0)
    bad = [
        dict(states=[a, {"w": a["w"]}]),
        dict(states=[a, _st(1.0, 2.0, 3.0)]),
        dict(states=[a], method="ties"),
        dict(states=[a], method="soup"),
        dict(states=[a], weights=[1, 2]),
        dict(states=[a], weights=[-1]),
        dict(states=[a], method="ties", base=a, density=0.0),
        dict(states=[]),
    ]
    for kw in bad:
        try:
            merge_states(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kw}")
    print("PASS test_validation")


def test_merge_script():
    cfg = config()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = []
        for i in range(2):
            torch.manual_seed(i)
            p = d / f"m{i}" / "ckpt.pt"
            p.parent.mkdir()
            torch.save({"model": HeliosLMv5(cfg).state_dict(), "step": 1}, p)
            paths.append(str(p))
        (d / "m0" / "run.json").write_text('{"arch": "x"}')
        subprocess.run([sys.executable,
                        str(_REPO / "examples" / "merge_checkpoints.py"),
                        "--models", *paths, "--out",
                        str(d / "soup" / "ckpt.pt")],
                       check=True, capture_output=True)
        st = torch.load(d / "soup" / "ckpt.pt", weights_only=False)
        assert st["merge"]["method"] == "linear"
        assert json.loads((d / "soup" / "run.json").read_text()) == \
            {"arch": "x"}
        model = HeliosLMv5(cfg)
        model.load_state_dict(st["model"])
        a = torch.load(paths[0], weights_only=False)["model"]
        b = torch.load(paths[1], weights_only=False)["model"]
        k = next(k for k, v in a.items() if torch.is_floating_point(v))
        assert torch.allclose(st["model"][k], (a[k] + b[k]) / 2)
        out = model(torch.tensor([[1, 3, 5]]))[0]
        assert torch.isfinite(out).all()
    print("PASS test_merge_script")


def test_topk_union_kl():
    torch.manual_seed(0)
    s = torch.randn(3, 10, requires_grad=True)
    t = torch.randn(3, 10)
    ls, lt = torch.log_softmax(s, -1), torch.log_softmax(t, -1)
    exact = (ls.exp() * (ls - lt)).sum(-1)
    assert torch.allclose(topk_union_reverse_kl(s, t, 10), exact, atol=1e-6)
    # hand case: V = 4, k = 1; student top is 0, teacher top is 1
    sl = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.1]))
    tl = torch.log(torch.tensor([0.1, 0.6, 0.2, 0.1]))
    ps = torch.tensor([0.5, 0.2]) / 0.7
    pt = torch.tensor([0.1, 0.6]) / 0.7
    want = (ps * (ps.log() - pt.log())).sum()
    got = topk_union_reverse_kl(sl, tl, 1)
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    t2 = t.clone().requires_grad_(False)
    topk_union_reverse_kl(s, t2, 3).sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert math.isfinite(got.item())
    print("PASS test_topk_union_kl")


def _mopd(student, teacher, cfg, **kw):
    tr = GRPOTrainer(student, copy.deepcopy(student), cfg)
    return MOPDTrainer(tr, {"a": teacher}, teacher_by_env={"a": "a"}, **kw)


def test_mopd_topk_union():
    cfg = config()
    torch.manual_seed(0)
    student = HeliosLMv5(cfg)
    torch.manual_seed(1)
    teacher = HeliosLMv5(cfg)
    ids = torch.tensor([[1, 3, 12, 15, 11, 19, 7, 8]])
    s = {"full_ids": ids, "prompt_len": 5, "env": "a"}
    V = cfg.vocab_size
    full = _mopd(student, teacher, cfg, full_vocab=True)
    union = _mopd(student, teacher, cfg, full_vocab=True, topk_union=V)
    assert torch.allclose(full._full_vocab_kl(s), union._full_vocab_kl(s),
                          atol=1e-5)
    small = _mopd(student, teacher, cfg, full_vocab=True, topk_union=4)
    before = small._full_vocab_kl(s).item()
    for _ in range(5):
        small.learn([dict(s)])
    assert small._full_vocab_kl(s).item() < before
    for kw in (dict(topk_union=4), dict(full_vocab=True, topk_union=-1),
               dict(full_vocab=True, topk_union=True),
               dict(full_vocab=True, topk_union=2.0)):
        try:
            _mopd(student, teacher, cfg, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kw}")
    print("PASS test_mopd_topk_union")


if __name__ == "__main__":
    test_linear()
    test_task_arithmetic()
    test_ties()
    test_validation()
    test_merge_script()
    test_topk_union_kl()
    test_mopd_topk_union()
