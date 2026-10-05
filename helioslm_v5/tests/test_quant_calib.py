"""v5.42 — calibration × quantization cross oracles (Phase 2.3).

The measurement must be trustworthy before any drift claim: a calibrated
stub must give ECE ~= 0, an overconfident stub must give ECE ~= conf-acc,
and the bf16-vs-NVFP4 comparison must leave the caller's model untouched.

Run from repo root: python helioslm_v5/tests/test_quant_calib.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.eval.quant_calib import (compare_bf16_vs_nvfp4, ece,
                                          run_quant_calib_probe)
from helioslm_v5.src.model_v5 import HeliosLMv5


class _BiasStub(nn.Module):
    """Logits are a per-token bias vector (content-blind). Softmax over
    the choice tokens then reproduces whatever confidence profile the
    test installs — a calibration ground-truth fixture."""

    def __init__(self, vocab=64, bias=None):
        super().__init__()
        b = torch.zeros(vocab) if bias is None else bias
        self.bias = nn.Parameter(b, requires_grad=False)

    def forward(self, ids):
        B, T = ids.shape
        return self.bias.unsqueeze(0).unsqueeze(0).expand(B, T, -1), None, None


def _mc_tasks(n, p_correct=0.8, choices=(3, 7)):
    """Choices (3, 7): 3 is correct a p_correct fraction of the time —
    with matching stub bias this yields a perfectly calibrated profile."""
    tasks = []
    for i in range(n):
        ans = choices[0] if (i % 100) < p_correct * 100 else choices[1]
        tasks.append({"prompt_ids": [10, 11, 12],
                      "choice_ids": list(choices),
                      "answer_idx": 0 if ans == choices[0] else 1})
    return tasks


def test_ece_perfectly_calibrated_stub():
    n = 500
    tasks = _mc_tasks(n, p_correct=0.8)
    bias = torch.zeros(64)
    bias[3] = torch.log(torch.tensor(0.8 / 0.2))   # P(3)=0.8 vs P(7)=0.2
    stub = _BiasStub(bias=bias)
    r = run_quant_calib_probe(stub, tasks, tag="calibrated-stub")
    assert abs(r["accuracy"] - 0.8) < 0.05, r
    assert r["ece"] < 0.05, f"calibrated stub ECE {r['ece']:.4f} != ~0"
    print(f"PASS test_ece_perfectly_calibrated_stub acc={r['accuracy']:.3f} "
          f"ece={r['ece']:.4f}")


def test_ece_overconfident_stub_positive():
    n = 500
    tasks = _mc_tasks(n, p_correct=0.8)
    bias = torch.zeros(64)
    bias[3] = torch.log(torch.tensor(0.99 / 0.01))
    stub = _BiasStub(bias=bias)
    r = run_quant_calib_probe(stub, tasks, tag="overconfident-stub")
    # ECE should approach |0.99 - 0.8| = 0.19
    assert 0.10 < r["ece"] < 0.25, f"ECE {r['ece']:.4f} not in overconfidence band"
    print(f"PASS test_ece_overconfident_stub_positive ece={r['ece']:.4f} "
          f"(~|0.99-0.8|=0.19)")


def test_ece_loud_errors():
    for bad in ([], None):
        try:
            ece([], [])
        except ValueError:
            pass
        else:
            raise AssertionError("empty ece accepted")
    try:
        ece([0.5], [1, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched ece inputs accepted")
    print("PASS test_ece_loud_errors")


def test_nvfp4_drift_runs_and_caller_untouched():
    torch.manual_seed(0)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    before = [p.detach().clone() for p in model.parameters()]
    tasks = [{"prompt_ids": [5, 6, 7], "choice_ids": [8, 9],
              "answer_idx": i % 2} for i in range(8)]
    rep = compare_bf16_vs_nvfp4(model, tasks,
                                model_tag="HeliosLMv5-lite-untrained")
    # caller's model untouched (deepcopy inside the runner)
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p), "comparison mutated the caller's model"
    for side in ("bf16", "nvfp4"):
        assert 0.0 <= rep[side]["accuracy"] <= 1.0
        assert 0.0 <= rep[side]["ece"] <= 1.0
    assert set(rep["drift"]) == {"accuracy", "ece"}
    assert "fakequant" in rep["method"], "deviation must be recorded"
    # untrained model: numbers are mechanics smoke, NOT a drift claim —
    # the report says so itself
    print(f"PASS test_nvfp4_drift_runs_and_caller_untouched "
          f"acc {rep['bf16']['accuracy']:.3f}->{rep['nvfp4']['accuracy']:.3f} "
          f"ece {rep['bf16']['ece']:.3f}->{rep['nvfp4']['ece']:.3f} "
          f"(untrained = mechanics only)")


if __name__ == "__main__":
    test_ece_perfectly_calibrated_stub()
    test_ece_overconfident_stub_positive()
    test_ece_loud_errors()
    test_nvfp4_drift_runs_and_caller_untouched()
    print("\n4/4 quant-calib cross tests passed")
