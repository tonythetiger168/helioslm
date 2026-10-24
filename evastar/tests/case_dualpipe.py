"""HeliosLM v5 test cases — DualPipe pipeline-parallel training.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math

import torch
import torch.nn as nn

from evastar.configs.config_v5 import EvastarV5Config
from evastar.tests.case_common import (_lite_model, _mpdp_attnres_stage_builder, _mpdp_stage_builder, _pass)


# ----------------------------------------------------------------------
# DualPipe: run_dual gradients == direct forward+backward (mean over MBs)
# ----------------------------------------------------------------------
def test_dualpipe():
    from evastar.src.training.dualpipe import (
        DualPipeScheduler, DualPipeStage, LayerWrap)

    model, config = _lite_model(seed=108)

    # LayerWrap (dualpipe) adapts HeliosLMv5Layer ((hidden, kv, attn_res)
    # output) to the tensor->tensor stage contract; with attention
    # residuals off (lite default) it threads nothing (v5.4 behaviour).
    stage = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0]),
                                         LayerWrap(model.layers[1])]))
    torch.manual_seed(109)
    inputs = [torch.randn(2, 5, config.hidden_size) for _ in range(4)]
    loss_fn = lambda out: out.pow(2).mean()
    M = len(inputs)

    sched = DualPipeScheduler([stage], num_micro_batches=M)
    loss_pipe = sched.run_dual(inputs, loss_fn)
    grads_pipe = {n: p.grad.clone()
                  for n, p in model.named_parameters() if p.grad is not None}
    assert grads_pipe, "run_dual produced no parameter gradients"

    # Reference: direct forward+backward with the same 1/M scaling.
    model.zero_grad(set_to_none=True)
    loss_ref = 0.0
    for x in inputs:
        loss = loss_fn(stage(x)) / M
        loss.backward()
        loss_ref += loss.item()

    assert math.isfinite(loss_pipe) and math.isfinite(loss_ref)
    assert abs(loss_pipe - loss_ref) < 1e-5, \
        f"loss mismatch: pipe={loss_pipe:.6f} ref={loss_ref:.6f}"
    max_diff = 0.0
    for n, p in model.named_parameters():
        if n in grads_pipe:
            assert p.grad is not None, f"{n}: grad missing in reference"
            d = (grads_pipe[n] - p.grad).abs().max().item()
            max_diff = max(max_diff, d)
            assert torch.allclose(grads_pipe[n], p.grad, atol=1e-5, rtol=1e-4), \
                f"{n}: pipe grad != direct grad (max diff {d:.3e})"

    # All attention parameters of both layers must have received a gradient
    # (recompute backward reaches every stage parameter on the path).
    attn_params = [p for layer in (model.layers[0], model.layers[1])
                   for p in layer.attention.parameters()]
    assert all(p.grad is not None for p in attn_params)
    _pass("test_dualpipe",
          f"loss={loss_pipe:.5f} (ref {loss_ref:.5f}), "
          f"grads match (max diff {max_diff:.2e}, atol 1e-5)")


def test_multi_process_dualpipe():
    """v5.14: multi-process DualPipe == single-process scheduler.

    Two spawn workers, three micro-batches: outputs, input grads, and
    per-stage parameter grads must match DualPipeScheduler.run_forward +
    run_backward to fp32 rounding.
    """
    from evastar.src.training.dualpipe import (DualPipeScheduler,
                                                   DualPipeStage)
    from evastar.src.training.multi_process_dualpipe import (
        MultiProcessDualPipeRunner)

    # single-process reference
    torch.manual_seed(1234)
    stages = [_mpdp_stage_builder(r) for r in range(2)]
    for st in stages:
        st.train()
    gen = torch.Generator().manual_seed(42)
    inputs = [torch.randn(2, 32, generator=gen) for _ in range(3)]
    ggen = torch.Generator().manual_seed(43)
    grad_outs = [torch.randn(2, 16, generator=ggen) for _ in range(3)]
    ref = DualPipeScheduler(stages, num_micro_batches=len(inputs))
    ref_outs = ref.run_forward(inputs)
    ref_in_grads = ref.run_backward(grad_outs)
    ref_stage_grads = [{n: p.grad.clone() for n, p in st.named_parameters()}
                       for st in stages]

    # multi-process pipeline
    runner = MultiProcessDualPipeRunner(_mpdp_stage_builder, num_stages=2,
                                        init_seed=1234)
    try:
        outs = runner.run_forward(inputs)
        in_grads = runner.run_backward(grad_outs)
        stage_grads = runner.collect_grads()
    finally:
        runner.shutdown()

    for i in range(3):
        assert (outs[i] - ref_outs[i]).abs().max().item() < 1e-5, \
            f"mb{i}: output mismatch"
        assert (in_grads[i] - ref_in_grads[i]).abs().max().item() < 1e-5, \
            f"mb{i}: input grad mismatch"
    for r in range(2):
        for name, g in stage_grads[r].items():
            rg = ref_stage_grads[r][name]
            assert (g - rg).abs().max().item() < 1e-5, \
                f"stage{r}.{name}: param grad mismatch"
    _pass("test_multi_process_dualpipe",
          "2 ranks x 3 micro-batches: outputs/grads == single-process <1e-5")


# ----------------------------------------------------------------------
# v5.5 (M4/M5 regression): attention-residual accumulator is threaded
# explicitly through the DualPipe path — attn_res_gate receives gradient
# under run_dual, and pipeline gradients match the direct forward+backward
# reference bitwise (single stage AND a two-stage split where the
# accumulator crosses the stage boundary)
# ----------------------------------------------------------------------
def test_attention_residuals_dualpipe():
    from evastar.src.model_v5 import EvastarV5
    from evastar.src.training.dualpipe import (
        DualPipeScheduler, DualPipeStage, LayerWrap)

    torch.manual_seed(170)
    config = EvastarV5Config(size="lite")
    config.use_attention_residuals = True
    model = EvastarV5(config)
    assert all(l.attn_res_gate is not None for l in model.layers)

    torch.manual_seed(171)
    inputs = [torch.randn(2, 5, config.hidden_size) for _ in range(4)]
    loss_fn = lambda out: out.pow(2).mean()
    M = len(inputs)

    def run_pair(stages, reference):
        """run_dual on `stages` vs a direct forward+backward `reference`;
        return (loss_pipe, loss_ref, max_grad_diff, grads_pipe)."""
        model.zero_grad(set_to_none=True)
        sched = DualPipeScheduler(stages, num_micro_batches=M)
        loss_pipe = sched.run_dual(inputs, loss_fn)
        grads_pipe = {n: p.grad.clone()
                      for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        loss_ref = 0.0
        for x in inputs:
            loss = loss_fn(reference(x)) / M
            loss.backward()
            loss_ref += loss.item()
        assert abs(loss_pipe - loss_ref) < 1e-6, \
            f"loss mismatch: pipe={loss_pipe:.8f} ref={loss_ref:.8f}"
        max_d = 0.0
        for n, p in model.named_parameters():
            if n in grads_pipe:
                assert p.grad is not None, f"{n}: grad missing in reference"
                d = (grads_pipe[n] - p.grad).abs().max().item()
                max_d = max(max_d, d)
        return loss_pipe, loss_ref, max_d, grads_pipe

    # --- 1) single stage holding both layers ---
    stage = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0]),
                                         LayerWrap(model.layers[1])]))
    assert stage.threads_attn_res
    _, _, d1, grads1 = run_pair([stage], lambda x: stage(x)[0])
    g1 = grads1.get("layers.1.attn_res_gate")
    assert g1 is not None and g1.abs().item() > 0, \
        "attn_res_gate got no gradient through DualPipe (M4 regression)"
    assert d1 < 1e-6, f"1-stage pipe vs direct grad diff {d1:.3e}"

    # --- 2) two-stage split: the accumulator must cross the stage
    #    boundary through the scheduler's detached-carry leaves ---
    s0 = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0])]))
    s1 = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[1])]))

    def ref2(x):
        h, ares = s0(x)
        h, _ = s1(h, ares)
        return h

    _, _, d2, grads2 = run_pair([s0, s1], ref2)
    g1b = grads2.get("layers.1.attn_res_gate")
    assert g1b is not None and g1b.abs().item() > 0, \
        "attn_res_gate grad missing when the accumulator crosses stages"
    assert d2 < 1e-6, f"2-stage pipe vs direct grad diff {d2:.3e}"

    # --- 3) residuals OFF: LayerWrap/stage degrade to the plain v5.4
    #    tensor->tensor contract (no carry threaded) ---
    model_off, _ = _lite_model(seed=175)
    stage_off = DualPipeStage(nn.ModuleList([LayerWrap(model_off.layers[0]),
                                             LayerWrap(model_off.layers[1])]))
    assert not stage_off.threads_attn_res
    assert torch.is_tensor(stage_off(torch.randn(1, 3, config.hidden_size)))
    _pass("test_attention_residuals_dualpipe",
          f"gate grads flow under DualPipe (|g1|={g1.abs().item():.3e}, "
          f"2-stage |g1|={g1b.abs().item():.3e}), pipe==direct bitwise "
          f"(max diff {max(d1, d2):.1e}), residuals-off path unchanged")


def test_multi_process_dualpipe_attention_residuals():
    """v5.20: the multi-process worker must seed a zeros gradient for the
    dropped attention-residual accumulator (regression: the None upstream
    grad crashed backward; a scalar accumulator would have silently taken
    implicit ONES). Mirrors DualPipeScheduler._backward_stage semantics."""
    from evastar.src.training.dualpipe import DualPipeScheduler
    from evastar.src.training.multi_process_dualpipe import (
        MultiProcessDualPipeRunner)

    config = EvastarV5Config(size="lite")
    torch.manual_seed(5678)
    stages = [_mpdp_attnres_stage_builder(r) for r in range(2)]
    for st in stages:
        st.train()
    gen = torch.Generator().manual_seed(44)
    inputs = [torch.randn(2, 5, config.hidden_size, generator=gen)
              for _ in range(3)]
    ggen = torch.Generator().manual_seed(45)
    grad_outs = [torch.randn(2, 5, config.hidden_size, generator=ggen)
                 for _ in range(3)]

    ref = DualPipeScheduler(stages, num_micro_batches=len(inputs))
    ref_outs = ref.run_forward(inputs)
    ref_in_grads = ref.run_backward(grad_outs)
    ref_stage_grads = [{n: p.grad.clone() for n, p in st.named_parameters()
                        if p.grad is not None} for st in stages]

    runner = MultiProcessDualPipeRunner(_mpdp_attnres_stage_builder,
                                        num_stages=2, init_seed=4321)
    try:
        outs = runner.run_forward(inputs)
        in_grads = runner.run_backward(grad_outs)
        stage_grads = runner.collect_grads()
    finally:
        runner.shutdown()

    for i in range(3):
        assert (outs[i] - ref_outs[i]).abs().max().item() < 1e-5, \
            f"mb{i}: output mismatch"
        assert (in_grads[i] - ref_in_grads[i]).abs().max().item() < 1e-5, \
            f"mb{i}: input grad mismatch"
    for r in range(2):
        for name, g in stage_grads[r].items():
            rg = ref_stage_grads[r].get(name)
            if g is None or rg is None:
                assert g is None and rg is None, \
                    f"stage{r}.{name}: grad presence mismatch"
                continue
            assert (g - rg).abs().max().item() < 1e-5, \
                f"stage{r}.{name}: param grad mismatch"

    def gate_grad(sd):
        return next((g for n, g in sd.items() if n.endswith("attn_res_gate")),
                    None)

    # the last stage's accumulator feeds downstream layers: its gate must
    # receive the (zeros-seeded) upstream gradient, matching single-process
    for tag, sd in (("ref", ref_stage_grads[1]), ("mp", stage_grads[1])):
        g = gate_grad(sd)
        assert g is not None and g.abs().item() > 0, \
            f"{tag} stage-1 attn_res_gate got no accumulator gradient"
    _pass("test_multi_process_dualpipe_attention_residuals",
          "attn-res pipeline: outputs/grads == single-process <1e-5, "
          "accumulator grad flows")
