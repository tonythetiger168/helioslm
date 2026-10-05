"""Multi-process DualPipe (v5.14).

Runs a pipeline of ``DualPipeStage`` modules across OS processes, one rank
per stage, exchanging detached activations and gradients through
``torch.multiprocessing`` queues. Autograd graphs never cross process
boundaries: every rank stores the detached leaf input of each micro-batch,
captures the RNG state before its forward, and rebuilds the graph by
recomputation during backward — the same self-consistency scheme as the
single-process ``DualPipeScheduler``.

Semantics are identical to ``DualPipeScheduler.run_forward`` +
``run_backward`` (all-forward, then all-backward in reverse): per-stage op
order is the same, so outputs and parameter gradients match the
single-process simulation to fp32 rounding.

Phased worker protocol (no select needed — phases are strictly ordered):
  phase A (forward):  ranks read fwd micro-batches from upstream until a
                      ``fwd_done`` sentinel, forward each, pass outputs down.
  phase B (backward): ranks read bwd grads from downstream until a
                      ``bwd_done`` sentinel, recompute the stage from the
                      stored leaf (+RNG state), backward, pass leaf grads up.
  phase C (control):  driver-only control queue serves ``grads`` and
                      ``shutdown``.

Queue map:
  fwd[i]  — written by rank i-1 (driver if i=0), read by rank i
            (driver if i == num_stages)
  bwd[i]  — written by rank i+1 (driver if i == num_stages), read by rank i
            (bwd[0] is rank0's output channel back to the driver)
"""

import multiprocessing as _mp
from typing import Callable, Dict, List

import torch

from helioslm_v5.src.training.dualpipe import DualPipeStage

__all__ = ["MultiProcessDualPipeRunner"]


def _worker(rank: int, num_stages: int,
            stage_builder: Callable[[int], DualPipeStage],
            fwd_in, fwd_out, bwd_in, bwd_out, ctrl_q, grad_q, init_seed: int):
    """One process per stage. ``stage_builder(rank)`` must construct the
    rank's stage deterministically (workers reseed with init_seed + rank)
    so parameters match the single-process reference built with the same
    seeds."""
    torch.manual_seed(init_seed + rank)
    torch.set_num_threads(1)
    stage = stage_builder(rank)
    stage.train()

    stage_inputs: Dict[int, tuple] = {}
    rng_states: Dict[int, torch.Tensor] = {}

    def run_stage(leaf, ares_leaf):
        if getattr(stage, "threads_attn_res", False):
            return stage(leaf, ares_leaf)
        return stage(leaf), None

    # ---- phase A: forward micro-batches until the sentinel -------------
    while True:
        msg = fwd_in.get(timeout=120)
        if msg[0] == "fwd_done":
            # propagate the sentinel downstream so every rank exits the
            # phase (the driver writes it to fwd[0] only)
            if rank < num_stages - 1:
                fwd_out.put(msg)
            break
        _, mb_idx, x, attn_res = msg
        leaf = x.detach().requires_grad_(True)
        ares_leaf = (attn_res.detach().requires_grad_(True)
                     if getattr(stage, "threads_attn_res", False)
                     and attn_res is not None else None)
        stage_inputs[mb_idx] = (leaf, ares_leaf)
        rng_states[mb_idx] = torch.get_rng_state()
        with torch.no_grad():
            out, ares_out = run_stage(leaf, ares_leaf)
        fwd_out.put(("fwd", mb_idx, out.detach(), ares_out))

    # ---- phase B: backward micro-batches until the sentinel ------------
    while True:
        msg = bwd_in.get(timeout=120)
        if msg[0] == "bwd_done":
            # propagate upstream so rank 0 also exits (the driver writes it
            # to bwd[num_stages] only)
            if rank > 0:
                bwd_out.put(msg)
            break
        _, mb_idx, grad, ares_grad = msg
        leaf, ares_leaf = stage_inputs.pop(mb_idx)
        torch.set_rng_state(rng_states.pop(mb_idx))
        out, ares_out = run_stage(leaf, ares_leaf)
        # Mirror DualPipeScheduler._backward_stage: a missing upstream
        # accumulator gradient is mathematically a zero gradient. Passing
        # the None straight through crashes backward on a non-scalar
        # ares_out (and would seed an implicit ONES gradient on a scalar
        # one), killing the worker and hanging the driver.
        outs, grads = [out], [grad]
        if ares_out is not None and ares_out.requires_grad:
            outs.append(ares_out)
            grads.append(ares_grad if ares_grad is not None
                         else torch.zeros_like(ares_out))
        torch.autograd.backward(outs, grads)
        bwd_out.put(("bwd", mb_idx, leaf.grad,
                     ares_leaf.grad if ares_leaf is not None else None))

    # ---- phase C: control (driver-only) --------------------------------
    while True:
        msg = ctrl_q.get(timeout=120)
        if msg[0] == "grads":
            grad_q.put(("grads", rank, {
                n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in stage.named_parameters()}))
        elif msg[0] == "shutdown":
            break
        else:
            raise RuntimeError(f"rank {rank}: unknown control message {msg[0]!r}")


class MultiProcessDualPipeRunner:
    """Driver for a multi-process DualPipe pipeline.

    Args:
        stage_builder: ``rank -> DualPipeStage``; must be deterministic
            given ``init_seed`` (workers reseed with ``init_seed + rank``).
        num_stages: pipeline depth (= number of worker processes).
        init_seed: base seed; rank r builds with ``init_seed + r``.

    Usage:
        runner = MultiProcessDualPipeRunner(builder, num_stages=2)
        outs = runner.run_forward(inputs)          # one per micro-batch
        in_grads = runner.run_backward(grads)      # reverse order internally
        stage_grads = runner.collect_grads()       # [{name: tensor}, ...]
        runner.shutdown()
    """

    def __init__(self, stage_builder: Callable[[int], DualPipeStage],
                 num_stages: int, init_seed: int = 1234):
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        self.num_stages = num_stages
        self.init_seed = init_seed
        ctx = _mp.get_context("spawn")
        # fwd[i]: written by rank i-1 (driver if i=0), read by rank i
        # (driver if i == num_stages). bwd[i]: written by rank i+1
        # (driver if i == num_stages), read by rank i (bwd[0] is rank0's
        # output channel back to the driver).
        self._fwd = [ctx.Queue() for _ in range(num_stages + 1)]
        self._bwd = [ctx.Queue() for _ in range(num_stages + 1)]
        self._ctrl = [ctx.Queue() for _ in range(num_stages)]
        self._grad_q = ctx.Queue()
        self._procs = []
        for r in range(num_stages):
            p = ctx.Process(
                target=_worker,
                args=(r, num_stages, stage_builder,
                      self._fwd[r], self._fwd[r + 1],      # fwd_in, fwd_out
                      self._bwd[r + 1], self._bwd[r],      # bwd_in, bwd_out
                      self._ctrl[r], self._grad_q, init_seed),
                daemon=True)
            p.start()
            self._procs.append(p)
        self._live = True

    # public API ----------------------------------------------------------
    def run_forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-forward pass; returns detached outputs (one per micro-batch)."""
        for i, x in enumerate(inputs):
            self._fwd[0].put(("fwd", i, x, None))
        self._fwd[0].put(("fwd_done",))
        outs = [None] * len(inputs)
        for _ in inputs:
            kind, i, out, _ = self._fwd[self.num_stages].get(timeout=120)
            assert kind == "fwd"
            outs[i] = out
        return outs

    def run_backward(self, grad_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-backward pass in reverse micro-batch order.

        Returns input gradients (grad of the first stage's leaf), one per
        micro-batch, in the original input order.
        """
        ch = self._bwd[self.num_stages]
        for i in reversed(range(len(grad_outputs))):
            ch.put(("bwd", i, grad_outputs[i], None))
        ch.put(("bwd_done",))
        grads = [None] * len(grad_outputs)
        for _ in grad_outputs:
            kind, i, g, _ = self._bwd[0].get(timeout=120)
            assert kind == "bwd"
            grads[i] = g
        return grads

    def collect_grads(self) -> List[Dict[str, torch.Tensor]]:
        """Per-rank accumulated parameter gradients (rank order)."""
        for r in range(self.num_stages):
            self._ctrl[r].put(("grads",))
        per_rank: Dict[int, dict] = {}
        for _ in range(self.num_stages):
            kind, rank, grads = self._grad_q.get(timeout=120)
            assert kind == "grads"
            per_rank[rank] = grads
        return [per_rank[r] for r in range(self.num_stages)]

    def shutdown(self):
        if not self._live:
            return
        for r in range(self.num_stages):
            try:
                self._ctrl[r].put(("shutdown",))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._live = False

    def __del__(self):
        self.shutdown()
