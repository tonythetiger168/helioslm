"""DualPipe - Bidirectional Pipeline Parallelism (DeepSeek-V3 Style)

NOTE: This module is a **single-process simulation** of the DualPipe
schedule. There is no distributed code here: no process groups, no p2p
send/recv, and no true cross-device overlap of forward and backward waves.
What it does reproduce on a single device is the *ordering* of the
DualPipe-style schedule:

  - warmup:       forward pass(es) only
  - steady state: interleaved 1F1B (one forward, one backward)
  - cooldown:     backward pass(es) only

Backward correctness is achieved via **activation recomputation**: the
forward pass stores only the detached stage inputs (as
``requires_grad_(True)`` leaves), and each backward recomputes the stage
forward to rebuild a connected autograd graph before calling
``torch.autograd.backward``. Parameter gradients therefore flow correctly
while only stage inputs (not full graphs) are kept alive between F and B.
"""
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Tuple


class DualPipeStage(nn.Module):
    """Single pipeline stage for DualPipe."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DualPipeScheduler:
    """Single-process simulation of the DualPipe schedule.

    The schedule ordering (warmup F -> steady-state 1F1B -> cooldown B) is
    simulated on one process; there is no actual multi-GPU parallelism or
    communication. ``self.trace`` records the schedule as a list of
    ("F", micro_batch_idx) / ("B", micro_batch_idx) events so the ordering
    can be inspected or verified.
    """

    def __init__(self, stages: List[DualPipeStage], num_micro_batches: int = 8):
        if num_micro_batches < 1:
            raise ValueError(
                f"num_micro_batches must be >= 1, got {num_micro_batches}")
        self.stages = list(stages)
        self.num_stages = len(stages)
        # Declared micro-batch count; run_dual validates its input list
        # against this so the parameter is load-bearing, not decorative.
        self.num_micro_batches = num_micro_batches

        # stage_inputs[s][i] = detached leaf input of micro-batch i at stage s.
        # Only these leaves are kept between forward and backward; activations
        # are recomputed during the backward pass.
        self._stage_inputs: List[Dict[int, torch.Tensor]] = [
            {} for _ in range(self.num_stages)
        ]
        # rng_states[s][i] = RNG state captured right BEFORE stage s ran the
        # forward of micro-batch i. Restored during activation recomputation
        # so dropout (and any other stochastic op) replays the identical
        # masks in the backward pass — the recompute is then self-consistent
        # with the original forward.
        self._rng_states: List[Dict[int, dict]] = [
            {} for _ in range(self.num_stages)
        ]
        # Schedule trace: list of ("F"|"B", micro_batch_idx) events.
        self.trace: List = []

    # ------------------------------------------------------------------
    # RNG capture / restore for self-consistent activation recomputation
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_rng_state() -> dict:
        state = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        torch.random.set_rng_state(state["cpu"])
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _recompute_stage(self, s: int, mb_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recompute stage s's forward of micro-batch mb_idx under the RNG
        state captured at the original forward, leaving the global RNG state
        untouched afterwards (fork semantics).

        Returns ``(stage_output, input_leaf)``: the recomputed, grad-carrying
        stage output and the stored detached input leaf (whose ``.grad`` the
        caller reads after backward)."""
        leaf = self._stage_inputs[s].pop(mb_idx)
        rng_state = self._rng_states[s].pop(mb_idx)
        saved = self._capture_rng_state()
        self._restore_rng_state(rng_state)
        try:
            out = self.stages[s](leaf)
        finally:
            self._restore_rng_state(saved)
        return out, leaf

    # ------------------------------------------------------------------
    # Per-micro-batch primitives
    # ------------------------------------------------------------------
    def _forward_one(self, mb_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Forward one micro-batch through all stages (no graph retained).

        For each stage, the input is turned into a detached leaf
        (``requires_grad_(True)``) and stored; the stage itself runs under
        ``torch.no_grad()`` because the graph is rebuilt (recomputed) during
        the backward pass.
        """
        self.trace.append(("F", mb_idx))
        for s, stage in enumerate(self.stages):
            leaf = x.detach().requires_grad_(True)
            self._stage_inputs[s][mb_idx] = leaf
            # Capture the RNG state right before the stage forward so the
            # backward-pass recomputation replays identical stochastic ops
            # (e.g. dropout masks) — without this the detached-leaf scheme
            # is not self-consistent for stochastic layers.
            self._rng_states[s][mb_idx] = self._capture_rng_state()
            with torch.no_grad():
                x = stage(leaf)
        return x

    def _backward_one(self, mb_idx: int, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward one micro-batch through all stages (reversed order).

        Each stage forward is *recomputed* from its stored leaf input to
        build a fresh autograd graph; ``torch.autograd.backward`` on the
        recomputed output then produces both the stage parameter gradients
        and the leaf ``.grad`` passed upstream.
        """
        self.trace.append(("B", mb_idx))
        grad = grad_output
        for s in reversed(range(self.num_stages)):
            # Activation recomputation under the forward's RNG state:
            # rebuild the stage's autograd graph with identical stochastic ops.
            out, leaf = self._recompute_stage(s, mb_idx)
            torch.autograd.backward(out, grad_tensors=grad)
            grad = leaf.grad
        return grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the forward pass for all micro-batches (values only)."""
        return [self._forward_one(i, x) for i, x in enumerate(inputs)]

    def run_backward(self, grad_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the backward pass for all micro-batches.

        Stage forwards are recomputed from the stored leaf inputs (activation
        recomputation), so gradients flow to the stage parameters.
        """
        input_grads = [None] * len(grad_outputs)
        for i in reversed(range(len(grad_outputs))):
            input_grads[i] = self._backward_one(i, grad_outputs[i])
        return input_grads

    def run_dual(self, inputs: List[torch.Tensor], loss_fn: Callable) -> torch.Tensor:
        """Run the simulated DualPipe schedule and return the mean loss.

        Schedule (M = number of micro-batches):
          - warmup:       forward micro-batch 0
          - steady state: for k = 1..M-1: forward micro-batch k, then
                          backward micro-batch k-1   (1F1B interleaving)
          - cooldown:     backward micro-batch M-1

        The loss is the *mean* over micro-batches (each micro-batch loss is
        scaled by 1/M before backward), matching a conventional
        micro-batch-averaged gradient.

        ``len(inputs)`` must equal the ``num_micro_batches`` declared at
        construction (the parameter is wired, not decorative), and the
        input list must be non-empty.
        """
        M = len(inputs)
        if M == 0:
            raise ValueError("run_dual requires a non-empty inputs list")
        if M != self.num_micro_batches:
            raise ValueError(
                f"run_dual got {M} micro-batches but the scheduler was "
                f"constructed with num_micro_batches={self.num_micro_batches}; "
                "construct the scheduler with the intended micro-batch count"
            )
        self.trace = []
        total_loss = 0.0

        def backward_with_loss(i: int) -> float:
            # Recompute the final stage with autograd enabled (under the
            # forward's captured RNG state) so the loss itself seeds the
            # gradient of this micro-batch.
            last = self.num_stages - 1
            self.trace.append(("B", i))
            out, leaf = self._recompute_stage(last, i)
            loss = loss_fn(out) / M        # micro-batch mean scaling
            loss.backward()
            grad = leaf.grad
            for s in reversed(range(last)):
                # Activation recomputation for upstream stages (RNG-restored).
                out_s, leaf_s = self._recompute_stage(s, i)
                torch.autograd.backward(out_s, grad_tensors=grad)
                grad = leaf_s.grad
            return loss.item()

        # --- warmup: forward of first micro-batch ---
        self._forward_one(0, inputs[0])
        # --- steady state: 1F1B interleaving ---
        for k in range(1, M):
            self._forward_one(k, inputs[k])        # 1F
            total_loss += backward_with_loss(k - 1)  # 1B
        # --- cooldown: backward of last micro-batch ---
        total_loss += backward_with_loss(M - 1)

        return total_loss


class ExpertParallelism:
    """Expert-to-device mapping for MoE expert parallelism.

    Only the routing/mapping logic is implemented; real communication
    requires an initialized ``torch.distributed`` process group.
    """

    def __init__(self, num_experts: int, num_devices: int):
        if num_experts % num_devices != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"num_devices ({num_devices})"
            )
        self.num_experts = num_experts
        self.num_devices = num_devices
        self.experts_per_device = num_experts // num_devices

        # Vectorized expert -> device lookup table.
        self.expert_device_lut = torch.arange(num_experts) // self.experts_per_device

    def route_to_device(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Route expert IDs to their respective devices (vectorized lookup).

        Out-of-range ids (negative or >= num_experts) are rejected instead
        of silently wrapping/indexing garbage.
        """
        if expert_ids.numel() > 0:
            lo = int(expert_ids.min())
            hi = int(expert_ids.max())
            if lo < 0 or hi >= self.num_experts:
                raise ValueError(
                    f"expert_ids out of range [{lo}, {hi}]; valid range is "
                    f"[0, {self.num_experts})"
                )
        lut = self.expert_device_lut.to(device=expert_ids.device)
        return lut[expert_ids]

    def all_to_all(self, tensors: List[torch.Tensor], src_device: int, dst_device: int):
        """All-to-all communication between devices.

        ``src_device``/``dst_device`` identify the participating ranks of
        this exchange and are validated against the configured device count.
        Requires an initialized torch.distributed process group; raises
        NotImplementedError in a single-process environment instead of
        silently pretending to communicate.
        """
        for name, dev in (("src_device", src_device), ("dst_device", dst_device)):
            if not (0 <= dev < self.num_devices):
                raise ValueError(
                    f"{name}={dev} out of range; valid range is "
                    f"[0, {self.num_devices})"
                )
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise NotImplementedError(
                "all_to_all requires an initialized torch.distributed process "
                "group; this single-process environment cannot perform real "
                "expert-parallel communication."
            )
        outputs = [torch.empty_like(t) for t in tensors]
        torch.distributed.all_to_all(outputs, tensors)
        return outputs
