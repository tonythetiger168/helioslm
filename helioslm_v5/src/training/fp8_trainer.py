"""FP8 Mixed Precision Training - DeepSeek-V3 Style

E4M3 for forward activations/weights, E5M2 for gradients (via backward
hooks). Dynamic per-tensor scaling with delayed scaling, and an AdamW
optimizer over fp32 master weights.

Quantization is real: values are rounded onto the representable FP8 grid,
either by casting through a native ``torch.float8_*`` dtype (PyTorch >= 2.1)
or, when the runtime lacks float8 support, by an explicit round-to-nearest
grid simulation with a straight-through gradient.
"""
import warnings

import torch
import torch.nn as nn

_FP8_RANGES = {
    # max_val, mantissa bits, min normal exponent (subnormal step = 2^(min_exp - mantissa))
    "e4m3": {"max_val": 448.0, "mantissa_bits": 3, "min_exp": -6,
             "dtype": getattr(torch, "float8_e4m3fn", None)},
    "e5m2": {"max_val": 57344.0, "mantissa_bits": 2, "min_exp": -14,
             "dtype": getattr(torch, "float8_e5m2", None)},
}


def _round_to_fp8_grid(v: torch.Tensor, max_val: float, mantissa_bits: int,
                       min_exp: int) -> torch.Tensor:
    """Round-to-nearest onto the representable FP8 grid (simulation).

    Fallback for runtimes without a native float8 dtype. Each value is
    rounded to the nearest grid point with mantissa grid precision
    2^(-mantissa_bits) relative to its binade; the subnormal floor is
    handled by clamping the exponent at ``min_exp`` (which yields the
    correct subnormal step 2^(min_exp - mantissa_bits)).

    A straight-through estimator is used: the returned *value* is
    quantized, while gradients pass through as identity.
    """
    abs_v = v.abs()
    exp = torch.floor(torch.log2(abs_v.clamp(min=2.0 ** min_exp))).clamp(min=min_exp)
    step = torch.pow(2.0, exp - mantissa_bits)
    q = (torch.round(abs_v / step) * step).clamp(max=max_val)
    q = torch.copysign(q, v)
    # Straight-through estimator: quantized value, identity gradient.
    return v + (q - v).detach()


class FP8Linear(nn.Module):
    """FP8 quantized linear layer with dynamic (delayed) scaling."""

    def __init__(self, in_features, out_features, bias=False, fp8_format="e4m3"):
        super().__init__()
        if fp8_format not in _FP8_RANGES:
            raise ValueError(f"fp8_format must be one of {list(_FP8_RANGES)}, "
                             f"got {fp8_format!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_format = fp8_format  # "e4m3" or "e5m2" — controls forward format

        # High precision master weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Dynamic scaling factors
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("weight_scale", torch.tensor(1.0))

        # History for delayed scaling (registered buffers so they are
        # captured by state_dict / checkpoints). Each history ring has its
        # own cursor: ``input_history_idx`` drives ``amax_history``
        # (activations) and ``weight_history_idx`` drives
        # ``weight_amax_history`` (weights).
        self.register_buffer("amax_history", torch.zeros(100))
        self.register_buffer("weight_amax_history", torch.zeros(100))
        self.register_buffer("input_history_idx", torch.zeros((), dtype=torch.long))
        self.register_buffer("weight_history_idx", torch.zeros((), dtype=torch.long))

    def _quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor,
                         format: str = None) -> torch.Tensor:
        """Quantize tensor to FP8 (real quantization, not a no-op).

        Prefers a native cast through the ``torch.float8_*`` dtype:
            x.div(scale).clamp(min, max).to(float8).to(x.dtype).mul(scale)
        Falls back to round-to-nearest simulation of the FP8 grid.
        The matmul in ``forward`` runs on these quantized values.
        """
        format = format or self.fp8_format
        spec = _FP8_RANGES[format]
        scale = torch.clamp(scale, min=1e-8)
        scaled = x / scale
        if spec["dtype"] is not None:
            try:
                q = scaled.clamp(-spec["max_val"], spec["max_val"]) \
                         .to(spec["dtype"]).to(x.dtype) * scale
                # NOTE: a straight-through estimator is required here.
                # Autograd through a native float8 cast silently casts the
                # *incoming gradient* to float8 as well; gradients smaller
                # than the min subnormal (2^-9 for e4m3) then underflow to
                # zero, killing all learning. STE: quantized value,
                # identity gradient.
                return x + (q - x).detach()
            except (RuntimeError, TypeError):
                pass  # fall through to grid simulation below
        # Simulate the FP8 grid with round-to-nearest (see _round_to_fp8_grid).
        return _round_to_fp8_grid(
            scaled, spec["max_val"], spec["mantissa_bits"], spec["min_exp"]
        ) * scale

    def _update_scale(self, x: torch.Tensor, history: torch.Tensor,
                      idx: torch.Tensor) -> torch.Tensor:
        """Update dynamic scale from amax history (tensor ops, no .item()).

        NaN guard (M-T2): a non-finite amax (NaN/±inf from a poisoned
        batch) is replaced with 0.0 BEFORE it is written into the history
        ring, so one bad batch can never permanently contaminate the
        delayed-scaling window. ``idx`` is this history's own cursor
        buffer (see __init__).
        """
        spec = _FP8_RANGES[self.fp8_format]
        amax = torch.nan_to_num(
            x.detach().abs().max(), nan=0.0, posinf=0.0, neginf=0.0)
        slot = int(idx) % history.numel()
        history[slot] = amax
        idx.add_(1)
        # Delayed scaling: use history max; clamp keeps the buffer a tensor
        # even for an all-zero batch.
        return torch.clamp(history.max() / spec["max_val"], min=1e-8)

    def _grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Quantize gradients to E5M2 (DeepSeek-V3 recipe)."""
        with torch.no_grad():
            spec = _FP8_RANGES["e5m2"]
            gscale = torch.clamp(grad.abs().max() / spec["max_val"], min=1e-8)
            return self._quantize_to_fp8(grad, gscale, "e5m2")

    def forward(self, x):
        """FP8 forward with high-precision output."""
        # Update scales (input scale from activation amax history, weight
        # scale refreshed from the weight amax every step).
        self.input_scale.copy_(self._update_scale(
            x, self.amax_history, self.input_history_idx))
        self.weight_scale.copy_(self._update_scale(
            self.weight, self.weight_amax_history, self.weight_history_idx))

        # Quantize inputs and weights to the configured FP8 format.
        x_fp8 = self._quantize_to_fp8(x, self.input_scale, self.fp8_format)
        w_fp8 = self._quantize_to_fp8(self.weight, self.weight_scale,
                                      self.fp8_format)

        # E5M2 gradient quantization via backward hooks.
        if torch.is_grad_enabled():
            if x_fp8.requires_grad:
                x_fp8.register_hook(self._grad_hook)
            if w_fp8.requires_grad:
                w_fp8.register_hook(self._grad_hook)

        # Matmul on the quantized values (production: torch._scaled_mm).
        output = torch.matmul(x_fp8, w_fp8.t())

        if self.bias is not None:
            output = output + self.bias

        return output


class FP8Trainer:
    """Training wrapper with FP8 mixed precision.

    Forward runs through FP8Linear layers; gradients are E5M2-quantized via
    backward hooks; optimization uses AdamW over fp32 master weights with
    grad clipping, and master weights are copied back to the model after
    each step.
    """

    def __init__(self, model, config, weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0):
        self.model = model
        self.config = config
        self.fp8_format = getattr(config, "fp8_format", "e4m3")
        # Optimizer hyperparameters are explicit constructor parameters
        # (previously hard-coded); defaults preserve the old behavior.
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        # Number of train_steps skipped because of non-finite loss/grads.
        self.skipped_steps = 0

        # Convert applicable layers to FP8
        self._convert_to_fp8()

        # fp32 master weights managed by AdamW, on the same device as the
        # model parameters (re-checked every train_step, see _sync_master_devices).
        self.param_map = []  # list of (model_param, master_param)
        master_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                master = param.detach().clone().float().requires_grad_(True)
                self.param_map.append((param, master))
                master_params.append(master)

        lr = getattr(config, "lr", 1e-3)
        self.optimizer = torch.optim.AdamW(master_params, lr=lr,
                                           weight_decay=self.weight_decay)

    def _sync_master_devices(self):
        """Migrate fp32 master weights (and their AdamW state) if the model
        was moved to another device after this trainer was constructed."""
        for param, master in self.param_map:
            if master.device == param.device:
                continue
            master.data = master.data.to(param.device)
            if master.grad is not None:
                master.grad = master.grad.to(param.device)
            state = self.optimizer.state.get(master)
            if state:
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(param.device)

    def _convert_to_fp8(self):
        """Replace Linear layers with FP8Linear."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                fp8_module = FP8Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    fp8_format=self.fp8_format,
                )
                # Copy weights
                fp8_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    fp8_module.bias.data.copy_(module.bias.data)

                # Replace
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, child_name, fp8_module)
                else:
                    setattr(self.model, child_name, fp8_module)

        # Re-bind MTP shared modules (weight tying): the loop above
        # replaces model.lm_head with an FP8Linear, which would leave
        # mtp_modules[i].lm_head pointing at the OLD nn.Linear (shared
        # modules are deduplicated by named_modules, so only one name is
        # visited) and silently break the tying. Point every MTP module's
        # lm_head / embed_tokens back at the main model's CURRENT modules,
        # whatever type they now are (same defensive rebinding as
        # QuantizationManager.quantize_model). Defensive getattr: the model
        # may not have MTP.
        mtp_modules = getattr(self.model, "mtp_modules", None)
        if mtp_modules is not None:
            lm_head = getattr(self.model, "lm_head", None)
            embed_tokens = getattr(self.model, "embed_tokens", None)
            for mtp in mtp_modules:
                if lm_head is not None and hasattr(mtp, "lm_head"):
                    mtp.lm_head = lm_head
                if embed_tokens is not None and hasattr(mtp, "embed_tokens"):
                    mtp.embed_tokens = embed_tokens

    @staticmethod
    def _extract_loss(output) -> torch.Tensor:
        """Defensively extract a scalar loss from model output."""
        if isinstance(output, dict):
            loss = output["loss"]
        elif isinstance(output, (tuple, list)):
            loss = output[0]
        elif hasattr(output, "loss"):
            loss = output.loss
        else:
            loss = output
        if loss.ndim > 0:
            loss = loss.mean()
        return loss

    def train_step(self, batch):
        """Single training step: forward -> zero_grad -> backward ->
        optimizer.step -> copy master weights back to the model.

        NaN guard (M-T2): if the loss or any gradient is non-finite, the
        optimizer step is SKIPPED — no NaN ever reaches the fp32 master
        weights — the stale gradients are cleared, ``self.skipped_steps``
        is incremented and a warning is emitted. The raw (possibly
        non-finite) loss value is still returned so callers can log it.
        """
        # Masters follow the model if it was moved after construction.
        self._sync_master_devices()

        # Zero gradients *before* the forward/backward so they never
        # accumulate across steps (including skipped ones).
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

        # Forward in FP8
        loss = self._extract_loss(self.model(**batch))

        if not torch.isfinite(loss):
            self.skipped_steps += 1
            warnings.warn(
                f"FP8Trainer: non-finite loss ({loss.item()}); skipping "
                f"optimizer step (skipped_steps={self.skipped_steps})")
            return loss.item()

        # Backward (E5M2-quantized gradients via hooks)
        loss.backward()

        # Move gradients onto the fp32 master weights; skip params whose
        # gradient was never produced.
        grads_finite = True
        for param, master in self.param_map:
            if param.grad is None:
                master.grad = None
                continue
            if not torch.isfinite(param.grad).all():
                grads_finite = False
            master.grad = param.grad.detach().to(torch.float32).clone()

        if not grads_finite:
            self.skipped_steps += 1
            warnings.warn(
                "FP8Trainer: non-finite gradient detected; skipping "
                f"optimizer step (skipped_steps={self.skipped_steps})")
            # Drop the poisoned grads so they cannot leak into a later step.
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            return loss.item()

        # Clip and step on master weights
        torch.nn.utils.clip_grad_norm_(
            [m for _, m in self.param_map], max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Copy master weights back to the model
        with torch.no_grad():
            for param, master in self.param_map:
                param.copy_(master.to(param.dtype))

        return loss.item()
