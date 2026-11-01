"""LoRA for small domain SFT runs (v5.217).

Plain-torch LoRA (no ``peft`` dependency, so CI and pods need nothing
new): each targeted ``nn.Linear`` becomes ``W x + (alpha / r) B A x``
with A ``[r, in]`` (Kaiming-uniform) and B ``[out, r]`` (zeros, so step 0
equals the base model); every other parameter is frozen.

* :func:`apply_lora` wraps the ``nn.Linear`` modules whose dotted name has
  a component in ``targets`` (default: the attention and MLP projections
  ``q_proj k_proj v_proj o_proj gate_proj up_proj down_proj``, which
  Gemma, Llama and Qwen checkpoints all use; a wrapper such as
  ``q_proj.linear`` matches too) and whose name contains none of
  ``exclude`` (default ``vision``, ``audio``: the encoders stay frozen).
* :func:`merge_lora` folds ``B A`` into the base weight and puts plain
  ``nn.Linear`` modules back, so the merged model saves with
  ``save_pretrained`` and packages with ``package_hf.py`` like any other.
* :func:`save_adapter` writes ``adapter_config.json`` and
  ``adapter_model.safetensors`` (``.bin`` without safetensors) with peft's
  key names (``base_model.model.<module>.lora_A.weight``), so
  ``peft.PeftModel.from_pretrained`` can also load the adapter.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
DEFAULT_EXCLUDE = ("vision", "audio")


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(dropout(x)))``; ``base`` stays frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=dev,
                                dtype=torch.float32)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=dev,
                                dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):          # code that reads .weight (tied heads, dtype probes)
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        h = self.dropout(x).to(self.lora_A.weight.dtype)
        return out + (self.lora_B(self.lora_A(h)) * self.scale).to(out.dtype)

    def delta(self):
        """``scale * B @ A`` in fp32, shaped like the base weight."""
        return (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scale

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_((self.base.weight.float() + self.delta()).to(lin.weight.dtype))
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def _set(model, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


def apply_lora(model, r=16, alpha=32.0, dropout=0.05, targets=DEFAULT_TARGETS,
               exclude=DEFAULT_EXCLUDE):
    """Freeze ``model`` and wrap the targeted linears in place. Returns
    {"modules": [names], "trainable": n, "frozen": n}; ValueError when no
    module matches (a wrong ``targets`` list must not train nothing)."""
    targets, exclude = set(targets), tuple(exclude or ())
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear)
             and targets & set(n.split("."))
             and not any(e in n for e in exclude)
             and ".lora_" not in n]
    if not names:
        raise ValueError(f"no nn.Linear matches LoRA targets {sorted(targets)}")
    for n in names:
        _set(model, n, LoRALinear(model.get_submodule(n), r, alpha, dropout))
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    model._helios_lora = {"r": r, "alpha": alpha, "dropout": dropout,
                          "targets": sorted(targets), "modules": names}
    return {"modules": names, "trainable": tr, "frozen": fr}


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def merge_lora(model):
    """Fold every adapter into its base weight (in place); returns the
    number of merged modules."""
    mods = lora_modules(model)
    for n, m in mods:
        _set(model, n, m.merged())
    if hasattr(model, "_helios_lora"):
        del model._helios_lora
    return len(mods)


def lora_state_dict(model):
    """Adapter weights with peft key names."""
    out = {}
    for n, m in lora_modules(model):
        out[f"base_model.model.{n}.lora_A.weight"] = m.lora_A.weight.detach().cpu().contiguous()
        out[f"base_model.model.{n}.lora_B.weight"] = m.lora_B.weight.detach().cpu().contiguous()
    return out


def save_adapter(model, out_dir, base_model_name=""):
    """Write a peft-format adapter; returns the weights file path."""
    cfg = getattr(model, "_helios_lora", None)
    if cfg is None:
        raise ValueError("model has no LoRA adapters")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaf = sorted({n.split(".")[-1] for n in cfg["modules"]})
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": str(base_model_name),
        "r": cfg["r"], "lora_alpha": cfg["alpha"], "lora_dropout": cfg["dropout"],
        "target_modules": leaf, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True}, indent=2))
    sd = lora_state_dict(model)
    try:
        from safetensors.torch import save_file
        path = out / "adapter_model.safetensors"
        save_file(sd, str(path))
    except ImportError:
        path = out / "adapter_model.bin"
        torch.save(sd, path)
    return path


def load_adapter_weights(model, path):
    """Load weights written by :func:`save_adapter` into a model that
    already has matching adapters (apply_lora with the same targets)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    mods = dict(lora_modules(model))
    seen = 0
    for k, v in sd.items():
        name, _, kind = k[len("base_model.model."):].rpartition(".lora_")
        part = kind.split(".")[0]
        getattr(mods[name], f"lora_{part}").weight.data.copy_(v)
        seen += 1
    if seen != 2 * len(mods):
        raise ValueError(f"adapter has {seen} tensors, model expects {2 * len(mods)}")
    return seen
