"""Hugging Face Hub checkpoint interchange for HeliosLM (#4).

Use explicit, trusted repositories: PyTorch pickle checkpoints are unsafe to
load from untrusted authors. No network call occurs on import.
"""
from dataclasses import asdict, fields
import json
from pathlib import Path
import tempfile

import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError("Install optional dependency: pip install huggingface_hub") from exc
    return huggingface_hub


def _serialize_config(config):
    return asdict(config)


def _config_from_dict(payload):
    """Rehydrate nested dataclasses without discarding explicit settings."""
    from dataclasses import is_dataclass
    from typing import get_type_hints
    def construct(cls, data):
        hints = get_type_hints(cls)
        kwargs = {}
        for field in fields(cls):
            if field.name not in data:
                continue
            value = data[field.name]
            typ = hints.get(field.name)
            # For Optional[Dataclass], find the dataclass in its union args.
            from typing import get_args
            nested = [t for t in (get_args(typ) or (typ,)) if isinstance(t, type) and is_dataclass(t)]
            if isinstance(value, dict) and nested:
                value = construct(nested[0], value)
            kwargs[field.name] = value
        return cls(**kwargs)
    return construct(HeliosLMv5Config, payload)


def save_pretrained(model, directory):
    """Write config JSON and tensor-only state_dict (no pickled model object)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(_serialize_config(model.config), indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), directory / "pytorch_model.bin")
    return directory


def load_pretrained(directory, map_location="cpu"):
    """Load a checkpoint from a TRUSTED local directory; strict tensor match."""
    directory = Path(directory)
    payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = _config_from_dict(payload)
    model = HeliosLMv5(config)
    weights = torch.load(directory / "pytorch_model.bin", map_location=map_location,
                         weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model


def push_to_hub(model, repo_id, *, token=None, private=None):
    """Upload config and weights; explicit opt-in network write."""
    hub = _hub()
    with tempfile.TemporaryDirectory() as folder:
        save_pretrained(model, folder)
        api = hub.HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        return api.upload_folder(folder_path=folder, repo_id=repo_id,
                                 commit_message="Upload HeliosLM config and checkpoint")


def from_hub(repo_id, *, revision=None, token=None, local_files_only=False,
             trust_checkpoint=False):
    """Download and strict-load a checkpoint.

    Explicit trust_checkpoint=True is required because a remote PyTorch
    .bin checkpoint must be treated as untrusted even with weights_only.
    Pin revision to an immutable commit SHA for reproducibility.
    """
    if not trust_checkpoint:
        raise ValueError("Set trust_checkpoint=True only for a trusted Hub repository")
    hub = _hub()
    folder = hub.snapshot_download(
        repo_id=repo_id, revision=revision, token=token,
        allow_patterns=["config.json", "pytorch_model.bin"],
        local_files_only=local_files_only)
    return load_pretrained(folder)
