"""Offline Hub round-trip tests; never use the network or require an HF token."""
import torch

from evastar.configs.config_v5 import EvastarV5Config
from evastar.src.model_v5 import EvastarV5
from evastar.hub import save_pretrained, load_pretrained, from_hub


def test_hub_local_roundtrip(tmp_path):
    torch.manual_seed(123)
    config = EvastarV5Config(size="lite")
    original = EvastarV5(config).eval()
    save_pretrained(original, tmp_path)
    restored = load_pretrained(tmp_path).eval()
    assert restored.config.size == "lite"
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    x = torch.randint(0, config.vocab_size, (1, 4))
    with torch.inference_mode():
        torch.testing.assert_close(original(x)[0], restored(x)[0], atol=0, rtol=0)


def test_hub_remote_requires_explicit_trust():
    import pytest
    with pytest.raises(ValueError, match="trust_checkpoint"):
        from_hub("someone/untrusted-checkpoint")


def test_hub_upload_download_with_mock_backend(tmp_path, monkeypatch):
    """Exercise the real Hub API wiring without credentials or network."""
    import shutil
    from evastar import hub

    remote = tmp_path / "remote"
    class FakeApi:
        def __init__(self, token=None):
            self.token = token
        def create_repo(self, repo_id, private=None, exist_ok=False):
            assert repo_id == "test/helioslm"
            assert private is True
            assert exist_ok
        def upload_folder(self, folder_path, repo_id, commit_message):
            assert repo_id == "test/helioslm"
            shutil.copytree(folder_path, remote)
            return "mock-commit"
    class FakeHub:
        HfApi = FakeApi
        @staticmethod
        def snapshot_download(repo_id, revision=None, token=None,
                              allow_patterns=None, local_files_only=False):
            assert repo_id == "test/helioslm"
            assert revision == "pinned-test-sha"
            assert sorted(allow_patterns) == ["config.json", "pytorch_model.bin"]
            return str(remote)
    monkeypatch.setattr(hub, "_hub", lambda: FakeHub)
    model = EvastarV5(EvastarV5Config(size="lite")).eval()
    assert hub.push_to_hub(model, "test/helioslm", private=True) == "mock-commit"
    restored = hub.from_hub("test/helioslm", revision="pinned-test-sha",
                            trust_checkpoint=True).eval()
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, atol=0, rtol=0)


def test_hub_config_roundtrip_every_preset():
    """config.json must rebuild every preset exactly (nested dataclasses,
    tuples that JSON turns into lists), not just the lite default (#4)."""
    import json
    from dataclasses import asdict
    from evastar.configs import config_v5 as C
    from evastar.hub import _config_from_dict
    presets = [EvastarV5Config(size="lite"), EvastarV5Config(size="mid"),
               C.k3_block_config(), C.mimo_hybrid_config(), C.qwen_next_config()]
    for cfg in presets:
        payload = json.loads(json.dumps(asdict(cfg)))
        assert _config_from_dict(payload) == cfg
