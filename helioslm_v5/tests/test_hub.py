"""Offline Hub round-trip tests; never use the network or require an HF token."""
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.hub import save_pretrained, load_pretrained, from_hub


def test_hub_local_roundtrip(tmp_path):
    torch.manual_seed(123)
    config = HeliosLMv5Config(size="lite")
    original = HeliosLMv5(config).eval()
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
