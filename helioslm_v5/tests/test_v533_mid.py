"""T32 - v5.33 P1: "mid" preset acceptance oracles.

The mid rung exists for one recorded purpose: separate toy artifacts
from scale-invariant findings. These oracles pin the CONTRACT the GPU
training run must satisfy before mid claims any result:
1. Config + model instantiate; parameter count inside the declared
   150-400M band; forward + loss work.
2. BPE pairing: HeliosBPE (v5.33 P0) vocab fits inside the mid vocab
   budget and encodes protocol strings to in-range ids.
3. Inference economics: mid stays CPU-runnable at bf16 (~720MB weights),
   matching the deployment promise in README.
Run from repo root: python3 helioslm_v5/tests/test_v533_mid.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import HeliosBPE, build_corpus

PARAM_BAND = (150e6, 400e6)


def test_mid_instantiates_and_forward():
    cfg = HeliosLMv5Config(size="mid")
    model = HeliosLMv5(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert PARAM_BAND[0] <= n <= PARAM_BAND[1], f"{n/1e6:.1f}M outside band"
    ids = torch.randint(0, 32000, (1, 32))
    attn = torch.ones((1, 32), dtype=torch.long)
    logits, _, _ = model(ids, attention_mask=attn)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    print(f"PASS test_mid_instantiates_and_forward ({n/1e6:.1f}M params, "
          f"loss {float(loss.detach()):.2f})")


def test_mid_bpe_pairing():
    cfg = HeliosLMv5Config(size="mid")
    tok = HeliosBPE.train([build_corpus()[:60000]], vocab_size=16000)
    assert len(tok.vocab) <= cfg.vocab_size, \
        f"BPE vocab {len(tok.vocab)} exceeds mid vocab_size {cfg.vocab_size}"
    s = '@@tool@@{"calls":[{"name":"calc","args":{"expr":"3*4"}}]}@@end@@'
    ids = tok.encode(s)
    assert ids and max(ids) < cfg.vocab_size
    assert tok.decode(ids) == s
    print(f"PASS test_mid_bpe_pairing (vocab {len(tok.vocab)} <= "
          f"{cfg.vocab_size})")


def test_mid_inference_economics():
    # bf16 weights ~2 bytes/param; deployment promise: CPU-runnable
    cfg = HeliosLMv5Config(size="mid")
    model = HeliosLMv5(cfg)
    n = sum(p.numel() for p in model.parameters())
    bf16_mb = n * 2 / 1e6
    assert bf16_mb < 1000, f"bf16 weights {bf16_mb:.0f}MB exceed 1GB promise"
    print(f"PASS test_mid_inference_economics (bf16 weights "
          f"{bf16_mb:.0f}MB, CPU-runnable)")


def test_unknown_size_rejected():
    from contextlib import contextmanager

    @contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__}")

    with raises(ValueError):
        HeliosLMv5Config(size="nano")
    print("PASS test_unknown_size_rejected")


if __name__ == "__main__":
    test_mid_instantiates_and_forward()
    test_mid_bpe_pairing()
    test_mid_inference_economics()
    test_unknown_size_rejected()
    print("\n4/4 mid-preset tests passed")
