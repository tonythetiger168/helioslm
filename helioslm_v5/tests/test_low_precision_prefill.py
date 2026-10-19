"""Low-precision prefill/decode parity, masks, cache continuation and gradients."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_masked_chunked_prefill(dtype):
    torch.manual_seed(71)
    model = HeliosLMv5(HeliosLMv5Config(size='lite')).to(dtype=dtype).eval()
    ids = torch.randint(4, 1000, (2, 11))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    with torch.inference_mode():
        full, hidden, no_cache = model(ids, attention_mask=mask)
        cached, states, past = [], [], None
        for start, stop in ((0, 3), (3, 7), (7, 11)):
            out, state, past = model(ids[:, start:stop], attention_mask=mask[:, :stop],
                                      past_key_values=past, use_cache=True)
            cached.append(out)
            states.append(state)
        assert torch.equal(full, torch.cat(cached, 1))
        assert torch.equal(hidden, torch.cat(states, 1))
        assert full.dtype == hidden.dtype == dtype
        assert no_cache is None
        assert all(t.dtype == dtype and t.shape[2] == 11 for layer in past for t in layer)
        # An omitted prefix mask means the cached prefix consists of real tokens.
        prefix = model(ids[:, :4], use_cache=True)[2]
        short_mask = torch.ones((2, 7), dtype=torch.long)
        short = model(ids[:, 4:], attention_mask=short_mask,
                      past_key_values=prefix)[0]
        explicit = model(ids[:, 4:], attention_mask=torch.ones_like(ids),
                         past_key_values=prefix)[0]
        assert torch.equal(short, explicit)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_explicit_positions_and_packed_layout(dtype):
    torch.manual_seed(72)
    cfg = HeliosLMv5Config(size='lite')
    model = HeliosLMv5(cfg).to(dtype=dtype).eval()
    ids = torch.randint(4, 1000, (1, 6))
    with torch.no_grad():
        default = model(ids)[0]
        assert torch.equal(default, model(ids, position_ids=torch.arange(6))[0])
        # Packed resets must retain the parallel, document-aware mask path.
        packed = torch.tensor([[0, 1, 2, 0, 1, 2]])
        stable = model(ids, position_ids=packed)[0]
        cfg.stable_low_precision_prefill = False
        parallel = model(ids, position_ids=packed)[0]
        assert torch.equal(stable, parallel)


def test_training_and_fp32_keep_existing_forward():
    torch.manual_seed(73)
    cfg = HeliosLMv5Config(size='lite')
    model = HeliosLMv5(cfg).eval()
    ids = torch.randint(4, 1000, (1, 6))
    with torch.no_grad():
        baseline = model._forward_impl(ids)[0]
        assert torch.equal(model(ids)[0], baseline)
    # Training and eval-with-grad preserve ordinary parallel autograd.
    model.bfloat16().train()
    logits = model(ids)[0]
    logits.float().square().mean().backward()
    assert torch.isfinite(model.embed_tokens.weight.grad).all()
    assert model.embed_tokens.weight.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    model(ids)[0].float().square().mean().backward()
    assert model.lm_head.weight.grad.abs().max() > 0


def test_stable_prefill_config_validation():
    with pytest.raises(ValueError, match='stable_low_precision_prefill'):
        HeliosLMv5Config(size='lite', stable_low_precision_prefill='yes')
