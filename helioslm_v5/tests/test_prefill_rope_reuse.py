"""Optimized stable prefill keeps the original token-reference arithmetic."""
import torch
import pytest

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.attention.mla import RotaryEmbedding


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_stable_prefill_matches_original_token_schedule(dtype):
    torch.manual_seed(88)
    model = HeliosLMv5(HeliosLMv5Config(size='lite')).to(dtype=dtype).eval()
    ids = torch.randint(4, 1000, (2, 13))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, use_cache=True)
        past, logits, states = None, [], []
        for i in range(ids.shape[1]):
            out, hidden, past = model._forward_impl(
                ids[:, i:i + 1], attention_mask=mask[:, :i + 1],
                past_key_values=past, use_cache=True)
            logits.append(out)
            states.append(hidden)
        assert torch.equal(actual[0], torch.cat(logits, 1))
        assert torch.equal(actual[1], torch.cat(states, 1))
        for cache_a, cache_b in zip(actual[2], past):
            assert all(torch.equal(a, b) for a, b in zip(cache_a, cache_b))


def test_contiguous_rotary_lookup_preserves_tables_and_bounds():
    rope = RotaryEmbedding(8, 4)
    for start, length in ((0, 3), (2, 5), (11, 2)):
        positions = torch.arange(start, start + length).unsqueeze(0).expand(2, -1)
        reference = rope(positions)
        actual = rope.contiguous(start, length, 2)
        assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    with pytest.raises(ValueError, match='non-negative'):
        rope(torch.tensor([[-1]]))
    with pytest.raises(ValueError, match='start >= 0'):
        rope.contiguous(-1, 2, 1)


def test_explicit_float_positions_still_fail_validation():
    model = HeliosLMv5(HeliosLMv5Config(size='lite')).half().eval()
    with torch.inference_mode(), pytest.raises((IndexError, RuntimeError)):
        model(torch.tensor([[4, 5, 6]]), position_ids=torch.arange(3).float())


def test_inconsistent_layer_cache_lengths_still_fail_validation():
    model = HeliosLMv5(HeliosLMv5Config(size='lite')).half().eval()
    with torch.inference_mode():
        past = model(torch.tensor([[4, 5, 6]]), use_cache=True)[2]
        past[1] = tuple(t[:, :, :2] for t in past[1])
        with pytest.raises(ValueError, match='cached keys are unknown'):
            model(torch.tensor([[7, 8]]), past_key_values=past)
