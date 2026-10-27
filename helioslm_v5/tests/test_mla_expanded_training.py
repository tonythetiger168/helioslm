"""T135 — MLA training forward on the expanded SDPA path, v5.139.

On an RTX 4060 the absorbed path's eager [B, H, T, T] scores made MLA
training 2.1x slower and 40% heavier at seq 1024 than the expanded path
with the same weights. ``attention.expanded_training`` (default on) routes
cache-free training forwards of absorbed-mode layers through the expanded
path.

Pins:

1. loss and every gradient match the absorbed training forward
   (expanded_training=False) across attention options: plain, output
   gate, soft cap, learnable sink, sliding window + global layers, K=V,
   padding mask, packed position_ids;
2. the expanded path is the one taken in training, and the absorbed one
   in eval, with a cache, and with fp8 caches / sparse top-k;
3. KV-sharing models still train (sources build their latent cache, so
   they stay absorbed) and match.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_mla_expanded_training.py
"""
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import AttentionConfig, HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.attention.mla import MLA  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

OPTIONS = {
    "plain": {},
    "output_gate": {"output_gate": True},
    "soft_cap": {"logit_soft_cap": 20.0},
    "sink": {"learnable_sink": True},
    "window": {"sliding_window": 8, "global_attention_every": 2},
    "k_eq_v": {"global_k_eq_v": True, "v_head_dim": 24},
}


def _pair(attn, layers=2, **top):
    models = []
    for flag in (False, True):
        torch.manual_seed(0)
        cfg = HeliosLMv5Config(size="lite", num_hidden_layers=layers,
                              attention=AttentionConfig(expanded_training=flag, **attn),
                              **top)
        models.append(HeliosLMv5(cfg).train())
    models[1].load_state_dict(models[0].state_dict())
    return models


def _loss_and_grads(model, ids, **kw):
    model.zero_grad(set_to_none=True)
    logits = model(ids, **kw)[0]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                           if p.grad is not None}


def _assert_match(ref, new):
    (l0, g0), (l1, g1) = ref, new
    assert torch.allclose(l0, l1, atol=1e-5, rtol=1e-5), (l0, l1)
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.allclose(g0[n], g1[n], atol=1e-5, rtol=1e-4), n


@pytest.mark.parametrize("name", list(OPTIONS))
def test_training_matches_absorbed(name):
    absorbed, expanded = _pair(OPTIONS[name])
    ids = torch.randint(3, absorbed.config.vocab_size, (2, 24),
                        generator=torch.Generator().manual_seed(1))
    _assert_match(_loss_and_grads(absorbed, ids), _loss_and_grads(expanded, ids))


def test_padding_and_packed_positions_match():
    absorbed, expanded = _pair({})
    ids = torch.randint(3, absorbed.config.vocab_size, (2, 20),
                        generator=torch.Generator().manual_seed(2))
    mask = torch.ones(2, 20, dtype=torch.long)
    mask[1, :5] = 0
    _assert_match(_loss_and_grads(absorbed, ids, attention_mask=mask),
                  _loss_and_grads(expanded, ids, attention_mask=mask))
    pos = torch.cat([torch.arange(12), torch.arange(8)]).expand(2, -1)
    _assert_match(_loss_and_grads(absorbed, ids, position_ids=pos),
                  _loss_and_grads(expanded, ids, position_ids=pos))


def _paths(model, run):
    taken = []
    for m in model.modules():
        if isinstance(m, MLA):
            for name in ("_forward_absorbed", "_forward_expanded"):
                fn = getattr(m, name)
                setattr(m, name, (lambda f, k: lambda *a, **kw: (taken.append(k), f(*a, **kw))[1])(fn, name))
    with torch.no_grad():
        run()
    return set(taken)


def test_path_selection():
    _, model = _pair({})
    ids = torch.randint(3, model.config.vocab_size, (1, 10))
    assert _paths(model, lambda: model(ids)) == {"_forward_expanded"}
    model.eval()
    assert _paths(model, lambda: model(ids)) == {"_forward_absorbed"}
    model.train()
    assert _paths(model, lambda: model(ids, use_cache=True)) == {"_forward_absorbed"}
    for attn in ({"kv_cache_dtype": "fp8"}, {"sparse_top_k": 4}):
        _, m = _pair(attn)
        assert _paths(m, lambda: m(ids)) == {"_forward_absorbed"}, attn


def test_kv_sharing_still_trains_and_matches():
    absorbed, expanded = _pair({"kv_share_last_n": 1}, layers=3)
    ids = torch.randint(3, absorbed.config.vocab_size, (2, 16),
                        generator=torch.Generator().manual_seed(3))
    _assert_match(_loss_and_grads(absorbed, ids), _loss_and_grads(expanded, ids))
