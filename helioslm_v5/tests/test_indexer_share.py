"""T76 - B1-1: IndexShare shared lightning indexer (indexer_share_every).

GLM-5.2/5.3-style sharing: one learned lightning indexer serves every
``indexer_share_every`` MLA layers instead of one indexer per layer.
Default 1 is the per-layer behaviour (bit-identical). Followers hold the
leader's indexer as an UNREGISTERED reference, so parameters and
state-dict keys are not duplicated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from helioslm_v5.configs.config_v5 import (
    AttentionConfig,
    HeliosLMv5Config,
)


def _make_model(share=1, top_k=8, indexer="learned", layers=4, seed=101):
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(seed)
    cfg = HeliosLMv5Config(size="lite")
    cfg.num_hidden_layers = layers
    cfg.attention.sparse_top_k = top_k
    cfg.attention.sparse_indexer = indexer
    cfg.attention.indexer_share_every = share
    return HeliosLMv5(cfg).eval(), cfg


def _single_indexer_params(cfg):
    a = cfg.attention
    return (cfg.hidden_size * a.indexer_num_heads * a.indexer_head_dim
            + a.kv_latent_dim * a.indexer_num_heads * a.indexer_head_dim
            + a.indexer_num_heads)


def test_config_validation():
    # Non-positive share factors fail loudly at construction.
    for bad in (0, -2):
        try:
            HeliosLMv5Config(size="lite",
                             attention=AttentionConfig(
                                 indexer_share_every=bad))
            raise AssertionError(f"share={bad} must raise")
        except ValueError:
            pass
    # Sharing requires the learned indexer: head_mean has no parameters.
    try:
        HeliosLMv5Config(size="lite",
                         attention=AttentionConfig(
                             sparse_indexer="head_mean",
                             indexer_share_every=2))
        raise AssertionError("share>1 with head_mean must raise")
    except ValueError:
        pass
    # The pre-existing learned-requires-top-k rule still fires.
    try:
        HeliosLMv5Config(size="lite",
                         attention=AttentionConfig(
                             sparse_indexer="learned",
                             indexer_share_every=2))
        raise AssertionError("learned without sparse_top_k must raise")
    except ValueError:
        pass
    print("PASS test_config_validation")


def test_default_per_layer_unchanged():
    """share=1 (default): every MLA layer owns its indexer — the
    pre-IndexShare behaviour, guarded as a regression."""
    model, cfg = _make_model(share=1, layers=4)
    indexers = [layer.attention.indexer for layer in model.layers]
    assert all(ix is not None for ix in indexers)
    assert len({id(ix) for ix in indexers}) == 4, \
        "default must stay one indexer per MLA layer"
    for layer in model.layers:
        assert layer.attention._effective_indexer() is layer.attention.indexer
    keys = model.state_dict().keys()
    for i in range(4):
        assert any(k.startswith(f"layers.{i}.attention.indexer.")
                   for k in keys), f"layer {i} must register its indexer"
    single = _single_indexer_params(cfg)
    total = sum(p.numel() for p in model.parameters())
    model2, _ = _make_model(share=1, layers=4)
    total2 = sum(p.numel() for p in model2.parameters())
    assert total == total2
    print("PASS test_default_per_layer_unchanged")


def test_share_groups_and_param_savings():
    """share=2 over 4 MLA layers: 2 groups, followers own nothing,
    indexer parameter total halves."""
    model, cfg = _make_model(share=2, layers=4)
    attns = [layer.attention for layer in model.layers]

    # Group layout: leaders 0 and 2 own; followers 1 and 3 reference.
    assert attns[0].indexer is not None and attns[2].indexer is not None
    assert attns[1].indexer is None and attns[3].indexer is None
    assert attns[1]._effective_indexer() is attns[0].indexer
    assert attns[3]._effective_indexer() is attns[2].indexer
    assert attns[0].indexer is not attns[2].indexer, \
        "separate groups must not share with each other"

    # No duplicated registration: followers contribute no indexer keys
    # and parameters() counts each shared indexer once.
    keys = model.state_dict().keys()
    assert any(k.startswith("layers.0.attention.indexer.") for k in keys)
    assert any(k.startswith("layers.2.attention.indexer.") for k in keys)
    assert not any(k.startswith("layers.1.attention.indexer.") for k in keys)
    assert not any(k.startswith("layers.3.attention.indexer.") for k in keys)

    single = _single_indexer_params(cfg)
    ref, _ = _make_model(share=1, layers=4)
    saved = sum(p.numel() for p in ref.parameters()) \
        - sum(p.numel() for p in model.parameters())
    assert saved == 2 * single, \
        f"share=2 over 4 layers must save exactly 2 indexers, got {saved}"

    # Stats tell the truth: followers report the shared scorer.
    s0 = attns[0].get_sparse_attention_stats(64)
    s1 = attns[1].get_sparse_attention_stats(64)
    assert s0["indexer_shared"] is False and s1["indexer_shared"] is True
    assert s0["indexer_params"] == s1["indexer_params"] == single
    print("PASS test_share_groups_and_param_savings")


def test_shared_sparse_decode_deterministic():
    """A shared-indexer model runs the sparse top-k decode path end to
    end and is bit-deterministic across identical runs."""
    torch.manual_seed(7)
    model, cfg = _make_model(share=2, layers=4, top_k=4)
    ids = torch.randint(3, cfg.vocab_size, (1, 6))

    def greedy():
        with torch.no_grad():
            logits, _, past = model(ids, use_cache=True)
            out = []
            for _ in range(8):  # decode steps: kv_len 7..14 > top_k=4
                nxt = logits[0, -1].argmax().view(1, 1)
                out.append(nxt.item())
                logits, _, past = model(nxt, past_key_values=past,
                                        use_cache=True)
        return out, past

    r1, p1 = greedy()
    r2, _ = greedy()
    assert r1 == r2, f"shared-indexer decode must be deterministic: {r1} vs {r2}"
    # Sanity: the sparse path actually engaged (cache grew past top_k).
    assert p1[0][0].shape[2] == 6 + 8
    print("PASS test_shared_sparse_decode_deterministic")


def test_shared_distill_trains_leader():
    """Distilling THROUGH a follower trains the leader's shared indexer."""
    model, cfg = _make_model(share=2, layers=4)
    leader = model.layers[0].attention
    follower = model.layers[1].attention
    h = torch.randn(1, 5, cfg.hidden_size)

    loss = follower.indexer_distill_loss(h)
    assert torch.isfinite(loss)
    loss.backward()
    assert leader.indexer.q_proj.weight.grad is not None, \
        "follower distill must route gradients to the leader's indexer"
    assert follower.indexer is None  # owns nothing, ever
    print("PASS test_shared_distill_trains_leader")


def test_shared_checkpoint_roundtrip():
    """state_dict of a shared model loads back into an identical build
    (leaders own the keys; followers add none)."""
    import io
    model, cfg = _make_model(share=2, layers=4)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    model2, _ = _make_model(share=2, layers=4, seed=999)
    model2.load_state_dict(torch.load(buf, weights_only=True))
    model2.eval()
    ids = torch.randint(3, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        a, _, _ = model(ids)
        b, _, _ = model2(ids)
    assert torch.equal(a, b), "roundtrip must be bit-identical"
    print("PASS test_shared_checkpoint_roundtrip")


if __name__ == "__main__":
    test_config_validation()
    test_default_per_layer_unchanged()
    test_share_groups_and_param_savings()
    test_shared_sparse_decode_deterministic()
    test_shared_distill_trains_leader()
    test_shared_checkpoint_roundtrip()
    print("\n6/6 IndexShare (B1-1) tests passed")
