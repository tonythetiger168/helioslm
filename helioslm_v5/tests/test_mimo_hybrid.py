"""T96 — roadmap B2-1 + B2-2: MiMo-style sliding/global interleave and
learnable attention sink (v5.62).

Pins:

1. defaults untouched: no ``attn_sink`` parameter, no global/sliding
   split, and validation refuses bad settings;
2. the interleave rule (layer i global iff i == 0 or i % N == N - 1)
   and its effect: a global layer sees a token far outside the window, a
   sliding layer does not;
3. sink math: weights equal the softmax over [scores, sink] with the sink
   column dropped (rows sum to < 1); a very negative sink recovers the
   sink-free output; the expanded and absorbed paths agree with the sink
   on; fully-masked rows stay finite;
4. the mimo_hybrid preset: layout, cached decode == full forward, and
   gradients reaching the sink logits;
5. the committed recall artifact: arm layouts and attendable-key counts
   recomputed from the experiment's own configs, the results hash, and
   the structural negative control (all-SWA at chance).

Run from repo root: python3 helioslm_v5/tests/test_mimo_hybrid.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    HeliosLMv5Config, mimo_hybrid_config)
from helioslm_v5.src.attention.mla import MLA  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402


def _expect_value_error(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def test_defaults_and_validation():
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    names = [n for n, _ in model.named_parameters()]
    assert not any("attn_sink" in n for n in names)
    assert not any(layer.attention.is_global_layer for layer in model.layers)

    def bad(**kw):
        c = HeliosLMv5Config(size="lite")
        for k, v in kw.items():
            setattr(c.attention, k, v)
        c._validate()
    _expect_value_error(lambda: bad(global_attention_every=6),
                        "requires attention.sliding_window")
    _expect_value_error(lambda: bad(sliding_window=8, global_attention_every=1),
                        "must be an int >= 2")
    _expect_value_error(lambda: bad(learnable_sink=1),
                        "learnable_sink must be a bool")
    print("PASS test_defaults_and_validation")


def test_interleave_rule_and_window_effect():
    cfg = HeliosLMv5Config(size="lite", num_hidden_layers=12)
    cfg.attention.sliding_window = 4
    cfg.attention.global_attention_every = 6
    cfg._validate()
    model = HeliosLMv5(cfg)
    got = "".join("G" if layer.attention.is_global_layer else "S"
                  for layer in model.layers)
    assert got == "GSSSSGSSSSSG", got

    torch.manual_seed(3)
    x = torch.randn(1, 16, cfg.hidden_size)
    x2 = x.clone()
    x2[0, 0] += 5.0                       # far outside a 4-token window
    for idx, sees_far in ((0, True), (1, False)):
        mla = MLA(cfg, layer_idx=idx).eval()
        with torch.no_grad():
            a, _ = mla(x)
            b, _ = mla(x2)
        changed = not torch.allclose(a[0, -1], b[0, -1])
        assert changed == sees_far, (idx, changed)
    print("PASS test_interleave_rule_and_window_effect")


def _mla_pair(learnable_sink, absorbed=True, window=None):
    cfg = HeliosLMv5Config(size="lite")
    cfg.attention.use_absorption = absorbed
    cfg.attention.sliding_window = window
    cfg.attention.learnable_sink = learnable_sink
    torch.manual_seed(11)
    return MLA(cfg).eval()


def test_sink_math():
    plain = _mla_pair(False)
    sinked = _mla_pair(True)
    sinked.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        sinked.attn_sink.copy_(torch.tensor([0.5, -1.0, 2.0, 0.0]))
    scores = torch.randn(2, 4, 5, 7)
    probs = sinked._softmax_with_sink(scores)
    full = F.softmax(torch.cat(
        [scores, sinked.attn_sink.view(1, 4, 1, 1).expand(2, 4, 5, 1)], -1), -1)
    assert torch.allclose(probs, full[..., :-1])
    assert (probs.sum(-1) < 1).all()

    x = torch.randn(2, 9, 256)
    with torch.no_grad():
        sinked.attn_sink.fill_(-1e4)       # a sink nobody attends to
        a, _ = plain(x)
        b, _ = sinked(x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()

    masked = torch.full((1, 4, 2, 3), float("-inf"))
    assert torch.equal(sinked._softmax_with_sink(masked),
                       torch.zeros(1, 4, 2, 3))
    print("PASS test_sink_math")


def test_sink_paths_agree():
    mla = _mla_pair(True, absorbed=True, window=3)
    with torch.no_grad():
        mla.attn_sink.copy_(torch.tensor([1.0, -0.5, 0.3, 2.0]))
    x = torch.randn(2, 8, 256)
    steps = [torch.randn(2, 1, 256) for _ in range(3)]

    def run(absorbed):
        mla.use_absorption = absorbed
        with torch.no_grad():
            out, past = mla(x, use_cache=True)
            outs = [out]
            for s in steps:
                o, past = mla(s, past_key_value=past, use_cache=True)
                outs.append(o)
        return torch.cat(outs, 1)

    diff = (run(True) - run(False)).abs().max().item()
    assert diff < 1e-5, diff
    print(f"PASS test_sink_paths_agree (max|diff|={diff:.1e})")


def test_mimo_preset():
    torch.manual_seed(0)
    cfg = mimo_hybrid_config(window=4)
    model = HeliosLMv5(cfg).eval()
    layout = "".join("G" if layer.attention.is_global_layer else "S"
                     for layer in model.layers)
    assert layout == "GSSSSG", layout
    assert all(layer.attention.attn_sink is not None for layer in model.layers)
    ids = torch.randint(3, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        full, _, _ = model(ids)
        logits, _, past = model(ids[:, :12], use_cache=True)
        outs = [logits]
        for t in range(12, 20):
            logits, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                    use_cache=True)
            outs.append(logits)
    diff = (torch.cat(outs, 1) - full).abs().max().item()
    assert diff < 1e-4, diff
    model.train()
    model(ids)[0].logsumexp(-1).mean().backward()
    assert all(layer.attention.attn_sink.grad is not None
               and layer.attention.attn_sink.grad.abs().sum() > 0
               for layer in model.layers)
    assert mimo_hybrid_config().attention.sliding_window == 128
    print(f"PASS test_mimo_preset (decode vs full max|diff|={diff:.1e})")


def test_recall_artifact():
    import hashlib
    import json
    sys.path.insert(0, str(_REPO / "examples"))
    import mimo_hybrid_recall as exp
    art = json.loads((_REPO / "benchmarks"
                      / "mimo_hybrid_2026-10-15.json").read_text())
    res = {r["arm"]: r for r in art["results"]}
    assert list(res) == list(exp.ARMS)
    assert art["results_sha256"] == hashlib.sha256(json.dumps(
        art["results"], sort_keys=True).encode()).hexdigest()
    for arm, r in res.items():
        cfg = exp.arm_config(arm)
        assert r["attendable_keys_last_query"] == exp.attendable_keys(cfg)
        model = HeliosLMv5(cfg)
        layout = "".join("G" if layer.attention.sliding_window is None
                         else "S" for layer in model.layers)
        assert r["layout"] == layout, (arm, layout)
    chance = art["task"]["chance"]
    assert abs(res["all_swa"]["query_acc"] - chance) < 0.02
    assert res["hybrid_5to1"]["query_acc"] > 5 * chance
    assert res["dense"]["query_acc"] > 5 * chance
    print("PASS test_recall_artifact")


if __name__ == "__main__":
    test_defaults_and_validation()
    test_interleave_rule_and_window_effect()
    test_sink_math()
    test_sink_paths_agree()
    test_mimo_preset()
    test_recall_artifact()
