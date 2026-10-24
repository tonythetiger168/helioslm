"""T93 — roadmap Phase B: CSA learned compressor (v5.59).

Pins:

1. a fresh learned compressor IS mean pooling (bitwise equal outputs,
   with padding), so switching it on changes nothing until trained;
2. learned pooling weights are a per-chunk softmax: they sum to 1 over
   real tokens, pads get exactly 0, and an all-pad chunk gets 0;
3. entries equal a manual softmax-weighted pool of k/v, and causality
   still holds with non-trivial (randomised) compressor weights;
4. gradients reach the compressor; bad arguments are refused;
5. the committed artifact's stored weights re-evaluate to the recorded
   numbers (one slice), selecting every chunk gives the same accuracy as
   top-8 (selection is not the bottleneck), and the artifact is
   registered.

Run from repo root: python3 evastar/tests/test_csa_learned.py
"""
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from evastar.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from evastar.src.attention.kv_compress import (  # noqa: E402
    CompressedSparseAttention)

D, H = 32, 2


def _pair():
    torch.manual_seed(5)
    mean = CompressedSparseAttention(D, H, ratio=4, topk=3).eval()
    learned = CompressedSparseAttention(D, H, ratio=4, topk=3,
                                        compressor="learned").eval()
    missing, unexpected = learned.load_state_dict(mean.state_dict(),
                                                  strict=False)
    assert all(k.startswith("compressor_") for k in missing) and not unexpected
    return mean, learned


def _randomise(m):
    with torch.no_grad():
        for p in m.compressor_mlp.parameters():
            p.normal_(0, 0.5)
        m.compressor_pos.normal_(0, 0.5)
    return m


def test_fresh_learned_is_mean():
    mean, learned = _pair()
    x = torch.randn(2, 18, D)
    mask = torch.ones(2, 18, dtype=torch.long)
    mask[1, :5] = 0
    with torch.no_grad():
        assert torch.equal(mean(x, mask), learned(x, mask))
    print("PASS test_fresh_learned_is_mean")


def test_pool_weights_are_masked_softmax():
    _, learned = _pair()
    _randomise(learned)
    x = torch.randn(1, 12, D)
    mask = torch.ones(1, 12, dtype=torch.long)
    mask[0, 4:8] = 0          # chunk 1 entirely padded
    mask[0, 9] = 0            # one pad inside chunk 2
    with torch.no_grad():
        w = learned._pool_weights(x, mask, 1, 3, 4)[..., 0]   # (1,3,4,H)
    assert torch.allclose(w[0, 0].sum(0), torch.ones(H))
    assert torch.equal(w[0, 1], torch.zeros(4, H))
    assert torch.equal(w[0, 2, 1], torch.zeros(H))
    assert torch.allclose(w[0, 2].sum(0), torch.ones(H))
    print("PASS test_pool_weights_are_masked_softmax")


def test_entries_and_causality():
    _, learned = _pair()
    _randomise(learned)
    x = torch.randn(1, 16, D)
    with torch.no_grad():
        k = learned.qkv(x).view(1, 16, 3, H, D // H)[:, :, 1]
        logits = learned.compressor_mlp(x).view(1, 4, 4, H) \
            + learned.compressor_pos.view(1, 1, 4, H)
        w = torch.softmax(logits, dim=2).unsqueeze(-1)
        want = (k.view(1, 4, 4, H, D // H) * w).sum(2)
        got_w = learned._pool_weights(x, torch.ones(1, 16, dtype=torch.long),
                                      1, 4, 4)
        got = (k.view(1, 4, 4, H, D // H) * got_w).sum(2)
        assert torch.equal(got, want)
        base = learned(x)
        for p in (6, 13):
            x2 = x.clone()
            x2[0, p] += 2.0
            assert torch.equal(learned(x2)[0, :p], base[0, :p]), p
    print("PASS test_entries_and_causality")


def test_gradients_and_validation():
    _, learned = _pair()
    learned.train()
    learned(torch.randn(2, 16, D)).sum().backward()
    # the zero-initialised output layer receives gradient first
    assert learned.compressor_mlp[2].weight.grad.abs().sum() > 0
    assert learned.compressor_pos.grad is not None
    for bad in (dict(compressor="max"),
                dict(compressor="learned", compressor_hidden=0)):
        try:
            CompressedSparseAttention(D, H, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    print("PASS test_gradients_and_validation")


def test_committed_artifact_slice():
    import csa_learned_compressor as script
    from hca_ab_passkey import probe_accuracy

    art = json.loads((_REPO / "benchmarks" / "csa_learned_2026-10-15.json")
                     .read_text())
    p = art["params"]
    row = next(r for r in art["results"] if r["ratio"] == 4)

    class Args:
        dim, heads, hidden = p["dim"], p["heads"], p["hidden"]

    state = script._from_json(row["compressor_state"])
    acc = probe_accuracy(lambda: script.make_csa(Args, 4, 8, state), 512,
                         p["pairs"], p["dim"], p["amp"], p["boost"],
                         p["trials"], p["eval_seed"])
    assert acc == row["eval"]["L=512/k8"]["accuracy"], acc
    assert acc > 4 * art["chance"]          # the headline: well above chance
    # selection is not the bottleneck: selecting EVERY chunk gives the
    # same accuracy as top-8 (the report's reading 2 rests on this)
    acc_all = probe_accuracy(lambda: script.make_csa(Args, 4, 10_000, state),
                             512, p["pairs"], p["dim"], p["amp"], p["boost"],
                             p["trials"], p["eval_seed"])
    assert acc_all == acc, (acc_all, acc)
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of("csa_learned_2026-10-15.json", reg) == "measured"
    print(f"PASS test_committed_artifact_slice (acc={acc})")


if __name__ == "__main__":
    test_fresh_learned_is_mean()
    test_pool_weights_are_masked_softmax()
    test_entries_and_causality()
    test_gradients_and_validation()
    test_committed_artifact_slice()
