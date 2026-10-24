"""T107 — roadmap U3: Llama 4 NoPE attention temperature, v5.69.

Pins:

1. off by default; on, every position with p + 1 < floor_scale gets a
   scale of exactly 1.0, so short contexts are bit-identical to off;
2. the scale matches Llama 4's formula (vLLM reference) exactly,
   including the boundary p = F - 1 where it first exceeds 1;
3. with a small floor scale (scaling active): cached decode == full
   forward, absorbed == expanded path, and the output differs from off;
4. validation (requires nope, floor >= 1, scale >= 0);
5. the committed A/B artifact: one set of weights, so accuracy at the
   training length is identical for every temperature (scale == 1
   there); hash.

Run from repo root: python3 evastar/tests/test_nope_temperature.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import AttentionConfig, EvastarV5Config  # noqa: E402
from evastar.src.attention.mla import MLA  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

_ART = _REPO / "benchmarks" / "nope_temperature_ab_2026-10-16.json"


def _cfg(temp, floor=8192, scale=0.1, absorb=True):
    return EvastarV5Config(size="lite", attention=AttentionConfig(
        nope=True, nope_attn_temperature=temp, nope_temp_floor_scale=floor,
        nope_temp_scale=scale, use_absorption=absorb))


def test_short_context_bit_identical():
    ids = torch.randint(3, 1000, (2, 14))
    outs = []
    for temp in (False, True):
        torch.manual_seed(0)
        m = EvastarV5(_cfg(temp, floor=64)).eval()
        with torch.no_grad():
            outs.append(m(ids)[0])
    assert torch.equal(outs[0], outs[1])
    assert not MLA(EvastarV5Config(size="lite")).nope_temp
    print("PASS test_short_context_bit_identical")


def test_formula_matches_llama4():
    F, s = 8192, 0.1
    mla = MLA(_cfg(True, floor=F, scale=s))
    pos = torch.tensor([[0, 1, F - 2, F - 1, F, 2 * F - 1, 10 * F, 100000]])
    got = mla.nope_attn_scale(pos)[0]
    for p, g in zip(pos[0].tolist(), got.tolist()):
        want = math.log(math.floor((p + 1.0) / F) + 1.0) * s + 1.0
        assert abs(g - want) < 1e-6, (p, g, want)
    assert got[:3].tolist() == [1.0, 1.0, 1.0]
    assert got[3] > 1.0
    print("PASS test_formula_matches_llama4")


def test_active_scaling_paths():
    ids = torch.randint(3, 1000, (2, 14))
    fulls = {}
    for absorb in (True, False):
        torch.manual_seed(0)
        m = EvastarV5(_cfg(True, floor=4, absorb=absorb)).eval()
        with torch.no_grad():
            full = m(ids)[0]
            lg, _, past = m(ids[:, :5], use_cache=True)
            outs = [lg]
            for t in range(5, 14):
                lg, _, past = m(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
                outs.append(lg)
        assert (torch.cat(outs, 1) - full).abs().max() < 1e-4
        fulls[absorb] = full
    assert (fulls[True] - fulls[False]).abs().max() < 1e-4
    torch.manual_seed(0)
    off = EvastarV5(_cfg(False, floor=4)).eval()
    with torch.no_grad():
        assert (off(ids)[0] - fulls[True]).abs().max() > 1e-3
    print("PASS test_active_scaling_paths")


def test_validation():
    for kw, needle in ((dict(nope_attn_temperature=True), "requires attention.nope"),
                       (dict(nope=True, nope_attn_temperature=1), "must be a bool"),
                       (dict(nope=True, nope_temp_floor_scale=0), "floor_scale"),
                       (dict(nope=True, nope_temp_scale=-1.0), "nope_temp_scale")):
        try:
            EvastarV5Config(size="lite", attention=AttentionConfig(**kw))
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = art["results"]
    train = str(art["task"]["train_seq"])
    assert len({r["acc"][train] for r in res}) == 1, \
        "temperature changed accuracy at the training length"
    assert res[0]["nope_temp_scale"] == 0.0
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(res, sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact " + json.dumps(
        {r["nope_temp_scale"]: r["acc"] for r in res}))


if __name__ == "__main__":
    test_short_context_bit_identical()
    test_formula_matches_llama4()
    test_active_scaling_paths()
    test_validation()
    test_ab_artifact()
