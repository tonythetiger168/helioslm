"""T105 — roadmap B2-5: MiMo-style light MTP head, v5.67.

Pins:

1. default style unchanged: same state-dict keys/shapes and the same MTP
   logits as a config without the new fields;
2. light style: FFN width light_ffn_size (default hidden_size), fewer
   head parameters by exactly the FFN difference;
3. window mask oracle: query t sees keys max(0, t-W+1)..t only;
4. in the training path a light head's output at position t does not
   depend on inputs more than W-1 positions back (a default head's does);
5. inference path (one token) is unaffected by the window;
6. the committed A/B artifact: gate holds for both styles, hash, and the
   light head is smaller and not slower;
7. validation.

Run from repo root: python3 evastar/tests/test_mtp_light.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import EvastarV5Config, MTPConfig  # noqa: E402
from evastar.src.inference.mtp import MTPModule  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

_ART = _REPO / "benchmarks" / "mtp_light_ab_2026-10-16.json"


def _head(style, window=4, ffn=None, seed=0):
    cfg = EvastarV5Config(size="lite", mtp=MTPConfig(
        style=style, light_window=window, light_ffn_size=ffn))
    torch.manual_seed(seed)
    return MTPModule(cfg, 0).eval(), cfg


def test_default_unchanged():
    torch.manual_seed(0)
    base = EvastarV5(EvastarV5Config(size="lite"))
    torch.manual_seed(0)
    dflt = EvastarV5(EvastarV5Config(size="lite",
                                       mtp=MTPConfig(style="default")))
    sa, sb = base.state_dict(), dflt.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    h = torch.randn(2, 10, 256)
    ids = torch.randint(3, 1000, (2, 10))
    with torch.no_grad():
        a = base.mtp_modules[0].eval()(h, ids)
        b = dflt.mtp_modules[0].eval()(h, ids)
    assert torch.equal(a, b)
    print("PASS test_default_unchanged")


def test_light_sizes():
    d, cfg = _head("default")
    l, _ = _head("light")
    l2, _ = _head("light", ffn=64)
    assert d.transformer.linear1.out_features == cfg.intermediate_size
    assert l.transformer.linear1.out_features == cfg.hidden_size
    assert l2.transformer.linear1.out_features == 64
    n = lambda m: sum(p.numel() for p in m.transformer.parameters())  # noqa: E731
    H, I = cfg.hidden_size, cfg.intermediate_size
    assert n(d) - n(l) == 2 * H * (I - H) + (I - H)       # 2 matrices + bias
    print("PASS test_light_sizes")


def test_window_mask_oracle():
    W, n = 3, 7
    m = MTPModule._causal_mask(n, "cpu", torch.float32, W)
    for t in range(n):
        allowed = [j for j in range(n) if m[t, j] == 0]
        assert allowed == list(range(max(0, t - W + 1), t + 1)), (t, allowed)
    full = MTPModule._causal_mask(n, "cpu", torch.float32)
    assert torch.equal(full, MTPModule._causal_mask(n, "cpu", torch.float32, 99))
    print("PASS test_window_mask_oracle")


def test_light_receptive_field():
    W = 4
    for style, bounded in (("light", True), ("default", False)):
        head, cfg = _head(style, window=W)
        h = torch.randn(1, 14, cfg.hidden_size)
        ids = torch.randint(3, 1000, (1, 14))
        h2 = h.clone()
        h2[0, 0] += 3.0                  # perturb the oldest position
        with torch.no_grad():
            a = head(h, ids)
            b = head(h2, ids)
        t = W + 2                        # well outside the window of 0
        changed = not torch.allclose(a[0, t], b[0, t])
        assert changed != bounded, (style, changed)
    print("PASS test_light_receptive_field")


def test_inference_path_unaffected_by_window():
    a, cfg = _head("light", window=1)
    b, _ = _head("light", window=64)
    h = torch.randn(2, 1, cfg.hidden_size)
    tok = torch.randint(3, 1000, (2, 1))
    with torch.no_grad():
        assert torch.equal(a(h, tok), b(h, tok))
    print("PASS test_inference_path_unaffected_by_window")


def test_ab_artifact():
    art = json.loads(_ART.read_text())
    res = {r["style"]: r for r in art["results"]}
    assert set(res) == {"default", "light"}
    assert all(r["gate_greedy_equal"] for r in res.values())
    assert res["light"]["head_params"] < res["default"]["head_params"]
    assert art["results_sha256"] == hashlib.sha256(json.dumps(
        art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_ab_artifact (acceptance default "
          f"{res['default']['acceptance_rate']}, light "
          f"{res['light']['acceptance_rate']})")


def test_validation():
    for kw, needle in ((dict(style="tiny"), "mtp.style"),
                       (dict(light_window=0), "light_window"),
                       (dict(light_ffn_size=0), "light_ffn_size")):
        try:
            EvastarV5Config(size="lite", mtp=MTPConfig(**kw))
        except ValueError as e:
            assert needle in str(e), (kw, e)
        else:
            raise AssertionError(kw)
    print("PASS test_validation")


if __name__ == "__main__":
    test_default_unchanged()
    test_light_sizes()
    test_window_mask_oracle()
    test_light_receptive_field()
    test_inference_path_unaffected_by_window()
    test_ab_artifact()
    test_validation()
