"""T79 - B1-5: --mla-every sweep config plumbing + bitwise gate oracles.

The hybrid interleave rule (v5.5) puts MLA at layer 0 and at
``i % every == every - 1``; B1-5 sweeps ``every`` over {2, 4, 8} via
examples/mla_every_sweep.py. These tests pin:

1. the layer-map rule at 8 layers (the sweep's default depth — stock
   lite=2 is degenerate for the rule),
2. config plumbing (explicit HybridAttentionConfig wins over the
   size-follow default; every<2 raises),
3. the HarnessEvolver-style bitwise gate: same config + seed built and
   greedy-decoded twice must be token-identical,
4. CLI smoke: the sweep script end-to-end at minimal settings produces
   valid JSON with gate-passing arms.

Run from repo root: python3 evastar/tests/test_mla_every.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import (EvastarV5Config,  # noqa: E402
                                           HybridAttentionConfig)
from evastar.src.model_v5 import _is_full_attention_layer  # noqa: E402

_SWEEP = _REPO / "examples" / "mla_every_sweep.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location("mla_every_sweep", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(every, layers=8):
    if every is None:
        return EvastarV5Config(size="lite", num_hidden_layers=layers)
    return EvastarV5Config(
        size="lite", num_hidden_layers=layers,
        hybrid_attention=HybridAttentionConfig(enabled=True,
                                               full_attention_every=every))


def test_layer_map_rule():
    expected = {
        2: [True, True, False, True, False, True, False, True],
        4: [True, False, False, True, False, False, False, True],
        8: [True, False, False, False, False, False, False, True],
    }
    for every, want in expected.items():
        cfg = _cfg(every)
        got = [_is_full_attention_layer(cfg, i) for i in range(8)]
        assert got == want, f"every={every}: {got} != {want}"
    dense = _cfg(None)
    assert all(_is_full_attention_layer(dense, i) for i in range(8))
    # layer 0 is always MLA regardless of the rule
    cfg = _cfg(8)
    assert _is_full_attention_layer(cfg, 0) is True
    print("PASS test_layer_map_rule")


def test_config_validation():
    try:
        _cfg(1)
        raise AssertionError("full_attention_every=1 must raise ValueError")
    except ValueError:
        pass
    # explicit hybrid config wins over the lite default (hybrid off)
    cfg = _cfg(2)
    assert cfg.hybrid_attention.enabled is True
    assert cfg.hybrid_attention.full_attention_every == 2
    # size-follow fields resolved (lite: 4 heads x 32 dim)
    assert cfg.hybrid_attention.linear_num_heads == 4
    assert cfg.hybrid_attention.linear_head_dim == 32
    print("PASS test_config_validation")


def test_bitwise_gate_double_run():
    sweep = _load_sweep()
    cfg = sweep.arm_config(2, layers=4)
    prompts = sweep.fixed_prompts(cfg, seed=5, n_prompts=1, prompt_len=6)
    outs1 = sweep.build_and_generate(cfg, seed=5, prompts=prompts,
                                     decode_tokens=2)
    outs2 = sweep.build_and_generate(cfg, seed=5, prompts=prompts,
                                     decode_tokens=2)
    assert outs1 == outs2, "bitwise gate: same config+seed diverged"
    # different seed -> (overwhelmingly) different weights -> different
    # greedy continuation; this guards against a vacuous gate
    outs3 = sweep.build_and_generate(cfg, seed=6, prompts=prompts,
                                     decode_tokens=2)
    assert outs1 != outs3, "gate is vacuous: different seeds agree"
    print("PASS test_bitwise_gate_double_run")


def test_sweep_cli_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sweep.json"
        r = subprocess.run(
            [sys.executable, str(_SWEEP),
             "--steps", "1", "--layers", "4", "--mla-every", "2",
             "--prompts", "1", "--decode-tokens", "2",
             "--cache-seq", "64", "--output", str(out)],
            capture_output=True, text=True, timeout=240,
            cwd=str(_REPO))
        assert r.returncode == 0, f"sweep CLI failed:\n{r.stderr[-800:]}"
        payload = json.loads(out.read_text())
    assert payload["experiment"].startswith("B1-5")
    arms = payload["results"]["arms"]
    assert [a["label"] for a in arms] == ["dense_mla", "mla_every=2"]
    for a in arms:
        assert a["bitwise_gate"]["pass"] is True
        assert len(a["bitwise_gate"]["outputs_sha256"]) == 64
        assert "training" in a and a["training"]["steps"] == 1
        assert a["cache_bytes"]["helioslm_bytes"] > 0
    # steps=1 winner is init-noise, not a quality signal: it flips with
    # torch build numerics (sandbox torch 2.8 -> mla_every=2, CI torch
    # -> dense_mla). This smoke test checks CLI plumbing only, so the
    # recommendation must name a gate-passing arm, not a fixed one.
    labels = {a["label"] for a in arms}
    assert payload["results"]["recommendation"]["winner"] in labels
    assert len(payload["results_sha256"]) == 64
    print("PASS test_sweep_cli_smoke")


def main():
    test_layer_map_rule()
    test_config_validation()
    test_bitwise_gate_double_run()
    test_sweep_cli_smoke()
    print("T79 PASS: 4/4")


if __name__ == "__main__":
    main()
