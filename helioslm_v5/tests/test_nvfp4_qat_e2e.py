"""T81 - B1-3: NVFP4 QAT end-to-end smoke oracles.

examples/nvfp4_qat_smoke.py trains lite models from scratch — fp32
reference, RTN-PTQ arm, and an NVFP4 QAT arm that bakes fake-quant
weights into plain nn.Linear modules for deploy. These tests pin:

1. RTN deploy helper: every Linear lands on the NVFP4 grid (the qdq is
   a fixed point — re-quantizing a deployed weight changes nothing),
2. ste_weight / bake parity: a wrapped model and its baked artifact
   produce bitwise-identical logits (the bake drops no information),
   including through MLA's absorbed kv_b_proj weight-slice path,
3. bitwise gate + vacuity guard: same seed -> identical result,
   different seed -> different result,
4. CLI smoke: the script end-to-end at minimal settings produces valid
   JSON with all three arms, a gate verdict, and results_sha256.

Run from repo root: python3 helioslm_v5/tests/test_nvfp4_qat_e2e.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.quantization.qat import (  # noqa: E402
    FakeQuantLinear, _qdq_nvfp4, apply_qat)

_SMOKE = _REPO / "examples" / "nvfp4_qat_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("nvfp4_qat_smoke",
                                                  _SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rtn_deploy_grid_fixed_point():
    smoke = _load_smoke()
    torch.manual_seed(81)
    model = nn.Sequential(nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 16))
    n = smoke.rtn_deploy(model)
    assert n == 2
    for m in model.modules():
        if isinstance(m, nn.Linear):
            redq = _qdq_nvfp4(m.weight.data, 16)
            diff = (redq - m.weight.data).abs().max().item()
            assert diff < 1e-6, \
                f"deployed weight not on the NVFP4 grid (diff {diff})"
    print("PASS test_rtn_deploy_grid_fixed_point")


def test_bake_deploy_parity():
    smoke = _load_smoke()
    torch.manual_seed(82)
    model = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 8))
    applied = apply_qat(model, method="nvfp4")
    x = torch.randn(3, 24)
    model.eval()
    with torch.no_grad():
        out_ste = model(x)
    smoke.bake_deploy(model, applied)
    wrappers_left = [n for n, m in model.named_modules()
                     if isinstance(m, FakeQuantLinear)]
    assert not wrappers_left, f"bake left wrappers: {wrappers_left}"
    with torch.no_grad():
        out_baked = model(x)
    assert torch.equal(out_ste, out_baked), \
        "baked artifact diverges from the STE forward"
    print("PASS test_bake_deploy_parity")


def test_bake_parity_through_mla():
    """The lite model's MLA absorbed path slices kv_b_proj.weight
    directly; ste_weight() must keep that path on the grid so bake
    parity holds for a real HeliosLMv5 forward, not just toy MLPs."""
    smoke = _load_smoke()
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(83)
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    applied = apply_qat(model, method="nvfp4")
    ids = torch.randint(3, cfg.vocab_size, (1, 12))
    model.eval()
    with torch.no_grad():
        logits_ste, _, _ = model(ids)
    smoke.bake_deploy(model, applied)
    with torch.no_grad():
        logits_baked, _, _ = model(ids)
    assert torch.equal(logits_ste, logits_baked), \
        "MLA absorbed path broke bake parity"
    # MTP re-bind survived the bake
    for mtp in model.mtp_modules:
        assert mtp.lm_head is model.lm_head, "MTP lm_head not re-bound"
    print("PASS test_bake_parity_through_mla")


def test_bitwise_gate_and_vacuity():
    smoke = _load_smoke()
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    cfg = HeliosLMv5Config(size="lite")
    torch.manual_seed(84)
    ids = torch.randint(0, cfg.vocab_size, (4096,))

    class Args:
        steps = 2
        batch = 2
        seq = 16
        lr = 3e-3
        mtp_weight = 0.0
        seed = 84
        prompt = "Hi"
        probe_tokens = 2

    win = smoke.val_window(ids, Args.seed)
    r1 = smoke.run_qat_arm(cfg, Args, ids, win)
    r2 = smoke.run_qat_arm(cfg, Args, ids, win)
    assert r1["val_loss_baked"] == r2["val_loss_baked"]
    assert r1["probe_ids"] == r2["probe_ids"]
    assert r1["final_train_loss10"] == r2["final_train_loss10"]
    Args.seed = 85  # different seed -> different weights/data order
    r3 = smoke.run_qat_arm(cfg, Args, ids, win)
    assert r3["val_loss_baked"] != r1["val_loss_baked"], \
        "gate is vacuous: different seeds agree bitwise"
    print("PASS test_bitwise_gate_and_vacuity")


def test_smoke_cli():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "smoke.json"
        r = subprocess.run(
            [sys.executable, str(_SMOKE),
             "--steps", "2", "--batch", "2", "--seq", "16",
             "--probe-tokens", "2", "--output", str(out)],
            capture_output=True, text=True, timeout=480,
            cwd=str(_REPO))
        assert r.returncode == 0, f"smoke CLI failed:\n{r.stderr[-800:]}"
        payload = json.loads(out.read_text())
    assert payload["experiment"].startswith("B1-3")
    res = payload["results"]
    labels = [a["label"] for a in res["arms"]]
    assert labels == ["fp32_scratch", "ptq_rtn_nvfp4", "qat_nvfp4"]
    qat = res["arms"][2]
    assert qat["deploy_parity_exact"] is True
    assert qat["wrapped_linears"] > 0
    assert res["bitwise_gate"]["pass"] is True
    assert res["verdict"]["deploy_parity_exact"] is True
    assert len(payload["results_sha256"]) == 64
    print("PASS test_smoke_cli")


def main():
    test_rtn_deploy_grid_fixed_point()
    test_bake_deploy_parity()
    test_bake_parity_through_mla()
    test_bitwise_gate_and_vacuity()
    test_smoke_cli()
    print("T81 PASS: 5/5")


if __name__ == "__main__":
    main()
