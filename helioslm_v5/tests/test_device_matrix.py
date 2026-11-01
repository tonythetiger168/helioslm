"""Cross-accelerator capability matrix, oracle and cost record (#175).

CPU always runs; CUDA/MPS/XPU/TPU(torch_xla)/NPU(torch_npu) cases skip
with a reason when the backend is absent. A skip is not a device pass.
"""
import copy
import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.eval import device_matrix as dm  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return HeliosLMv5(dm.oracle_config()).eval()


@pytest.fixture(scope="module")
def moe_model():
    torch.manual_seed(0)
    return HeliosLMv5(HeliosLMv5Config(size="lite")).eval()


def test_matrix_is_well_formed_and_modules_exist():
    dm.validate_matrix()
    for name, row in dm.FEATURES.items():
        importlib.import_module(row["module"])


def test_native_requires_evidence():
    bad = {"x": {"module": "m", "backends": {
        b: {"status": "native"} for b in dm.BACKENDS}}}
    with pytest.raises(ValueError, match="evidence"):
        dm.validate_matrix(bad)


def test_no_accelerator_claim_without_a_run():
    # Only CPU and CUDA have recorded runs; nothing else may say native.
    for name, row in dm.FEATURES.items():
        for b in ("mps", "xpu", "xla", "npu"):
            assert row["backends"][b]["status"] != "native", (name, b)


def test_strict_mode_refuses_fallback():
    assert dm.resolve("kda_fla_kernel", "cpu")["status"] == "fallback"
    with pytest.raises(dm.FallbackError, match="falls back"):
        dm.resolve("kda_fla_kernel", "cpu", strict=True)
    assert dm.resolve("kda_fla_kernel", "cuda", strict=True)["status"] == "native"


def test_unsupported_always_raises():
    with pytest.raises(dm.FallbackError, match="unsupported"):
        dm.resolve("gpu_monitor", "xla")
    with pytest.raises(KeyError):
        dm.resolve("nope", "cpu")
    with pytest.raises(KeyError):
        dm.resolve("mla_attention", "tpu")


def test_probe_reports_reasons():
    p = dm.probe_backends()
    assert set(p) == set(dm.BACKENDS)
    assert p["cpu"]["available"]
    for b, r in p.items():
        assert r["backend"] == b
        if not r["available"]:
            assert r["reason"]


def test_probe_absent_plugin(monkeypatch):
    monkeypatch.setattr(dm.importlib.util, "find_spec", lambda name: None)
    r = dm._probe_plugin("torch_xla", dm._xla_check)
    assert r == {"available": False, "reason": "torch_xla is not installed"}


def test_oracle_fp32_cpu_passes(model):
    r = dm.forward_oracle(model, "cpu", torch.float32)
    assert r["passed"], r
    assert r["full_rel_err"] < 1e-5 and r["decode_rel_err"] < 1e-4
    assert r["top1_agreement"] == 1.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_oracle_half_cpu_within_tolerance(model, dtype):
    for seed in range(3):
        r = dm.forward_oracle(model, "cpu", dtype, seed=seed)
        assert r["passed"], r
        assert r["full_rel_err"] > 0  # it really ran in the half type


def test_oracle_moe_fp32_and_bf16_routing_flip(moe_model):
    assert dm.forward_oracle(moe_model, "cpu", torch.float32)["passed"]
    # Why the half-precision gate runs dense (see oracle_config).
    r = dm.forward_oracle(moe_model, "cpu", torch.bfloat16)
    assert r["full_rel_err"] > 0.1


def test_oracle_catches_a_wrong_kernel(model):
    # A target with one weight off by 2% must fail the fp32 gate.
    bad = copy.deepcopy(model)
    with torch.no_grad():
        bad.lm_head.weight.mul_(1.02)
    r = dm.forward_oracle(model, "cpu", torch.float32, target=bad)
    assert not r["passed"] and r["full_rel_err"] > 5e-3


def test_measure_and_record(model):
    t = dm.measure_generation(model, "cpu", batch=1, prompt_len=8,
                              new_tokens=3, repeats=2, warmup=0)
    assert t["ttft_s_p50"] > 0 and t["decode_tok_s_p50"] > 0
    assert t["peak_mem_gb"] is None and "not measured" in t["peak_mem_source"]
    oracle = {"passed": True}
    fields = dict(backend="cpu", device_name="x", dtype="float32", batch=1,
                  prompt_len=8, new_tokens=3, model="lite",
                  torch_version=torch.__version__, commit="abc",
                  oracle=oracle, timing=t)
    rec = dm.make_record(**fields)
    assert rec["schema"] == "device-bench/1" and rec["usd_per_mtok_decode"] is None
    rec = dm.make_record(usd_per_hour=0.5, **fields)
    assert rec["usd_per_mtok_decode"] == pytest.approx(
        0.5 / (t["decode_tok_s_p50"] * 3600) * 1e6)


def test_record_rejects_missing_context_and_failed_oracle():
    with pytest.raises(ValueError, match="missing"):
        dm.make_record(backend="cpu", timing={})
    fields = dict(backend="cpu", device_name="x", dtype="float32", batch=1,
                  prompt_len=8, new_tokens=3, model="lite",
                  torch_version="t", commit="abc",
                  oracle={"passed": False}, timing={"decode_tok_s_p50": 1})
    with pytest.raises(ValueError, match="oracle"):
        dm.make_record(**fields)


def test_cost_needs_explicit_price():
    with pytest.raises(ValueError):
        dm.cost_per_mtok(100, None)
    with pytest.raises(ValueError):
        dm.cost_per_mtok(0, 1.0)
    assert dm.cost_per_mtok(1000, 3.6) == pytest.approx(1.0)


def test_cli_writes_a_record(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import device_bench
    out = tmp_path / "r.json"
    rc = device_bench.main(["--device", "cpu", "--prompt-len", "4",
                            "--new-tokens", "2", "--repeats", "1",
                            "--usd-per-hour", "1", "--out", str(out)])
    assert rc == 0
    import json
    rec = json.loads(out.read_text())
    assert rec["model"] == "lite-dense" and rec["oracle"]["passed"]
    assert rec["usd_per_mtok_decode"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_oracle_cuda(model):
    for dtype in (torch.float32, torch.bfloat16):
        r = dm.forward_oracle(model, "cuda", dtype)
        assert r["passed"], r


@pytest.mark.skipif(not dm.probe_backends()["xla"]["available"],
                    reason="torch_xla / TPU not available")
def test_oracle_xla(model):  # pragma: no cover - needs a TPU host
    import torch_xla.core.xla_model as xm
    r = dm.forward_oracle(model, xm.xla_device(), torch.bfloat16)
    assert r["passed"], r


@pytest.mark.skipif(not dm.probe_backends()["npu"]["available"],
                    reason="torch_npu / NPU not available")
def test_oracle_npu(model):  # pragma: no cover - needs an NPU host
    r = dm.forward_oracle(model, "npu", torch.float16)
    assert r["passed"], r
