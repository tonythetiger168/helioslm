"""Device bugs found by examples/cuda_smoke.py on an RTX 6000 Ada (#1).

CI has no GPU, so both are reproduced on the CPU:

1. FP8Trainer built FP8Linear layers on the default device, so a model
   already on CUDA got CPU layers ("mat2 is on cpu ... cuda:0"). Here the
   model is put on the ``meta`` device: every converted layer must follow
   it instead of landing on the CPU.
2. GPUMonitor.collect called torch.cuda.utilization, which needs NVML
   (nvidia-ml-py); without it the whole monitor raised. It must record
   memory, skip utilization, and still give a recommendation.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402


def test_fp8_conversion_keeps_the_model_device():
    from helioslm_v5.src.training.fp8_trainer import FP8Linear, FP8Trainer
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).to("meta")
    FP8Trainer(model, cfg)
    fp8 = [m for m in model.modules() if isinstance(m, FP8Linear)]
    assert fp8, "no layer was converted"
    for m in fp8:
        devices = {t.device.type for t in list(m.parameters()) + list(m.buffers())}
        assert devices == {"meta"}, devices


def test_gpu_monitor_without_nvml(monkeypatch):
    from helioslm_v5.src.inference import vllm_engine

    def no_nvml(i=None):
        raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed")

    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "utilization", no_nvml)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 24 * gib)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(total_memory=48 * gib))
    mon = vllm_engine.GPUMonitor(export_prometheus=False)
    mon.collect()
    assert mon.metrics["gpu_utilization"] == []
    assert mon.metrics["gpu_memory_total"] == [48.0]
    assert mon.metrics["gpu_memory_ratio"] == [0.5]
    rec = mon.get_hpa_recommendation(current_replicas=2)
    assert rec["action"] == "hold" and rec["load"] == 0.5
