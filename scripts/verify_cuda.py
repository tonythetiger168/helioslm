"""Explicit-device CUDA area checks + optional unchanged CPU regression suite.

No GPU: write a blocked report and exit 2. --cpu-smoke is a separate harness
self-check, never CUDA evidence. No automatic default-device substitution.
"""
import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config, HybridAttentionConfig  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402



def source_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = _REPO / "SOURCE_REVISION.txt"
        return marker.read_text(encoding="utf-8").strip() if marker.exists() else "unversioned"


def model_on(device, dtype=torch.float32, hybrid=False):
    torch.manual_seed(17)
    cfg = HeliosLMv5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        enabled=hybrid, full_attention_every=2))
    model = HeliosLMv5(cfg).to(device=device, dtype=dtype).eval()
    assert all(t.device == device for t in list(model.parameters()) + list(model.buffers()))
    return model, cfg


def cache_check(device, dtype, hybrid=False):
    model, cfg = model_on(device, dtype, hybrid)
    ids = torch.tensor([[4, 9, 15, 7, 23, 5]], device=device)
    with torch.inference_mode():
        full = model(ids)[0]
        past, steps = None, []
        for index in range(ids.shape[1]):
            out, _, past = model(ids[:, index:index + 1], past_key_values=past, use_cache=True)
            steps.append(out)
        cached = torch.cat(steps, dim=1)
    assert full.device == cached.device == device
    assert full.dtype == cached.dtype == dtype
    error = (full.float() - cached.float()).abs().max().item()
    scaled = error / (full.float().abs().max().item() + 1e-8)
    tol = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 5e-2}[dtype]
    assert torch.isfinite(full).all() and scaled < tol, (scaled, tol)
    return {"max_abs_error": error, "max_scaled_error": scaled, "tolerance": tol}


def mtp_check(device):
    model, _ = model_on(device)
    ids = torch.tensor([[4, 5, 6]], device=device)
    with torch.inference_mode():
        plain = model.generate(ids, max_new_tokens=5, temperature=0)
        drafted = model.generate(ids, max_new_tokens=5, temperature=0, use_mtp=True)
    assert plain.device == drafted.device == device
    assert torch.equal(plain, drafted)
    return {"greedy_exact": True}


def engine_check(device):
    from helioslm_v5.src.inference.vllm_engine import VLLMEngine
    from helioslm_v5.src.inference.paged_attention import BlockManager, PagedAttention
    model, cfg = model_on(device)
    engine = VLLMEngine(model, cfg, block_size=4, max_num_blocks=32)
    prompts = [[4, 8, 9], [12, 5]]
    rids = [engine.add_request(p, max_new_tokens=4, temperature=0) for p in prompts]
    actual = engine.run()
    finished = {r.request_id: r for r in engine.finished_requests}
    for p, rid in zip(prompts, rids):
        assert finished[rid].error is None and actual[rid]
        with torch.inference_mode():
            ref = model.generate(torch.tensor([p], device=device), max_new_tokens=4, temperature=0)
        assert actual[rid] == ref[0, len(p):len(p) + len(actual[rid])].tolist()
    assert engine.block_manager.num_free_blocks() == 32
    pa = PagedAttention(cfg).to(device)
    bm = BlockManager(block_size=4, num_blocks=16, device=str(device))
    bm.allocate(0, 0, pa.num_heads, pa.head_dim)
    x = torch.randn(1, 5, cfg.hidden_size, device=device)
    with torch.inference_mode():
        out = pa(x, bm, [0])
    assert out.device == bm.k_cache.device == device
    assert out.dtype == bm.k_cache.dtype == x.dtype and torch.isfinite(out).all()
    return {"greedy_prefix_exact": True, "paged_cache_device": str(bm.k_cache.device)}


def quant_check(device, method):
    from helioslm_v5.src.quantization.standard_quant import AWQLinear, GPTQLinear, MXFP4Linear
    torch.manual_seed(23)
    linear = torch.nn.Linear(65, 33, device=device)
    x = torch.randn(3, 65, device=device)
    if method == "awq":
        mod = AWQLinear.from_linear(linear, group_size=16, activations=x)
    elif method == "gptq":
        mod = GPTQLinear.from_linear(linear, group_size=16, calibration_data=
                                     torch.randn(96, 65, device=device), act_order=True)
    else:
        mod = MXFP4Linear.from_linear(linear, block_size=32)
    assert all(t.device == device for t in list(mod.parameters()) + list(mod.buffers()))
    with torch.inference_mode():
        mod.fused = False
        ref = mod(x)
        mod.fused = True
        out = mod(x)
    error = (ref - out).abs().max().item() / (ref.abs().max().item() + 1e-8)
    assert out.device == device and torch.isfinite(out).all() and error < 1e-4
    return {"max_scaled_error": error, "tolerance": 1e-4,
            "scope": "quantized linear; not whole-model speed or native CUDA kernels"}


def fp8_check(device):
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer
    model, cfg = model_on(device)
    trainer = FP8Trainer(model, cfg)
    assert all(p.device == device for p in model.parameters())
    assert all(master.device == device for _, master in trainer.param_map)
    ids = torch.tensor([[4, 8, 9, 10, 5]], device=device)
    losses = [trainer.train_step({"input_ids": ids}, labels=ids) for _ in range(2)]
    assert all(math.isfinite(loss) for loss in losses)
    assert trainer.skipped_steps == 0
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()
    return {"losses": losses, "scope": "FP8 value quantization/STE; no tensor-core claim"}


def grpo_check(device):
    from helioslm_v5.src.training.grpo import GRPOTrainer
    model, cfg = model_on(device)
    reference, _ = model_on(device)
    cfg.grpo.group_size = 2
    cfg.grpo.max_new_tokens = 4
    trainer = GRPOTrainer(model, reference, cfg)
    result = trainer.train_step(["2+2?"], ["4"])
    assert math.isfinite(result["loss"])
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(g.device == device and torch.isfinite(g).all() for g in gradients)
    return {"loss": result["loss"], "gradient_tensors": len(gradients)}


def dualpipe_check(device):
    from helioslm_v5.src.training.dualpipe import DualPipeScheduler, DualPipeStage, LayerWrap
    model, cfg = model_on(device)
    stage = DualPipeStage(torch.nn.ModuleList([LayerWrap(layer) for layer in model.layers]))
    scheduler = DualPipeScheduler([stage], num_micro_batches=2)
    inputs = [torch.randn(1, 5, cfg.hidden_size, device=device) for _ in range(2)]
    loss = scheduler.run_dual(inputs, lambda out: out.square().mean())
    grad = model.layers[0].attention.q_a_proj.weight.grad
    assert math.isfinite(loss) and grad is not None and grad.device == device
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    return {"loss": loss, "scope": "single-process schedule; not multi-GPU/NCCL"}


def audio_check(device):
    from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder
    cfg = HeliosLMv5Config(size="lite")
    cfg.multimodal.audio_hidden_size, cfg.multimodal.audio_num_layers = 64, 1
    enc = StreamingAudioEncoder(cfg).to(device).eval()
    mel = torch.randn(1, cfg.multimodal.audio_n_mels, 30, device=device)
    with torch.inference_mode():
        enc.reset_state()
        ref = enc(mel)
        enc.reset_state()
        out = torch.cat([enc(mel[:, :, i:i + 10]) for i in range(0, 30, 10)], dim=1)
    error = (ref - out).abs().max().item()
    assert out.device == device and error < 1e-3 and torch.isfinite(out).all()
    return {"max_abs_error": error, "tolerance": 1e-3}


def vision_check(device):
    from helioslm_v5.src.vision.navit import NaViTEncoder
    cfg = HeliosLMv5Config(size="lite")
    cfg.multimodal.vision_hidden_size = 64
    cfg.multimodal.vision_num_heads = 4
    cfg.multimodal.vision_num_layers = 1
    enc = NaViTEncoder(cfg).to(device).eval()
    with torch.inference_mode():
        out = enc(torch.randn(1, 3, 56, 56, device=device))
    assert out.device == device and out.shape == (1, 16, 64) and torch.isfinite(out).all()
    return {"shape": list(out.shape)}


def run_areas(device, dtypes):
    jobs = [("cache_" + name, lambda dtype=getattr(torch, name): cache_check(device, dtype))
            for name in dtypes]
    jobs += [("hybrid_cache", lambda: cache_check(device, torch.float32, True)),
             ("mtp", lambda: mtp_check(device)), ("engine_paged", lambda: engine_check(device)),
             ("fp8_training", lambda: fp8_check(device)), ("grpo", lambda: grpo_check(device)),
             ("dualpipe", lambda: dualpipe_check(device)), ("audio", lambda: audio_check(device)),
             ("vision", lambda: vision_check(device))]
    jobs += [("quant_" + name, lambda name=name: quant_check(device, name))
             for name in ("awq", "gptq", "mxfp4")]
    rows = []
    for name, fn in jobs:
        row = {"area": name, "device": str(device)}
        start = time.perf_counter()
        try:
            if device.type == "cuda" and name == "cache_bfloat16" and not torch.cuda.is_bf16_supported():
                row.update(status="unsupported", reason="Device does not support bfloat16")
            else:
                row.update(metrics=fn(), status="passed")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        except Exception:
            row.update(status="failed", traceback=traceback.format_exc())
        row["seconds"] = time.perf_counter() - start
        print(f"{name}: {row['status']}", flush=True)
        rows.append(row)
    return rows


def regression(out, timeout):
    commands = [[sys.executable, "-m", "helioslm_v5.tests.test_v5"],
                [sys.executable, "-m", "pytest", "helioslm_v5/tests/", "-q", "-x",
                 "--ignore=helioslm_v5/tests/test_v5.py"],
                [sys.executable, "integration_test_v51.py"]]
    logs = out.parent / (out.stem + "_logs")
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, command in enumerate(commands):
        path = logs / f"regression_{index}.log"
        row = {"command": command, "scope": "unchanged regression fixtures (primarily CPU)",
               "log": str(path)}
        try:
            with path.open("w") as stream:
                result = subprocess.run(command, cwd=_REPO, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            row.update(returncode=result.returncode, status="passed" if result.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            row.update(status="timeout")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_REPO / "benchmarks/cuda_verification.json")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"],
                    default=["float32", "float16", "bfloat16"])
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--run-regression", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.device_index < 0 or args.timeout < 1:
        ap.error("device-index must be nonnegative; timeout must be positive")
    torch.set_num_threads(1)
    available = torch.cuda.is_available()
    report = {"schema_version": 1, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "cuda_build": torch.version.cuda, "cuda_available": available,
                              "device_count": torch.cuda.device_count()},
              "source_revision": source_revision(),
              "cuda_checks_passed": False, "areas": [], "regression": [],
              "unverified_scope": ["multi-GPU/NCCL", "native FP8 tensor-core kernels",
                                   "legacy CPU fixtures do not become CUDA tests"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.cpu_smoke and (not available or args.device_index >= torch.cuda.device_count()):
        report.update(status="blocked_no_cuda", reason="Requested CUDA device unavailable; no CPU fallback")
        code = 2
    else:
        device = torch.device("cpu" if args.cpu_smoke else f"cuda:{args.device_index}")
        if device.type == "cuda":
            try:
                torch.cuda.set_device(device)
                prop = torch.cuda.get_device_properties(device)
                report["environment"].update(gpu_name=prop.name,
                    compute_capability=[prop.major, prop.minor], gpu_total_memory=prop.total_memory)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                report.update(status="failed_cuda_initialization", traceback=traceback.format_exc())
                args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"Report: {args.out}; CUDA initialization failed")
                return 1
        dtypes = ["float32"] if args.cpu_smoke else args.dtypes
        report.update(dtypes_run=dtypes, execution_device=str(device))
        report["areas"] = run_areas(device, dtypes)
        passed = all(row["status"] == "passed" for row in report["areas"])
        report["cuda_checks_passed"] = passed and device.type == "cuda"
        report["status"] = ("cpu_smoke_passed" if args.cpu_smoke else "cuda_area_checks_passed") if passed else "failed_or_partial"
        if args.run_regression:
            report["regression"] = regression(args.out, args.timeout)
        regression_passed = all(r["status"] == "passed" for r in report["regression"])
        if not regression_passed:
            report["status"] = "regression_failed"
        code = 0 if passed and regression_passed else 1
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.out}; status={report['status']}; cuda_checks_passed={report['cuda_checks_passed']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
