"""cuda_smoke.py — Issue #1: run the CUDA paths on a real GPU.

The test suite pins every model to the CPU, so running it on a GPU machine
checks the environment, not CUDA. This script moves the same code to the
GPU and compares it with the CPU where a reference exists:

  forward_fp32     lite model logits on CUDA vs CPU (same weights, TF32 off)
  greedy_generate  16 greedy tokens with the KV cache, CUDA vs CPU
  bf16_train_step  bf16 autocast forward / backward / AdamW step is finite
  fp8_kv_cache     absorbed MLA with the float8_e4m3fn latent cache generates
  pilot_<arch>     the v5.83 pretraining loop on CUDA in bf16 for each pilot
                   architecture: loss falls, resume continues, MoE router
                   biases move
  quant_<method>   AWQ / GPTQ / MXFP4 fused vs unfused on CUDA
  fp8_trainer      FP8Trainer with labels: finite token CE, gradients flow
  gpu_monitor      vllm_engine.GPUMonitor reads the device

Every check runs on its own; a failure is recorded with its error and the
rest still run. Usage (on the GPU machine, from the repo root):

    python examples/cuda_smoke.py --out gpu_smoke.json
"""
import argparse
import json
import math
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

DEV = torch.device("cuda")
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _lite(seed=0, **attn):
    torch.manual_seed(seed)
    cfg = HeliosLMv5Config(size="lite")
    for k, v in attn.items():
        setattr(cfg.attention, k, v)
    return HeliosLMv5(cfg).eval(), cfg


@check
def forward_fp32():
    model, cfg = _lite(1)
    ids = torch.randint(3, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        ref = model(ids)[0]
        got = model.to(DEV)(ids.to(DEV))[0].cpu()
    diff = (got - ref).abs().max().item()
    assert torch.isfinite(got).all()
    assert diff < 1e-3, f"max |cuda - cpu| = {diff:.3e}"
    return {"max_abs_diff": diff}


@check
def greedy_generate():
    model, cfg = _lite(2)
    ids = torch.randint(3, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=16, temperature=0)[0].tolist()
        got = model.to(DEV).generate(ids.to(DEV), max_new_tokens=16,
                                     temperature=0)[0].cpu().tolist()
    same = sum(a == b for a, b in zip(ref, got))
    assert got == ref, f"CUDA and CPU greedy tokens differ ({same}/{len(ref)} equal)"
    return {"tokens": len(got)}


@check
def bf16_train_step():
    model, cfg = _lite(3)
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(3, cfg.vocab_size, (4, 64), device=DEV)
    losses = []
    for _ in range(3):
        with torch.autocast(DEV.type, dtype=torch.bfloat16):
            logits = model(ids[:, :-1])[0]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        assert math.isfinite(gn), f"grad norm {gn}"
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(x) for x in losses), losses
    return {"losses": [round(x, 4) for x in losses]}


@check
def fp8_kv_cache():
    model, cfg = _lite(4, use_absorption=True, kv_cache_dtype="fp8")
    ids = torch.randint(3, cfg.vocab_size, (2, 12), device=DEV)
    with torch.no_grad():
        out = model.to(DEV).generate(ids, max_new_tokens=8, temperature=0)
    assert out.shape == (2, 20), tuple(out.shape)
    return {"shape": list(out.shape)}


def _pilot(arch):
    from helioslm_v5.src.training.pretrain import (
        TokenShard, pilot_config, pretrain, write_shards)
    cfg = pilot_config(arch, vocab_size=512, hidden_size=128, num_layers=4,
                       intermediate_size=256, max_seq=128)
    with tempfile.TemporaryDirectory() as d:
        write_shards([[i % 97 for i in range(20000)]], d, "train")
        write_shards([[i % 97 for i in range(3000)]], d, "val")
        tr, va = TokenShard(Path(d) / "train.bin"), TokenShard(Path(d) / "val.bin")
        kw = dict(batch=8, seq=64, grad_accum=2, lr=3e-3, warmup=5,
                  eval_every=20, eval_windows=8, ckpt_every=20, seed=0,
                  device=DEV.type,
                  dtype=torch.bfloat16 if DEV.type == "cuda" else torch.float32,
                  log=lambda m: None)
        out = Path(d) / "run"
        pretrain(cfg, tr, va, out, steps=40, **kw)
        s = pretrain(cfg, tr, va, out, steps=80, **kw)     # resumes at 40
        log = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
        state = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    evals = [x["val_loss"] for x in log if "val_loss" in x]
    assert [x["step"] for x in log] == list(range(1, 81)), "log not continuous"
    assert all(math.isfinite(x["train_loss"]) for x in log)
    assert evals[-1] < evals[0], f"val loss did not fall: {evals}"
    biases = [v for k, v in state["model"].items() if k.endswith("route_bias")]
    if arch != "dense":
        assert biases and any(b.abs().sum() > 0 for b in biases), "router biases never moved"
    return {"val_loss": [round(e, 4) for e in evals],
            "tokens_per_s_last": log[-1].get("tokens_per_s"), "steps": s["steps"]}


def _named(name, fn):
    fn.__name__ = name
    return check(fn)


for _arch in ("dense", "moe", "k3_block", "mimo_hybrid"):
    _named(f"pilot_{_arch}", lambda a=_arch: _pilot(a))


def _quant(method):
    from helioslm_v5.src.quantization.standard_quant import (
        AWQLinear, GPTQLinear, MXFP4Linear)
    torch.manual_seed(5)
    lin = torch.nn.Linear(512, 1024, bias=False)
    q = (MXFP4Linear.from_linear(lin, block_size=32) if method == "mxfp4" else
         {"awq": AWQLinear, "gptq": GPTQLinear}[method].from_linear(lin, group_size=128))
    q = q.to(DEV).eval()
    x = torch.randn(4, 512, device=DEV)
    with torch.no_grad():
        q.fused = False
        ref = q(x)
        q.fused = True
        got = q(x)
    diff = (got - ref).abs().max().item()
    assert diff < 1e-3, f"fused vs unfused on CUDA: {diff:.3e}"
    return {"max_abs_diff": diff}


for _m in ("awq", "gptq", "mxfp4"):
    _named(f"quant_{_m}", lambda m=_m: _quant(m))


@check
def fp8_trainer():
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer
    model, cfg = _lite(6)
    model = model.to(DEV)
    trainer = FP8Trainer(model, cfg)
    ids = torch.randint(3, cfg.vocab_size, (2, 16), device=DEV)
    losses = [trainer.train_step({"input_ids": ids}, labels=ids.clone()) for _ in range(3)]
    ln_v = math.log(cfg.vocab_size)
    assert all(math.isfinite(x) for x in losses), losses
    assert 0.05 < losses[0] < 3 * ln_v, f"implausible CE {losses[0]} (ln V={ln_v:.2f})"
    g = model.embed_tokens.weight.grad
    assert g is not None and g.norm().item() > 0, "no gradient reached embed_tokens"
    return {"losses": [round(float(x), 4) for x in losses]}


@check
def gpu_monitor():
    from helioslm_v5.src.inference.vllm_engine import GPUMonitor
    mon = GPUMonitor(export_prometheus=False)
    mon.collect()
    total = mon.metrics["gpu_memory_total"]
    assert total and total[0] > 1, mon.metrics
    return {"memory_total_gib": round(total[0], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_smoke.json")
    ap.add_argument("--only", nargs="*", help="run only these checks")
    ap.add_argument("--device", default="cuda",
                    help="cpu: check this script's logic without a GPU "
                         "(proves nothing about CUDA)")
    args = ap.parse_args()
    global DEV
    DEV = torch.device(args.device)
    if DEV.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available on this machine")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": DEV.type, "python": sys.version.split()[0],
           "platform": platform.platform()}
    if DEV.type == "cuda":
        env.update(gpu=torch.cuda.get_device_name(0), capability=".".join(
            map(str, torch.cuda.get_device_capability(0))))
    print(json.dumps(env))
    rows = []
    for fn in CHECKS:
        if args.only and fn.__name__ not in args.only:
            continue
        t0 = time.time()
        try:
            detail = fn()
            row = {"check": fn.__name__, "status": "PASS", "detail": detail}
        except Exception as e:                       # report and continue
            row = {"check": fn.__name__, "status": "FAIL",
                   "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-2000:]}
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{row['status']}] {row['check']} ({row['seconds']}s) "
              f"{row.get('detail', row.get('error'))}", flush=True)
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    n_pass = sum(r["status"] == "PASS" for r in rows)
    print(f"CUDA SMOKE: {n_pass}/{len(rows)} PASS")
    Path(args.out).write_text(json.dumps({"environment": env, "results": rows},
                                         indent=2) + "\n")
    sys.exit(0 if n_pass == len(rows) else 1)


if __name__ == "__main__":
    main()
