"""cuda_smoke.py — Issue #1: run the CUDA paths on a real GPU.

The test suite pins every model to the CPU, so running it on a GPU machine
checks the environment, not CUDA. This script moves the same code to the
GPU and compares it with the CPU where a reference exists:

  forward_fp32     lite model logits on CUDA vs CPU (same weights, TF32 off)
  greedy_generate  16 greedy tokens with the KV cache, CUDA vs CPU
  bf16_train_step  bf16 autocast forward / backward / AdamW step is finite
  training (v5.90) train_parity_fp32, grad_accum_equivalence, compile_train,
                   checkpoint_portability, mtp_train_step, grpo_train_step,
                   real_size_dense / real_size_moe (the ~100M pilot model, one
                   bf16 step at batch 8 x 1024: peak memory, tokens/s),
                   determinism, long_seq_4k
  fp8_kv_cache     absorbed MLA with the float8_e4m3fn latent cache generates
  pilot_<arch>     the v5.83 pretraining loop on CUDA in bf16 for each pilot
                   architecture: loss falls, resume continues, MoE router
                   biases move
  quant_<method>   AWQ / GPTQ / MXFP4 fused vs unfused on CUDA
  fp8_trainer      FP8Trainer with labels: finite token CE, gradients flow
  gpu_monitor      vllm_engine.GPUMonitor reads the device
  moe_batched_dispatch / real_size_moe_batched
                   v5.106 batched MoE dispatch equals blocked on the GPU,
                   and its real-size step speed
  kda_chunked      v5.107 chunked KDA recurrence equals the loop on the GPU,
                   and its speed

Training checks (v5.90):
  train_parity_fp32       same init and batches, 5 AdamW steps: CUDA tracks CPU
  grad_accum_equivalence  one batch of 8 vs 2 x 4 micro-batches: same grads
  compile_train           the pilot loop with torch.compile: finite, loss falls
  checkpoint_portability  train on CUDA, load on CPU: same logits
  mtp_train_step          main + MTP losses in bf16: MTP heads get gradients
  grpo_train_step         GRPOTrainer.train_step with both models on the device
  real_size_dense / _moe  the ~100M pilot model, bf16 step at batch 8 x 1024:
                          peak memory and tokens/s
  determinism             two identical bf16 runs: loss gap
  long_seq_4k             a 4096-token bf16 forward / backward

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

from evastar.configs.config_v5 import EvastarV5Config  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

DEV = torch.device("cuda")
CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _lite(seed=0, **attn):
    torch.manual_seed(seed)
    cfg = EvastarV5Config(size="lite")
    for k, v in attn.items():
        setattr(cfg.attention, k, v)
    return EvastarV5(cfg).eval(), cfg


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
    from evastar.src.training.pretrain import (
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
    from evastar.src.quantization.standard_quant import (
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
    from evastar.src.training.fp8_trainer import FP8Trainer
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
    from evastar.src.inference.vllm_engine import GPUMonitor
    mon = GPUMonitor(export_prometheus=False)
    mon.collect()
    total = mon.metrics["gpu_memory_total"]
    assert total and total[0] > 1, mon.metrics
    return {"memory_total_gib": round(total[0], 1)}


# ------------------------------------------------------------ training (v5.90)
def _batches(cfg, n, b=4, t=32, seed=7):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, cfg.vocab_size, (b, t + 1), generator=g) for _ in range(n)]


def _train(model, batches, device, dtype=torch.float32, lr=1e-3):
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for data in batches:
        data = data.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(data[:, :-1])[0]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   data[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@check
def train_parity_fp32(steps=5, lr=1e-3):
    """Same init, same batches, 5 AdamW steps: CUDA tracks CPU.

    The max weight difference is not a usable gate on its own: AdamW
    divides by sqrt(v), so an element whose gradient is ~0 moves by up to
    lr per step in the direction of its rounding noise, and CPU and CUDA
    round differently. Measured: on the CPU alone, 1 vs 4 threads already
    gives a max of 2.2e-4 on a handful of elements while the 99.9th
    percentile stays at 4.6e-7; on CUDA (torch 2.8.0+cu128) the max
    reached 3.0e-3. So the gate is the 99.9th percentile (a real device
    bug moves many weights), with the max only bounded by AdamW's own
    limit of 2 * lr per step.
    """
    model, cfg = _lite(11)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    batches = _batches(cfg, steps)
    ref = _train(model, batches, torch.device("cpu"), lr=lr)
    ref_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(init)
    got = _train(model, batches, DEV, lr=lr)
    diff_l = max(abs(a - b) for a, b in zip(ref, got))
    d = torch.cat([(model.state_dict()[k].detach().cpu().float()
                    - ref_w[k].float()).abs().flatten() for k in ref_w])
    p999 = d.kthvalue(max(1, int(0.999 * d.numel()))).values.item()
    diff_w = d.max().item()
    assert diff_l < 1e-3, f"loss drift {diff_l:.2e}: cpu {ref} vs {got}"
    assert p999 < 1e-4, f"weight drift p99.9 {p999:.2e} (max {diff_w:.2e})"
    assert diff_w <= 2 * lr * steps, f"weight drift {diff_w:.2e} > AdamW bound"
    return {"max_loss_diff": diff_l, "max_weight_diff": diff_w,
            "p999_weight_diff": p999,
            "weights_over_1e-4": int((d > 1e-4).sum()), "weights": d.numel(),
            "losses": [round(x, 4) for x in got]}


@check
def grad_accum_equivalence():
    """One batch of 8 vs two micro-batches of 4: identical gradients."""
    model, cfg = _lite(12)
    model = model.to(DEV).train()
    data = _batches(cfg, 1, b=8)[0].to(DEV)

    def grads(chunks):
        model.zero_grad(set_to_none=True)
        for c in chunks:
            logits = model(c[:, :-1])[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   c[:, 1:].reshape(-1)) / len(chunks)
            loss.backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters()
                if p.grad is not None}
    full, accum = grads([data]), grads([data[:4], data[4:]])
    diff = max((full[n] - accum[n]).abs().max().item() for n in full)
    assert set(full) == set(accum) and diff < 1e-5, f"max grad diff {diff:.2e}"
    return {"params": len(full), "max_grad_diff": diff}


@check
def compile_train():
    """The v5.83 loop with torch.compile: finite, loss falls."""
    from evastar.src.training.pretrain import (
        TokenShard, pilot_config, pretrain, write_shards)
    cfg = pilot_config("dense", vocab_size=512, hidden_size=128, num_layers=4,
                       intermediate_size=256, max_seq=128)
    with tempfile.TemporaryDirectory() as d:
        write_shards([[i % 97 for i in range(20000)]], d, "train")
        write_shards([[i % 97 for i in range(3000)]], d, "val")
        s = pretrain(cfg, TokenShard(Path(d) / "train.bin"),
                     TokenShard(Path(d) / "val.bin"), Path(d) / "run", steps=40,
                     batch=8, seq=64, lr=3e-3, warmup=5, eval_every=20,
                     eval_windows=8, ckpt_every=40, seed=0, device=DEV.type,
                     dtype=torch.bfloat16 if DEV.type == "cuda" else torch.float32,
                     compile_model=True, log=lambda m: None)
        log = [json.loads(x) for x in (Path(d) / "run" / "log.jsonl").read_text().splitlines()]
    evals = [x["val_loss"] for x in log if "val_loss" in x]
    assert all(math.isfinite(x["train_loss"]) for x in log)
    assert evals[-1] < evals[0], evals
    return {"val_loss": evals, "tokens_per_s_last": log[-1].get("tokens_per_s"),
            "steps": s["steps"]}


@check
def checkpoint_portability():
    """Train on CUDA, save, load on CPU: same logits."""
    import io
    model, cfg = _lite(13)
    _train(model, _batches(cfg, 3), DEV)
    model.eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        on_dev = model(ids.to(DEV))[0].cpu()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    cpu_model = EvastarV5(cfg).eval()
    cpu_model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    with torch.no_grad():
        on_cpu = cpu_model(ids)[0]
    diff = (on_dev - on_cpu).abs().max().item()
    assert diff < 1e-3, f"logits differ after GPU -> CPU load: {diff:.2e}"
    return {"max_abs_diff": diff}


@check
def mtp_train_step():
    """Main + MTP losses (as train_toy_checkpoint.py) in bf16: MTP heads learn."""
    model, cfg = _lite(14)
    if model.mtp_modules is None:
        raise RuntimeError("lite config has no MTP modules")
    model = model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _batches(cfg, 1, b=4, t=48)[0].to(DEV)
    inp = x[:, :-1]
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
        logits, hidden, _ = model(inp)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1))
        mtp_loss = 0.0
        for mtp in model.mtp_modules:
            d = mtp.module_index + 1
            usable = inp.shape[1] - d - 1
            mlogits, _ = mtp.forward_with_hidden(hidden, inp)
            mtp_loss = mtp_loss + F.cross_entropy(
                mlogits.float().reshape(-1, mlogits.shape[-1]),
                inp[:, d + 1:d + 1 + usable].reshape(-1))
    opt.zero_grad(set_to_none=True)
    (loss + 0.3 * mtp_loss).backward()
    mtp_grads = [p.grad for m in model.mtp_modules for p in m.parameters()
                 if p.grad is not None]
    assert math.isfinite(loss.item()) and math.isfinite(mtp_loss.item())
    assert mtp_grads and all(torch.isfinite(g).all() for g in mtp_grads)
    opt.step()
    return {"loss": round(loss.item(), 4), "mtp_loss": round(mtp_loss.item(), 4),
            "mtp_params_with_grad": len(mtp_grads)}


@check
def grpo_train_step():
    """GRPOTrainer.train_step (integration T4) with both models on the device."""
    from evastar.src.training.grpo import GRPOTrainer
    model, cfg = _lite(4)
    ref, _ = _lite(14)
    cfg.grpo.group_size = 4
    cfg.grpo.max_new_tokens = 8
    trainer = GRPOTrainer(model.to(DEV), ref.to(DEV), cfg)
    out = trainer.train_step(["What is 2+2?"], ["4"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert math.isfinite(out["loss"]), out
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(g.device.type == DEV.type for g in grads)
    return {k: round(float(out[k]), 4) for k in ("loss", "policy_loss", "kl_penalty")}


def _real_size(arch, batch=8, seq=1024, moe_dispatch="blocked"):
    """One bf16 optimizer step of the real ~100M pilot model: memory and speed."""
    from evastar.src.training.pretrain import count_params, pilot_config
    cfg = pilot_config(arch, moe_dispatch=moe_dispatch)
    torch.manual_seed(0)
    model = EvastarV5(cfg).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(3, cfg.vocab_size, (batch, seq + 1), device=DEV)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        t0 = time.time()
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            logits = model(x[:, :-1])[0]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    assert math.isfinite(loss.item())
    out = {"params_active_M": round(count_params(model)["active"] / 1e6, 1),
           "tokens_per_s": round(batch * seq / min(times[1:]), 1),
           "loss": round(loss.item(), 3)}
    if DEV.type == "cuda":
        out["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    return out


for _arch in ("dense", "moe"):
    _named(f"real_size_{_arch}", lambda a=_arch: _real_size(a))
# v5.106: the same MoE step with moe.dispatch="batched" (compare tokens/s
# with real_size_moe)
_named("real_size_moe_batched", lambda: _real_size("moe", moe_dispatch="batched"))


@check
def moe_batched_dispatch():
    """v5.106: batched MoE dispatch equals blocked on this device (fp32
    outputs and gradients), and under bf16 autocast."""
    import copy
    from evastar.configs.config_v5 import MoEConfig
    from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE
    cfg = EvastarV5Config(size="lite", moe=MoEConfig(num_experts=16,
                           num_activated_experts=2, device_group_size=1))
    torch.manual_seed(15)
    blocked = DeviceLimitedMoE(cfg).to(DEV).train()
    batched = copy.deepcopy(blocked)
    batched.dispatch = "batched"
    assert batched._batchable()
    x = torch.randn(4, 256, cfg.hidden_size, device=DEV)
    outs = []
    for m in (blocked, batched):
        y = m(x)
        y.pow(2).mean().backward()
        outs.append((y.detach(), [p.grad for p in m.parameters()]))
    out_diff = (outs[0][0] - outs[1][0]).abs().max().item()
    grad_diff = max((a - b).abs().max().item()
                    for a, b in zip(outs[0][1], outs[1][1])
                    if a is not None and b is not None)
    assert out_diff < 1e-4 and grad_diff < 1e-4, (out_diff, grad_diff)
    with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.bfloat16):
        bf = (blocked(x).float() - batched(x).float()).abs().max().item()
    assert bf < 5e-2, bf
    return {"max_abs_diff": out_diff, "grad_max_abs_diff": grad_diff,
            "bf16_max_abs_diff": bf}


@check
def kda_chunked():
    """v5.107: the chunked-parallel KDA recurrence equals the per-token loop
    on this device (outputs, state, gradients), and how much faster one
    layer's forward + backward is at 8 x 1024 tokens."""
    import copy
    from evastar.configs.config_v5 import HybridAttentionConfig
    from evastar.src.attention.linear_attention import GatedDeltaAttention
    cfg = EvastarV5Config(size="lite", hybrid_attention=HybridAttentionConfig(
        enabled=True, per_channel_decay=True))
    torch.manual_seed(16)
    loop = GatedDeltaAttention(cfg).to(DEV)
    chunked = copy.deepcopy(loop)
    chunked.chunk_size = 32
    x = torch.randn(2, 200, cfg.hidden_size, device=DEV)
    res = []
    for m in (loop, chunked):
        y, (s,) = m(x, use_cache=True)
        y.pow(2).mean().backward()
        res.append((y.detach(), s, [p.grad for p in m.parameters()]))
    diff = max((res[0][0] - res[1][0]).abs().max().item(),
               (res[0][1] - res[1][1]).abs().max().item())
    gdiff = max((a - b).abs().max().item() for a, b in zip(res[0][2], res[1][2]))
    assert diff < 1e-4 and gdiff < 1e-4, (diff, gdiff)
    xb = torch.randn(8, 1024, cfg.hidden_size, device=DEV)
    times = {}
    for name, m in (("loop", loop), ("chunked", chunked)):
        best = float("inf")
        for _ in range(2):
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(DEV.type, dtype=torch.bfloat16,
                                enabled=DEV.type == "cuda"):
                y, _ = m(xb)
            y.float().pow(2).mean().backward()
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.time() - t0)
        times[name] = round(best, 3)
    return {"max_abs_diff": diff, "grad_max_abs_diff": gdiff,
            "fwd_bwd_s": times, "speedup": round(times["loop"] / times["chunked"], 2)}


@check
def determinism():
    """Two identical bf16 runs of the tiny pilot model: same losses (report
    the gap; atomics in backward can make CUDA runs differ slightly)."""
    def run():
        model, cfg = _lite(15)
        return _train(model, _batches(cfg, 5), DEV,
                      dtype=torch.bfloat16 if DEV.type == "cuda" else torch.float32)
    a, b = run(), run()
    gap = max(abs(x - y) for x, y in zip(a, b))
    assert gap < 1e-2, f"runs diverge by {gap:.2e}: {a} vs {b}"
    return {"max_loss_gap": gap, "bitwise_equal": a == b}


@check
def long_seq_4k():
    """A 4096-token bf16 forward / backward (lite width; positions raised
    from the lite default of 2048)."""
    torch.manual_seed(16)
    cfg = EvastarV5Config(size="lite", max_position_embeddings=4096)
    model = EvastarV5(cfg).to(DEV).train()
    x = torch.randint(3, cfg.vocab_size, (1, 4097), device=DEV)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
        logits = model(x[:, :-1])[0]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               x[:, 1:].reshape(-1))
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    assert math.isfinite(loss.item()) and math.isfinite(gn)
    out = {"loss": round(loss.item(), 3), "grad_norm": round(gn, 3)}
    if DEV.type == "cuda":
        out["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    return out


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
