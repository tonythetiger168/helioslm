"""cpt_hf.py — route B: continued pretraining of a Hugging Face base on
per-source token shards (v5.137).

  # one GPU
  python examples/cpt_hf.py --model google/gemma-4-E2B \\
      --data /workspace/cpt_data --tokens 3e9 --out /workspace/runs/b0
  # all GPUs of the pod (DDP)
  torchrun --nproc_per_node 8 examples/cpt_hf.py --model google/gemma-4-E2B \\
      --data /workspace/cpt_data --tokens 3e9 --max-hours 6 --out /workspace/runs/b0

Data: ``fetch_mixture.py`` output (``manifest.json`` + uint32 shards per
source and phase). Every training sequence is a random window of one
source, the source drawn by its renormalised mixture weight, so the
per-source shards never need interleaving on disk. The WSD schedule
(``pretrain.lr_at``, LB2) switches the draw from the main-phase sources
to the decay-phase sources at the first decay step (LB5).

Training: fp32 master weights under bf16 autocast, AdamW, gradient
checkpointing, a global batch of about ``--batch-tokens`` per optimiser
step. ``--freeze`` (a substring list; default the per-layer embedding
tables and the vision / audio towers, which text training never uses)
keeps matching parameters fixed. ``--max-hours`` stops early
and still saves, so a pod's spend has a ceiling. Rank 0 writes
``<out>/log.jsonl`` and, at the end, ``<out>/final`` (bf16
``save_pretrained`` plus the tokenizer), which ``base_gap_eval.py
--models <out>/final`` scores.

``train()`` takes any causal LM returning logits, so CI tests it on a
tiny torch model; ``main()`` adds Hugging Face loading and DDP.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.pretrain import (  # noqa: E402
    TokenShard, lr_at, wsd_decay_start)


class MixedSource:
    """Random windows from weighted per-source shards."""

    def __init__(self, shards, weights):
        if not shards or len(shards) != len(weights):
            raise ValueError("need one weight per shard")
        w = np.asarray(weights, dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("weights must be > 0")
        self.shards, self.p = shards, w / w.sum()

    def batch(self, batch, seq, step, seed):
        rng = np.random.default_rng([seed, step])
        picks = rng.choice(len(self.shards), size=batch, p=self.p)
        xs, ys = [], []
        for i, src in enumerate(picks):
            x, y = self.shards[src].batch(1, seq, step * 1_000 + i, seed)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)


def load_phases(data_dir, seq):
    """{"main": MixedSource, "decay": MixedSource or None} from a
    fetch_mixture.py directory; sources too short for one window are
    dropped."""
    data_dir = Path(data_dir)
    man = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    phases = {}
    for ph in ("main", "decay"):
        shards, weights = [], []
        for name, r in man["sources"].items():
            if r.get("phase") != ph or not r.get("path") or not r.get("tokens"):
                continue
            if r["tokens"] < seq + 2:
                continue
            shards.append(TokenShard(data_dir / r["path"],
                                     dtype=man.get("dtype", "uint32")))
            weights.append(r["weight"])
        phases[ph] = MixedSource(shards, weights) if shards else None
    if phases["main"] is None:
        raise ValueError(f"{data_dir}: no main-phase shard")
    return phases


def lm_loss(model, x, y):
    out = model(input_ids=x)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                           y.reshape(-1))


def train(model, phases, *, steps, micro_batch, seq, grad_accum, lr,
          warmup, decay_frac=0.2, min_ratio=0.1, weight_decay=0.1,
          seed=0, rank=0, world=1, device="cpu", max_hours=None,
          log=None, autocast=True):
    """WSD continued pretraining. Returns {"steps", "tokens", "loss",
    "stopped"}; ``log(row)`` gets one dict per optimiser step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                            weight_decay=weight_decay)
    decay_at = wsd_decay_start(steps, warmup, decay_frac)
    t0, tokens, loss_v, stopped, done = time.time(), 0, float("nan"), None, 0
    model.train()
    dev_type = torch.device(device).type
    for step in range(steps):
        if max_hours is not None and (time.time() - t0) / 3600 > max_hours:
            stopped = "max_hours"
            break
        src = phases["decay"] if (step >= decay_at and phases.get("decay")) \
            else phases["main"]
        cur = lr_at(step, steps, lr, warmup, min_ratio, "wsd", decay_frac)
        for g in opt.param_groups:
            g["lr"] = cur
        total = 0.0
        for k in range(grad_accum):
            x, y = src.batch(micro_batch, seq, step * grad_accum + k,
                             seed * 7919 + rank)
            x, y = x.to(device), y.to(device)
            last = k == grad_accum - 1
            ctx = (model.no_sync() if not last and hasattr(model, "no_sync")
                   else _null())
            with ctx:
                with torch.autocast(dev_type, dtype=torch.bfloat16,
                                    enabled=autocast):
                    loss = lm_loss(model, x, y) / grad_accum
                loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += micro_batch * seq * grad_accum * world
        loss_v = total
        done += 1
        if log is not None and rank == 0:
            el = time.time() - t0
            log({"step": step, "loss": round(loss_v, 4), "lr": cur,
                 "phase": "decay" if src is phases.get("decay") else "main",
                 "tokens": tokens, "elapsed_s": round(el, 1),
                 "tok_per_s": round(tokens / max(el, 1e-9))})
        if not math.isfinite(loss_v):
            stopped = "non_finite_loss"
            break
    return {"steps": done, "tokens": tokens,
            "loss": loss_v, "stopped": stopped}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fit_warmup(steps, warmup, decay_frac):
    """Warmup clamped so the WSD decay (and the switch to the decay-phase
    sources) still starts at the last ``decay_frac`` of the run.
    ``wsd_decay_start`` never starts decay before warmup ends, so a short
    run with the default warmup of 100 would otherwise never decay."""
    decay_len = max(1, round(decay_frac * steps))
    return min(warmup, max(0, steps - decay_len))


def freeze(model, patterns):
    """requires_grad False for parameters whose name contains any pattern;
    returns (trainable, frozen) parameter counts."""
    tr = fr = 0
    for name, p in model.named_parameters():
        if any(s in name for s in patterns):
            p.requires_grad_(False)
            fr += p.numel()
        else:
            tr += p.numel()
    return tr, fr


def frozen_groups(model, depth=3):
    """{name prefix: frozen parameter count}, so a run can confirm its
    --freeze patterns hit the towers they were meant for."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            key = ".".join(name.split(".")[:depth])
            groups[key] = groups.get(key, 0) + p.numel()
    return groups


def dry_run_report(model, out, trainable, frozen):
    rep = {"model_class": type(model).__name__, "trainable_params": trainable,
           "frozen_params": frozen, "frozen_groups": frozen_groups(model)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dry_run.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--batch-tokens", type=float, default=1 << 20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--freeze", nargs="*",
                    default=["per_layer", "vision", "audio"])
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="load the model, apply --freeze, write "
                         "<out>/dry_run.json (model class, trainable / "
                         "frozen counts, frozen parameter groups) and exit")
    a = ap.parse_args(argv)

    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, trust_remote_code=True)
    trainable, frozen = freeze(model, a.freeze)
    if a.dry_run:
        return dry_run_report(model, a.out, trainable, frozen)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)
    phases = load_phases(a.data, a.seq)
    per_micro = a.micro_batch * a.seq * world
    grad_accum = max(1, round(a.batch_tokens / per_micro))
    steps = max(1, int(a.tokens // (per_micro * grad_accum)))
    warmup = fit_warmup(steps, a.warmup, a.decay_frac)
    if warmup != a.warmup and rank == 0:
        print(f"warmup {a.warmup} >= decay start of a {steps}-step run; "
              f"using warmup {warmup}", flush=True)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(dict(
            vars(a), world=world, grad_accum=grad_accum, steps=steps,
            warmup_used=warmup,
            trainable_params=trainable, frozen_params=frozen), indent=2))
    logf = open(out / "log.jsonl", "a") if rank == 0 else None

    def log(row):
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        if row["step"] % 20 == 0 or row["step"] == steps - 1:
            print(json.dumps(row), flush=True)

    res = train(net, phases, steps=steps, micro_batch=a.micro_batch,
                seq=a.seq, grad_accum=grad_accum, lr=a.lr, warmup=warmup,
                decay_frac=a.decay_frac, seed=a.seed, rank=rank, world=world,
                device=device, max_hours=a.max_hours, log=log)
    if rank == 0:
        logf.close()
        final = out / "final"
        model.to(torch.bfloat16).save_pretrained(final,
                                                 safe_serialization=True)
        AutoTokenizer.from_pretrained(a.model, trust_remote_code=True) \
            .save_pretrained(final)
        (out / "result.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return res


if __name__ == "__main__":
    main()
