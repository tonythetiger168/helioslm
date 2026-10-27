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

from helioslm_v5.src.training.pretrain import (  # noqa: E402
    TokenShard, chunked_lm_loss, lr_at, wsd_decay_start)


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


def _unwrap(model):
    return getattr(model, "module", model)          # DDP


def final_softcap(model):
    """The model's final logit soft cap (Gemma), or None."""
    cfg = getattr(_unwrap(model), "config", None)
    if cfg is not None and hasattr(cfg, "get_text_config"):
        cfg = cfg.get_text_config()
    return getattr(cfg, "final_logit_softcapping", None)


def lm_loss(model, x, y, ce_chunk=0):
    """Next-token cross-entropy. ``ce_chunk`` > 0 (v5.152) never builds
    the [tokens, vocab] logits: the model runs with ``logits_to_keep=1``
    and ``output_hidden_states=True`` (through the DDP wrapper, so its
    gradient sync is unchanged) and ``chunked_lm_loss`` applies the output
    head and the soft cap ``ce_chunk`` rows at a time. At Gemma's 262k
    vocabulary the fp32 logits of one 4096-token micro-batch are 4.3 GB."""
    if not ce_chunk:
        out = model(input_ids=x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1))
    out = model(input_ids=x, output_hidden_states=True, logits_to_keep=1)
    head = _unwrap(model).get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        raise ValueError("--ce-chunk needs an output head without bias")
    loss = chunked_lm_loss(out.hidden_states[-1], head.weight, y,
                           softcap=final_softcap(model), chunk=ce_chunk)
    # the one kept logit row stays in the graph, so every parameter the
    # forward used gets a gradient (DDP find_unused_parameters)
    return loss + 0.0 * out.logits.float().sum()


def check_chunked_head(model, x, y, ce_chunk, tol=1e-3):
    """Raise ValueError unless ``lm_loss(ce_chunk)`` equals the full-logits
    loss on (x, y) (fp32, no grad): guards against a model whose last
    hidden state is not the output head's input. Returns both losses."""
    with torch.no_grad():
        full = lm_loss(model, x, y).item()
        chunked = lm_loss(model, x, y, ce_chunk).item()
    if not abs(full - chunked) <= tol * max(1.0, abs(full)):
        raise ValueError(f"chunked CE {chunked:.6f} != full logits {full:.6f}: "
                         "this model's last hidden state is not the output "
                         "head input; run without --ce-chunk")
    return full, chunked


def train(model, phases, *, steps, micro_batch, seq, grad_accum, lr,
          warmup, decay_frac=0.2, min_ratio=0.1, weight_decay=0.1,
          seed=0, rank=0, world=1, device="cpu", max_hours=None,
          log=None, autocast=True, ce_chunk=0):
    """WSD continued pretraining. Returns {"steps", "tokens", "loss",
    "stopped"}; ``log(row)`` gets one dict per optimiser step;
    ``ce_chunk`` goes to :func:`lm_loss`."""
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
                    loss = lm_loss(model, x, y, ce_chunk) / grad_accum
                loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        # A rejected batch must never reach AdamW or poison the final model.
        # Agree across DDP ranks before any rank advances the optimizer.
        bad = torch.tensor([not math.isfinite(total),
                            not bool(torch.isfinite(grad_norm))],
                           device=device, dtype=torch.int32)
        if world > 1 and torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bool(bad.any()):
            loss_v = total
            stopped = "non_finite_loss" if bool(bad[0]) else "non_finite_grad"
            opt.zero_grad(set_to_none=True)
            break
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


def cast_frozen(model, dtype):
    """Store frozen parameters in ``dtype`` (v5.179, e.g. bf16): they get
    no gradient or optimizer state, and autocast runs their matmuls in
    bf16 anyway, so fp32 copies only cost memory. Returns bytes saved."""
    saved = 0
    for p in model.parameters():
        if not p.requires_grad and p.dtype != dtype:
            saved += p.numel() * (p.element_size() - torch.empty(0, dtype=dtype).element_size())
            p.data = p.data.to(dtype)
    return saved


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
    ap.add_argument("--frozen-bf16", action="store_true",
                    help="store --freeze parameters in bf16 (saves 2 bytes "
                         "per frozen parameter of GPU memory)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="v5.152: output head + cross-entropy this many "
                         "tokens at a time, without the full logits "
                         "(0 = full logits); checked against the full "
                         "loss on one short window before training")
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
    if a.frozen_bf16:
        saved = cast_frozen(model, torch.bfloat16)
        if rank == 0:
            print(f"frozen parameters in bf16: {saved / 2**30:.1f} GiB saved", flush=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)
    phases = load_phases(a.data, a.seq)
    if a.ce_chunk:
        x, y = phases["main"].batch(1, min(a.seq, 128), 0, a.seed)
        full, chunked = check_chunked_head(model, x.to(device), y.to(device), a.ce_chunk)
        if rank == 0:
            print(f"ce-chunk check: full {full:.6f} chunked {chunked:.6f}", flush=True)
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
                device=device, max_hours=a.max_hours, log=log,
                ce_chunk=a.ce_chunk)
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
