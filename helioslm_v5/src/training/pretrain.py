"""Pretraining loop for the ~100M pilot run (v5.83, roadmap Phase C gate).

The last gate before renting a GPU cluster is one complete small run on a
single GPU (an RTX 4060 8 GB is enough at 100M): a sane loss curve,
measured throughput, a held-out metric, and a run that survives being
killed. This module is that loop; ``examples/pilot_prepare_data.py`` and
``examples/pilot_100m.py`` are the command lines.

- ``pilot_config``: the four architectures to compare, at one width:
    dense        every FFN dense (``moe.first_k_dense = num_layers``);
    moe          the mid preset's routed experts (4, top 2) at half the
                 dense FFN width, so active FFN compute matches dense;
    k3_block     3:1 KDA:Gated-MLA, attention residuals, LatentMoE;
    mimo_hybrid  5:1 sliding-window (128) : global MLA with sinks.
  MTP is off so the loss is plain next-token cross-entropy.
- Data: ``uint16`` token shards (``write_shards`` / ``TokenShard``).
  Batch positions come from a generator seeded by (seed, micro-step), so
  a resumed run reads exactly the batches the uninterrupted run would.
- ``pretrain``: AdamW, linear warmup then cosine decay to ``min_lr``,
  gradient accumulation and clipping, bf16 autocast on CUDA, the routers'
  aux-loss-free bias update after every optimizer step (the repo's other
  training loops never call it), held-out bits per byte (comparable
  across tokenizers), tokens/s and an MFU estimate (6 * active params *
  tokens / s / peak), atomic checkpoints, a JSONL log, exact resume.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from helioslm_v5.configs.config_v5 import (
    AttentionConfig, HeliosLMv5Config, HybridAttentionConfig, MoEConfig,
    MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5

ARCHS = ("dense", "moe", "k3_block", "mimo_hybrid")


def pilot_config(arch="dense", vocab_size=32768, hidden_size=640,
                 num_layers=8, intermediate_size=1792, max_seq=2048,
                 balance="smebu"):
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    kw = dict(size="mid", vocab_size=vocab_size, hidden_size=hidden_size,
              num_hidden_layers=num_layers,
              intermediate_size=intermediate_size,
              max_position_embeddings=max(max_seq, 2048),
              mtp=MTPConfig(enabled=False))
    # routed experts at half the dense width: top-2 of 4 activates the same
    # FFN compute per token as the dense arm (equal active FFN parameters)
    moe = MoEConfig(balance_strategy=balance,
                    expert_hidden_size=intermediate_size // 2)
    if arch == "dense":
        moe = MoEConfig(first_k_dense=num_layers)
    elif arch == "k3_block":
        moe = MoEConfig(balance_strategy=balance, latent_dim=hidden_size // 2,
                        expert_hidden_size=intermediate_size // 2)
        kw.update(use_attention_residuals=True,
                  attention=AttentionConfig(output_gate=True),
                  hybrid_attention=HybridAttentionConfig(
                      enabled=True, full_attention_every=4,
                      first_layer_full=False, per_channel_decay=True))
    elif arch == "mimo_hybrid":
        kw.update(attention=AttentionConfig(sliding_window=128,
                                            global_attention_every=6,
                                            learnable_sink=True),
                  hybrid_attention=HybridAttentionConfig(enabled=False))
    return HeliosLMv5Config(moe=moe, **kw)


def count_params(model):
    """total, embedding (input embedding + output head) and active
    (routed experts counted at top_k / num_experts) parameter counts."""
    cfg = model.config
    total = emb = routed = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if "embed_tokens" in name or "lm_head" in name:
            emb += n
        if ".experts." in name:
            routed += n
    frac = cfg.moe.num_activated_experts / cfg.moe.num_experts
    return {"total": total, "embedding": emb,
            "non_embedding": total - emb,
            "active": int(total - routed * (1 - frac))}


# ---------------------------------------------------------------- data
def write_shards(token_lists, out_dir, name):
    """Concatenate documents (each a token-id list, already ending in EOS)
    into ``<out_dir>/<name>.bin`` as uint16. Returns the token count."""
    out = Path(out_dir) / f"{name}.bin"
    n = 0
    with open(out, "wb") as f:
        for ids in token_lists:
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size and (arr.min() < 0 or arr.max() > 65535):
                raise ValueError("token ids must fit in uint16")
            f.write(arr.astype(np.uint16).tobytes())
            n += arr.size
    return n


class TokenShard:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def batch(self, batch, seq, micro_step, seed):
        """Random windows; positions depend only on (seed, micro_step)."""
        if len(self.data) < seq + 2:
            raise ValueError(f"shard has {len(self.data)} tokens, needs "
                             f"more than seq + 1 = {seq + 1}")
        gen = torch.Generator().manual_seed(seed * 1_000_003 + micro_step)
        starts = torch.randint(0, len(self.data) - seq - 1, (batch,),
                               generator=gen).tolist()
        w = np.stack([self.data[s:s + seq + 1] for s in starts]).astype(np.int64)
        w = torch.from_numpy(w)
        return w[:, :-1], w[:, 1:]

    def eval_windows(self, seq, max_windows):
        """Consecutive, non-overlapping windows from the start (fixed)."""
        n = min(max_windows, (len(self.data) - 1) // seq)
        for i in range(n):
            w = torch.from_numpy(
                self.data[i * seq:i * seq + seq + 1].astype(np.int64))
            yield w[:-1].unsqueeze(0), w[1:].unsqueeze(0)


# -------------------------------------------------------------- schedule
def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5
                      * (1 + math.cos(math.pi * min(1.0, t))))


def update_router_biases(model):
    n = 0
    for m in model.modules():
        if hasattr(m, "update_bias") and hasattr(m, "route_bias"):
            m.update_bias()
            n += 1
    return n


@torch.no_grad()
def evaluate(model, val, seq, max_windows, bytes_per_token, device, dtype):
    """Held-out mean loss (nats/token) and bits per UTF-8 byte."""
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for x, y in val.eval_windows(seq, max_windows):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=dtype,
                            enabled=dtype != torch.float32):
            logits = model(x)[0]
        tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train(was)
    loss = tot / max(n, 1)
    return {"val_loss": round(loss, 5),
            "val_bits_per_byte": round(loss / math.log(2) / bytes_per_token, 5)}


def pretrain(cfg, train, val, out_dir, *, steps, batch, seq, grad_accum=1,
             lr=6e-4, warmup=100, min_lr_ratio=0.1, weight_decay=0.1,
             clip=1.0, eval_every=100, eval_windows=50, ckpt_every=100,
             seed=0, device="cpu", dtype=torch.float32, bytes_per_token=1.0,
             peak_flops=None, compile_model=False, log=print):
    """Train (or resume) to ``steps`` optimizer steps. Returns the summary.

    Resume: if ``<out_dir>/ckpt.pt`` exists, model and optimizer state and
    the step counter are restored; batches are a function of the step, so
    the continuation is identical to an uninterrupted run (bitwise on CPU).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(seed)
    model = HeliosLMv5(cfg).to(device)
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    step = 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        log(f"resumed at step {step}")
        # drop log lines written after the checkpoint (they are redone)
        lp = out / "log.jsonl"
        if lp.exists():
            keep = [ln for ln in lp.read_text().splitlines()
                    if ln and json.loads(ln)["step"] <= step]
            lp.write_text("".join(ln + "\n" for ln in keep))
    params = count_params(model)
    fwd = torch.compile(model) if compile_model else model
    model.train()
    tokens_per_step = batch * seq * grad_accum
    logf = open(out / "log.jsonl", "a")
    t_window, tok_window = time.perf_counter(), 0
    last = {}
    while step < steps:
        cur_lr = lr_at(step, steps, lr, warmup, min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(grad_accum):
            x, y = train.batch(batch, seq, step * grad_accum + micro, seed)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                logits = fwd(x)[0]
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    y.reshape(-1)) / grad_accum
            loss.backward()
            loss_sum += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        opt.step()
        update_router_biases(model)
        step += 1
        tok_window += tokens_per_step
        rec = {"step": step, "lr": round(cur_lr, 8),
               "train_loss": round(loss_sum, 5), "grad_norm": round(gnorm, 4)}
        if step % eval_every == 0 or step == steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_window
            tps = tok_window / dt
            rec.update(evaluate(model, val, seq, eval_windows,
                                bytes_per_token, device, dtype))
            rec["tokens_per_s"] = round(tps, 1)
            if peak_flops:
                rec["mfu"] = round(6 * params["active"] * tps / peak_flops, 4)
            t_window, tok_window = time.perf_counter(), 0
            log(json.dumps(rec))
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        last = rec
        if step % ckpt_every == 0 or step == steps:
            tmp = out / "ckpt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "config": cfg.__class__.__name__}, tmp)
            tmp.replace(ckpt)
    logf.close()
    return {"params": params, "steps": step, "tokens": step * tokens_per_step,
            "last": last}
