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
    mimo_hybrid  5:1 sliding-window (128) : global MLA with sinks;
    lfm_conv     (v5.126, LB9) LFM2-style: gated short convolutions with
                 MLA on every third layer and the last, dense FFN.
  MTP is off so the loss is plain next-token cross-entropy.
- Data: ``uint16`` token shards (``write_shards`` / ``TokenShard``;
  ``uint32`` for a vocabulary over 65,536, v5.127).
  Batch positions come from a generator seeded by (seed, micro-step), so
  a resumed run reads exactly the batches the uninterrupted run would.
- ``pretrain``: AdamW, linear warmup then cosine decay to ``min_lr`` (or
  warmup-stable-decay with an optional decay-phase data switch, LB2),
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

from evastar.configs.config_v5 import (
    AttentionConfig, EvastarV5Config, HybridAttentionConfig, MoEConfig,
    MTPConfig)
from evastar.src.model_v5 import EvastarV5

ARCHS = ("dense", "moe", "k3_block", "mimo_hybrid", "lfm_conv")


def pilot_config(arch="dense", vocab_size=32768, hidden_size=640,
                 num_layers=8, intermediate_size=1792, max_seq=2048,
                 balance="smebu", moe_dispatch="batched", kda_chunk=64,
                 ple_dim=0):
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    kw = dict(size="mid", vocab_size=vocab_size, hidden_size=hidden_size,
              num_hidden_layers=num_layers,
              intermediate_size=intermediate_size,
              max_position_embeddings=max(max_seq, 2048),
              mtp=MTPConfig(enabled=False), per_layer_embed_dim=ple_dim)
    # routed experts at half the dense width: top-2 of 4 activates the same
    # FFN compute per token as the dense arm (equal active FFN parameters)
    moe = MoEConfig(balance_strategy=balance, dispatch=moe_dispatch,
                    expert_hidden_size=intermediate_size // 2)
    if arch == "dense":
        moe = MoEConfig(first_k_dense=num_layers)
    elif arch == "k3_block":
        moe = MoEConfig(balance_strategy=balance, latent_dim=hidden_size // 2,
                        expert_hidden_size=intermediate_size // 2,
                        dispatch=moe_dispatch)
        kw.update(use_attention_residuals=True,
                  attention=AttentionConfig(output_gate=True),
                  hybrid_attention=HybridAttentionConfig(
                      enabled=True, full_attention_every=4,
                      first_layer_full=False, per_channel_decay=True,
                      chunk_size=kda_chunk))
    elif arch == "lfm_conv":
        # LB9 (LFM2): dense FFN; MLA on every third layer and the last,
        # gated short convolutions elsewhere (6 of 16 attention in LFM2)
        moe = MoEConfig(first_k_dense=num_layers)
        attn = {i for i in range(num_layers) if (i + 1) % 3 == 0}
        attn.add(num_layers - 1)
        kw.update(gated_conv_layers=tuple(i for i in range(num_layers)
                                          if i not in attn),
                  hybrid_attention=HybridAttentionConfig(enabled=False))
    elif arch == "mimo_hybrid":
        kw.update(attention=AttentionConfig(sliding_window=128,
                                            global_attention_every=6,
                                            learnable_sink=True),
                  hybrid_attention=HybridAttentionConfig(enabled=False))
    return EvastarV5Config(moe=moe, **kw)


def count_params(model):
    """total, embedding (input embedding + output head + the LB8
    per-layer embedding table) and active (routed experts counted at
    top_k / num_experts) parameter counts."""
    cfg = model.config
    total = emb = routed = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if ("embed_tokens" in name or "lm_head" in name
                or name.startswith("ple.embed.")):
            emb += n
        if ".experts." in name:
            routed += n
    frac = cfg.moe.num_activated_experts / cfg.moe.num_experts
    return {"total": total, "embedding": emb,
            "non_embedding": total - emb,
            "active": int(total - routed * (1 - frac))}


# ---------------------------------------------------------------- data
TOKEN_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def token_dtype(vocab_size):
    """Narrowest shard dtype for a vocabulary: uint16 up to 65,536 ids,
    uint32 above (route B bases: Gemma 4 has 262,144, v5.127)."""
    return "uint16" if vocab_size <= 65536 else "uint32"


def write_shards(token_lists, out_dir, name, dtype="uint16"):
    """Concatenate documents (each a token-id list, already ending in EOS)
    into ``<out_dir>/<name>.bin`` as ``dtype`` (uint16, or uint32 for a
    vocabulary over 65,536). Returns the token count."""
    np_dtype = TOKEN_DTYPES[dtype]
    hi = np.iinfo(np_dtype).max
    out = Path(out_dir) / f"{name}.bin"
    n = 0
    with open(out, "wb") as f:
        for ids in token_lists:
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size and (arr.min() < 0 or arr.max() > hi):
                raise ValueError(f"token ids must fit in {dtype}")
            f.write(arr.astype(np_dtype).tobytes())
            n += arr.size
    return n


class TokenShard:
    def __init__(self, path, dtype="uint16"):
        self.data = np.memmap(path, dtype=TOKEN_DTYPES[dtype], mode="r")

    def __len__(self):
        return len(self.data)

    def batch(self, batch, seq, micro_step, seed):
        """Random windows; positions depend only on (seed, micro_step)."""
        if len(self.data) < seq + 2:
            raise ValueError(f"shard has {len(self.data)} tokens, needs "
                             f"more than seq + 1 = {seq + 1}")
        starts = self._starts(batch, seq, micro_step, seed)
        w = np.stack([self.data[s:s + seq + 1] for s in starts]).astype(np.int64)
        w = torch.from_numpy(w)
        return w[:, :-1], w[:, 1:]

    def _starts(self, batch, seq, micro_step, seed):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + micro_step)
        return torch.randint(0, len(self.data) - seq - 1, (batch,),
                             generator=gen).tolist()

    def eval_windows(self, seq, max_windows):
        """Consecutive, non-overlapping windows from the start (fixed)."""
        n = min(max_windows, (len(self.data) - 1) // seq)
        for i in range(n):
            w = torch.from_numpy(
                self.data[i * seq:i * seq + seq + 1].astype(np.int64))
            yield w[:-1].unsqueeze(0), w[1:].unsqueeze(0)


class KDShard(TokenShard):
    """A token shard with LB4 teacher targets (``logit_kd.KDTargetWriter``
    output for the same stream). ``batch`` reads the same windows as
    ``TokenShard`` and adds an all-ones loss mask and the targets of
    every input position: (x, y, mask, kd_idx [B, T, k], kd_p [B, T, k])."""

    def __init__(self, path, kd_dir=None):
        super().__init__(path)
        from .logit_kd import load_targets
        path = Path(path)
        self.kd_idx, self.kd_p = load_targets(kd_dir or path.parent, path.stem)
        for a in (self.kd_idx, self.kd_p):
            if len(a) != len(self.data) or a.shape != self.kd_idx.shape:
                raise ValueError(f"target arrays {self.kd_idx.shape} / "
                                 f"{self.kd_p.shape} for {len(self.data)} "
                                 "tokens")

    def batch(self, batch, seq, micro_step, seed):
        x, y = super().batch(batch, seq, micro_step, seed)
        starts = self._starts(batch, seq, micro_step, seed)
        idx = np.stack([self.kd_idx[s:s + seq] for s in starts])
        p = np.stack([self.kd_p[s:s + seq] for s in starts])
        return (x, y, torch.ones_like(y),
                torch.from_numpy(idx.astype(np.int64)),
                torch.from_numpy(p.astype(np.float32)))


class _ChunkedLinearCE(torch.autograd.Function):
    """Fused output head + cross-entropy, ``chunk`` rows at a time: the
    forward also computes the gradients, so neither pass holds more than
    one chunk's logits and nothing is recomputed."""

    @staticmethod
    def forward(ctx, h, weight, targets, mask, softcap, chunk):
        w = weight.to(h.dtype)
        n = mask.sum().clamp(min=1) if mask is not None else targets.new_tensor(
            targets.numel(), dtype=torch.float32)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        for i in range(0, h.shape[0], chunk):
            hc, tc = h[i:i + chunk], targets[i:i + chunk]
            z = (hc @ w.T).float()
            if softcap is not None:
                t = torch.tanh(z / softcap)
                z = softcap * t
            lse = torch.logsumexp(z, -1)
            ce = lse - z.gather(1, tc[:, None])[:, 0]
            g = torch.softmax(z, -1)
            g[torch.arange(len(tc), device=g.device), tc] -= 1.0
            if mask is not None:
                mc = mask[i:i + chunk]
                ce, g = ce * mc, g * mc[:, None]
            total += ce.sum()
            if softcap is not None:
                g = g * (1.0 - t * t)
            g = g / n
            grad_h[i:i + chunk] = (g.to(h.dtype) @ w).to(h.dtype)
            grad_w += g.T @ hc.float()
        ctx.save_for_backward(grad_h, grad_w)
        ctx.weight_dtype = weight.dtype
        return total / n

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return (grad_h * grad_out.to(grad_h.dtype),
                (grad_w * grad_out).to(ctx.weight_dtype), None, None, None, None)


def chunked_lm_loss(hidden, weight, targets, mask=None, softcap=None,
                    chunk=2048):
    """Cross-entropy of ``hidden @ weight.T`` against ``targets`` without
    the full [N, vocab] logits (v5.142).

    The output head and the loss run ``chunk`` rows at a time and the
    forward computes the gradients too (one fused pass, nothing
    recomputed), so only one chunk's fp32 logits exist; at 262k vocab
    the full logits of a 8 x 1024 micro-batch are 8.6 GB in fp32.
    ``mask`` (0/1, same shape as targets) scores target tokens only;
    ``softcap`` is the final logit soft cap. Returns the mean over scored
    tokens, equal to the full-logits loss to float tolerance.
    """
    h = hidden.reshape(-1, hidden.shape[-1])
    m = None if mask is None else mask.reshape(-1).to(torch.float32)
    return _ChunkedLinearCE.apply(h, weight, targets.reshape(-1), m, softcap, chunk)


def sparse_kd_loss(logits, idx, p):
    """Cross-entropy to the teacher's top-k: mean over positions with a
    target (p sums to > 0) of -sum_k p_k log p_student(idx_k). Returns
    (loss, positions); loss is 0 with no target in the batch."""
    logp = F.log_softmax(logits.float(), dim=-1)
    lp = logp.gather(-1, idx.to(logp.device))
    p = p.to(logp.device)
    per = -(p * lp).sum(-1)
    has = p.sum(-1) > 0
    n = int(has.sum())
    if n == 0:
        return per.sum() * 0.0, 0
    return per[has].sum() / n, n


def mixed_kd_loss(logits, y, mask, idx, p, alpha):
    """Mix CE/KD per token; unaligned positions retain their full CE loss.

    Normalize once over the scored tokens, so changing teacher coverage
    cannot silently change the relative weight of the remaining positions.
    The second result is mean KD on scored, aligned positions for logging.
    """
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         y.reshape(-1), reduction="none")
    p = p.to(logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    kd = -(p * logp.gather(-1, idx.to(logits.device))).sum(-1).reshape(-1)
    has = p.sum(-1).reshape(-1) > 0
    m = mask.to(logits.device).reshape(-1).float()
    per = torch.where(has, (1 - alpha) * ce + alpha * kd, ce)
    loss = (per * m).sum() / m.sum().clamp(min=1)
    aligned = has * m
    return loss, (kd * aligned).sum() / aligned.sum().clamp(min=1)


# -------------------------------------------------------------- schedule
SCHEDULES = ("cosine", "wsd")
DECAY_SHAPES = ("sqrt", "linear", "cosine")


def wsd_decay_start(total, warmup, decay_frac):
    """First step of the WSD decay phase: the last ``decay_frac`` of the
    run, never before warmup ends."""
    if not 0.0 < decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in (0, 1], got {decay_frac}")
    return max(warmup, total - max(1, round(decay_frac * total)))


def lr_at(step, total, base_lr, warmup, min_ratio=0.1, schedule="cosine",
          decay_frac=0.2, decay_shape="sqrt"):
    """Learning rate for optimizer step ``step`` of ``total``.

    cosine: linear warmup, then cosine decay to ``min_ratio * base_lr``.
    wsd (warmup-stable-decay, LB2; MiniCPM's stable -> decay schedule):
    linear warmup, flat at ``base_lr``, then the last ``decay_frac`` of
    the run decays to ``min_ratio * base_lr``. ``decay_shape`` "sqrt" is
    1 - sqrt(t), which Hagele et al. (2024, arXiv 2405.18392) found best
    among WSD cooldowns; "linear" and "cosine" are the alternatives.
    The stable phase does not depend on ``total``, so a stable-phase
    checkpoint can be resumed with a larger ``steps`` and decayed later.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got "
                         f"{schedule!r}")
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "cosine":
        t = (step - warmup) / max(1, total - warmup)
        return base_lr * (min_ratio + (1 - min_ratio) * 0.5
                          * (1 + math.cos(math.pi * min(1.0, t))))
    if decay_shape not in DECAY_SHAPES:
        raise ValueError(f"decay_shape must be one of {DECAY_SHAPES}, got "
                         f"{decay_shape!r}")
    start = wsd_decay_start(total, warmup, decay_frac)
    if step < start:
        return base_lr
    t = min(1.0, (step - start) / max(1, total - start))
    if decay_shape == "sqrt":
        f = 1 - math.sqrt(t)
    elif decay_shape == "linear":
        f = 1 - t
    else:
        f = 0.5 * (1 + math.cos(math.pi * t))
    return base_lr * (min_ratio + (1 - min_ratio) * f)


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
    for w in val.eval_windows(seq, max_windows):
        x, y = w[0].to(device), w[1].to(device)
        with torch.autocast(device.type, dtype=dtype,
                            enabled=dtype != torch.float32):
            logits = model(x)[0]
        if len(w) > 2:              # SFT windows: score target tokens only
            m = w[2].to(device).reshape(-1).float()
            ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                 y.reshape(-1), reduction="none")
            tot += (ce * m).sum().item()
            n += int(m.sum().item())
            continue
        tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train(was)
    loss = tot / max(n, 1)
    return {"val_loss": round(loss, 5),
            "val_bits_per_byte": round(loss / math.log(2) / bytes_per_token, 5)}


def pretrain(cfg, train, val, out_dir, *, steps, batch, seq, grad_accum=1,
             lr=6e-4, warmup=100, min_lr_ratio=0.1, weight_decay=0.1,
             schedule="cosine", decay_frac=0.2, decay_shape="sqrt",
             decay_train=None, schedule_total=None,
             clip=1.0, eval_every=100, eval_windows=50, ckpt_every=100,
             seed=0, device="cpu", dtype=torch.float32, bytes_per_token=1.0,
             peak_flops=None, compile_model=False, init_from=None,
             kd_alpha=0.5, ce_chunk=0, log=print):
    """Train (or resume) to ``steps`` optimizer steps. Returns the summary.

    Resume: if ``<out_dir>/ckpt.pt`` exists, model and optimizer state and
    the step counter are restored; batches are a function of the step, so
    the continuation is identical to an uninterrupted run (bitwise on CPU).

    Fine-tuning (v5.113, distillation SFT): ``init_from`` names a
    checkpoint whose model weights start a NEW run (ignored on resume,
    optimizer fresh). A shard whose ``batch`` / ``eval_windows`` yield a
    third tensor, a 0/1 loss mask aligned with ``y`` (``distill_sft.
    SFTShard``), trains and evaluates on the masked tokens only: the mean
    over target tokens of each micro-batch.

    Schedule (LB2): ``schedule="wsd"`` with ``decay_frac`` / ``decay_shape``
    (see ``lr_at``). ``decay_train``, a second shard, replaces ``train``
    from the first decay step on: the decay-phase data switch (annealing
    on higher-quality or mid-training data). It needs ``schedule="wsd"``.
    ``schedule_total`` (default ``steps``) is the run length the schedule
    is laid out for: train to ``steps`` < ``schedule_total`` and resume
    later with a larger ``steps``, and the LR and data follow one plan.

    Logit KD (LB4): a shard whose ``batch`` yields teacher targets as a
    fourth and fifth tensor (``KDShard``) trains on
    ``(1 - kd_alpha) * CE + kd_alpha * sparse_kd_loss``; positions
    without a target contribute CE only. ``kd_loss`` is logged.
    """
    if not 0.0 <= kd_alpha <= 1.0:
        raise ValueError(f"kd_alpha must be in [0, 1], got {kd_alpha}")
    if isinstance(ce_chunk, bool) or not isinstance(ce_chunk, int) or ce_chunk < 0:
        raise ValueError(f"ce_chunk must be an int >= 0, got {ce_chunk!r}")
    if decay_train is not None and schedule != "wsd":
        raise ValueError("decay_train needs schedule='wsd'")
    plan = steps if schedule_total is None else schedule_total
    if plan < steps:
        raise ValueError(f"schedule_total {plan} < steps {steps}")
    decay_start = (wsd_decay_start(plan, warmup, decay_frac)
                   if schedule == "wsd" else plan)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.manual_seed(seed)
    model = EvastarV5(cfg).to(device)
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
    elif init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        log(f"initialised from {init_from}")
    params = count_params(model)
    fwd = torch.compile(model) if compile_model else model
    model.train()
    tokens_per_step = batch * seq * grad_accum
    logf = open(out / "log.jsonl", "a")
    t_window, tok_window = time.perf_counter(), 0
    t_step = t_window
    last = {}
    while step < steps:
        cur_lr = lr_at(step, plan, lr, warmup, min_lr_ratio, schedule,
                       decay_frac, decay_shape)
        if decay_train is not None and step >= decay_start:
            # batches stay a function of the step, so resume is still exact
            train = decay_train
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        loss_sum = kd_sum = 0.0
        kd_used = False
        for micro in range(grad_accum):
            b = train.batch(batch, seq, step * grad_accum + micro, seed)
            x, y = b[0].to(device), b[1].to(device)
            with torch.autocast(device.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                if ce_chunk and len(b) <= 4:
                    # v5.142: loss from hidden states, no full logits
                    hidden = fwd(x, return_logits=False)[1]
                    mask = b[2].to(device) if len(b) > 2 else None
                    loss = chunked_lm_loss(
                        hidden, model.lm_head.weight, y, mask,
                        model.config.final_logit_soft_cap, ce_chunk) / grad_accum
                    logits = None
                else:
                    logits = fwd(x)[0]
                if logits is None:
                    pass
                elif len(b) > 4:
                    loss, kd = mixed_kd_loss(logits, y, b[2], b[3],
                                             b[4], kd_alpha)
                    loss = loss / grad_accum
                    kd_sum += kd.item() / grad_accum
                    kd_used = True
                elif len(b) > 2:
                    m = b[2].to(device).reshape(-1).float()
                    ce = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        y.reshape(-1), reduction="none")
                    loss = (ce * m).sum() / m.sum().clamp(min=1) / grad_accum
                else:
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        rec = {"step": step, "lr": round(cur_lr, 8),
               "phase": "decay" if step > decay_start else "main",
               "train_loss": round(loss_sum, 5), "grad_norm": round(gnorm, 4),
               # v5.139: per-step throughput (the window value below also
               # spans warmup and eval) and peak memory, for MFU tuning.
               "step_tokens_per_s": round(tokens_per_step / max(now - t_step, 1e-9), 1)}
        t_step = now
        if device.type == "cuda":
            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if kd_used:
            rec["kd_loss"] = round(kd_sum, 5)
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
            t_step = t_window
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
