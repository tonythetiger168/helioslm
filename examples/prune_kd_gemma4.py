#!/usr/bin/env python3
"""Prune a Gemma 4 text model (E2B) to a small student and distil it from
a larger Gemma 4 teacher (E4B) with the same tokenizer (v1.0 L-Edge pilot).

Two sub-commands:

``prune``  Run calibration windows through the source model, score
           layers (block influence, 1 - cos(in, out)), hidden channels
           (mean |x| after every pre-attention / pre-FFN norm) and FFN
           channels (mean |x| into ``down_proj``), then write a smaller
           ``Gemma4ForCausalLM`` with ``--layers-kv`` non-shared plus
           ``--layers-shared`` KV-shared layers, ``--hidden`` channels and
           ``--ffn`` FFN channels per layer. The last non-shared layer of
           each attention type is always kept, so the shared layers read
           the same KV source as in the original model; the source's last
           layer is kept so the model still ends on full attention.
           Embedding scales that depend on ``hidden_size`` are compensated.

``kd``     Train the student on fetch_mixture.py shards with the forward
           KL divergence to the teacher's next-token distribution (logit
           soft cap applied to both), computed a chunk of rows at a time
           (no full [tokens, 262k] logits). ``--hours`` sizes the run from
           the measured step time, with a WSD schedule (the stable phase
           does not depend on the total), so a fixed GPU budget always
           ends with a full decay. AdamW only.

Both write ``run.json`` / ``log.jsonl`` / ``result.json`` and a bf16
``final/`` with the tokenizer, loadable by AutoModelForCausalLM.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from helioslm_v5.src.training.pretrain import lr_at  # noqa: E402


# --------------------------------------------------------------------- utils

def text_model(model):
    """The Gemma4TextModel inside a CausalLM / ConditionalGeneration."""
    for m in model.modules():
        if type(m).__name__ == "Gemma4TextModel":
            return m
    raise ValueError("no Gemma4TextModel inside this model")


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def first_shared(cfg):
    n = cfg.num_hidden_layers - (getattr(cfg, "num_kv_shared_layers", 0) or 0)
    return n if n > 0 else cfg.num_hidden_layers


def softcap(x, cap):
    return cap * torch.tanh(x / cap) if cap else x


# --------------------------------------------------------------------- prune

@torch.no_grad()
def calibrate(model, windows, device):
    """Importance statistics over ``windows`` (list of [1, T] LongTensors)."""
    tm = text_model(model)
    layers = list(tm.layers)
    L = len(layers)
    H = tm.config.hidden_size
    bi = torch.zeros(L, dtype=torch.float64)
    hid = torch.zeros(H, dtype=torch.float64, device=device)
    ffn = [None] * L
    count = [0]
    hooks = []

    def layer_hook(i):
        def f(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            y = out[0] if isinstance(out, tuple) else out
            cos = F.cosine_similarity(x.float(), y.float(), dim=-1)
            bi[i] += (1 - cos).sum().item()
        return f

    def norm_hook(mod, args, out):
        hid.add_(out.float().abs().sum(dim=(0, 1)).double())

    def down_hook(i):
        def f(mod, args):
            a = args[0].float().abs().sum(dim=(0, 1)).double()
            ffn[i] = a if ffn[i] is None else ffn[i] + a
        return f

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(layer_hook(i), with_kwargs=True))
        hooks.append(layer.input_layernorm.register_forward_hook(norm_hook))
        hooks.append(layer.pre_feedforward_layernorm.register_forward_hook(norm_hook))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(down_hook(i)))
    try:
        for x in windows:
            model(input_ids=x.to(device), use_cache=False, logits_to_keep=1)
            count[0] += x.numel()
    finally:
        for h in hooks:
            h.remove()
    n = max(1, count[0])
    return {"bi": (bi / n).tolist(), "hidden": (hid / n).cpu().tolist(),
            "ffn": [(f / n).cpu().tolist() for f in ffn], "tokens": n}


def choose_layers(cfg, bi, n_kv, n_shared):
    """Kept source layer indices (sorted): ``n_kv`` non-shared, ``n_shared``
    KV-shared. Forced: the last non-shared layer of every type, the last
    source layer; the rest by block influence (higher = more important)."""
    types = list(cfg.layer_types)
    fs = first_shared(cfg)
    kv_src = list(range(fs))
    sh_src = list(range(fs, cfg.num_hidden_layers))
    force_kv = {max(i for i in kv_src if types[i] == t) for t in set(types[:fs])}
    if n_kv < len(force_kv):
        raise ValueError(f"--layers-kv must be >= {len(force_kv)}")
    rest = sorted((i for i in kv_src if i not in force_kv), key=lambda i: -bi[i])
    keep_kv = sorted(force_kv | set(rest[: n_kv - len(force_kv)]))
    keep_sh = []
    if sh_src and n_shared:
        if n_shared > len(sh_src):
            raise ValueError(f"--layers-shared must be <= {len(sh_src)}")
        force_sh = {sh_src[-1]}
        rest = sorted((i for i in sh_src if i not in force_sh), key=lambda i: -bi[i])
        keep_sh = sorted(force_sh | set(rest[: n_shared - 1]))
    elif not n_shared and types[keep_kv[-1]] != "full_attention":
        raise ValueError("without shared layers the last kept layer must be full attention")
    shared_types = {types[i] for i in keep_sh}
    missing = shared_types - {types[i] for i in keep_kv}
    if missing:
        raise ValueError(f"no non-shared KV source for {missing}")
    return keep_kv, keep_sh


def top_idx(scores, k):
    s = torch.as_tensor(scores)
    if k > len(s):
        raise ValueError(f"cannot keep {k} of {len(s)} channels")
    return torch.sort(torch.topk(s, k).indices).values


def student_config(src_cfg, keep_kv, keep_sh, hidden, ffn):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    d = src_cfg.to_dict()
    for k in ("per_layer_config", "architectures", "transformers_version",
              "_name_or_path", "auto_map"):
        d.pop(k, None)
    keep = keep_kv + keep_sh
    d.update(num_hidden_layers=len(keep), num_kv_shared_layers=len(keep_sh),
             layer_types=[src_cfg.layer_types[i] for i in keep],
             hidden_size=hidden, intermediate_size=ffn,
             use_double_wide_mlp=False, model_type="gemma4_text")
    d["global_head_dim"] = getattr(src_cfg, "global_head_dim", None) or \
        src_cfg.per_layer_config[[i for i, t in enumerate(src_cfg.layer_types)
                                  if t == "full_attention"][0]].head_dim
    d["architectures"] = ["Gemma4ForCausalLM"]
    return Gemma4TextConfig(**d)


@torch.no_grad()
def build_student(src_model, stats, keep_kv, keep_sh, hidden, ffn):
    """A new Gemma4ForCausalLM whose every tensor is a slice of the source."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    src = text_model(src_model)
    scfg = text_config(src_model)
    cfg = student_config(scfg, keep_kv, keep_sh, hidden, ffn)
    old = torch.get_default_dtype()
    torch.set_default_dtype(src.embed_tokens.weight.dtype)
    try:
        dst_full = Gemma4ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old)
    dst = dst_full.model
    H0 = scfg.hidden_size
    h = top_idx(stats["hidden"], hidden)
    keep = keep_kv + keep_sh
    s = math.sqrt(H0 / hidden)               # embed_scale sqrt(H) compensation
    P = scfg.hidden_size_per_layer_input or 0
    done = set()

    def put(name, value):
        p = dict(dst.named_parameters()).get(name)
        if p is None:
            p = dict(dst.named_buffers())[name]
        if tuple(p.shape) != tuple(value.shape):
            raise ValueError(f"{name}: {tuple(value.shape)} != {tuple(p.shape)}")
        p.copy_(value.to(p.dtype))
        done.add(name)

    put("embed_tokens.weight", src.embed_tokens.weight[:, h] * s)
    put("norm.weight", src.norm.weight[h] / s)
    if P:
        ple = src.embed_tokens_per_layer.weight.view(-1, scfg.num_hidden_layers, P)
        put("embed_tokens_per_layer.weight", ple[:, keep, :].reshape(ple.shape[0], -1))
        proj = src.per_layer_model_projection.weight.view(scfg.num_hidden_layers, P, H0)
        put("per_layer_model_projection.weight",
            proj[keep][:, :, h].reshape(-1, hidden) / s)
        put("per_layer_projection_norm.weight", src.per_layer_projection_norm.weight)
    for j, i in enumerate(keep):
        a, b = src.layers[i], dst.layers[j]
        f = top_idx(stats["ffn"][i], ffn)
        pre = f"layers.{j}."
        put(pre + "self_attn.q_proj.weight", a.self_attn.q_proj.weight[:, h])
        put(pre + "self_attn.o_proj.weight", a.self_attn.o_proj.weight[h, :])
        put(pre + "self_attn.q_norm.weight", a.self_attn.q_norm.weight)
        if not b.self_attn.is_kv_shared_layer:
            put(pre + "self_attn.k_proj.weight", a.self_attn.k_proj.weight[:, h])
            if getattr(b.self_attn, "v_proj", None) is not None:
                put(pre + "self_attn.v_proj.weight", a.self_attn.v_proj.weight[:, h])
            put(pre + "self_attn.k_norm.weight", a.self_attn.k_norm.weight)
        put(pre + "mlp.gate_proj.weight", a.mlp.gate_proj.weight[f][:, h])
        put(pre + "mlp.up_proj.weight", a.mlp.up_proj.weight[f][:, h])
        put(pre + "mlp.down_proj.weight", a.mlp.down_proj.weight[h][:, f])
        for n in ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            put(pre + n + ".weight", getattr(a, n).weight[h])
        if P:
            put(pre + "per_layer_input_gate.weight", a.per_layer_input_gate.weight[:, h])
            put(pre + "per_layer_projection.weight", a.per_layer_projection.weight[h, :])
            put(pre + "post_per_layer_input_norm.weight", a.post_per_layer_input_norm.weight[h])
        put(pre + "layer_scalar", a.layer_scalar)
    want = {n for n, _ in dst.named_parameters()}
    want |= {n for n, _ in dst.named_buffers()
             if n.endswith("layer_scalar")}
    left = sorted(want - done)
    if left:
        raise ValueError(f"student tensors not filled from the source: {left[:8]}")
    dst_full.tie_weights()
    return dst_full


def param_counts(model):
    tm = text_model(model)
    total = sum(p.numel() for p in model.parameters())
    ple = tm.embed_tokens_per_layer.weight.numel() if getattr(
        tm, "embed_tokens_per_layer", None) is not None else 0
    emb = tm.embed_tokens.weight.numel()
    return {"total": total, "per_layer_embedding_table": ple,
            "effective": total - ple, "embedding": emb,
            "non_embedding": total - ple - emb}


def calib_windows(data, seq, n, seed=0):
    from cpt_hf import load_phases
    src = load_phases(data, seq)["main"]
    return [src.batch(1, seq, k, seed + 99)[0] for k in range(n)]


def cmd_prune(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    windows = calib_windows(a.data, a.seq, a.calib)
    stats = calibrate(model, windows, device)
    cfg = text_config(model)
    keep_kv, keep_sh = choose_layers(cfg, stats["bi"], a.layers_kv, a.layers_shared)
    model.to("cpu")
    student = build_student(model, stats, keep_kv, keep_sh, a.hidden, a.ffn)
    counts = param_counts(student)
    final = out / "final"
    student.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(final)
    res = dict(source=a.model, kept_kv_layers=keep_kv, kept_shared_layers=keep_sh,
               hidden=a.hidden, ffn=a.ffn, params=counts,
               source_params=param_counts(model), calib_tokens=stats["tokens"],
               block_influence=stats["bi"], seconds=round(time.time() - t0, 1))
    (out / "prune.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "block_influence"}))
    return res


# ------------------------------------------------------------------------ kd

def _kl_chunk(hs, ht, ws, wt, cap_s, cap_t, temp):
    dt = torch.bfloat16 if hs.is_cuda else torch.float32
    ls = softcap((hs.to(dt) @ ws.to(dt).t()).float(), cap_s) / temp
    with torch.no_grad():
        lt = softcap((ht.to(dt) @ wt.to(dt).t()).float(), cap_t) / temp
        pt = F.softmax(lt, dim=-1)
        ent = (pt * F.log_softmax(lt, dim=-1)).sum()
    return ent - (pt * F.log_softmax(ls, dim=-1)).sum()


def kd_loss(hs, ht, ws, wt, cap_s, cap_t, chunk=1024, temp=1.0):
    """Mean over rows of KL(teacher || student), ``chunk`` rows at a time,
    each chunk recomputed in backward (torch.utils.checkpoint) so only one
    chunk of logits is alive. hs/ht: [N, Hs]/[N, Ht]."""
    from torch.utils.checkpoint import checkpoint
    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        total = total + checkpoint(_kl_chunk, hs[i:i + chunk], ht[i:i + chunk],
                                   ws, wt, cap_s, cap_t, temp, use_reentrant=False)
    return total * (temp * temp) / hs.shape[0]


def last_hidden(model, x, grad):
    with torch.set_grad_enabled(grad):
        out = model(input_ids=x, output_hidden_states=True, logits_to_keep=1,
                    use_cache=False)
    return out.hidden_states[-1], out.logits


def check_head(model, x):
    """The last hidden state must be the output head's input (Gemma: after
    the final norm) or the chunked loss would be wrong."""
    with torch.no_grad():
        full = model(input_ids=x, use_cache=False).logits[0, -1].float()
        h, _ = last_hidden(model, x, False)
        w = model.get_output_embeddings().weight
        mine = softcap(h[0, -1].float() @ w.float().t(),
                       getattr(text_config(model), "final_logit_softcapping", None))
    err = (full - mine).abs().max().item() / max(1.0, full.abs().max().item())
    if err > 5e-2:
        raise ValueError(f"last hidden state is not the head input (rel err {err:.3g})")
    return err


def cmd_kd(a):
    from cpt_hf import load_phases
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    student = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    student.config.use_cache = False
    if a.freeze_ple:
        tm = text_model(student)
        if getattr(tm, "embed_tokens_per_layer", None) is not None:
            tm.embed_tokens_per_layer.weight.requires_grad_(False)
            tm.embed_tokens_per_layer.to(torch.bfloat16)
    if a.grad_ckpt:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    student.to(device).train()
    teacher.to(device).eval().requires_grad_(False)
    cap_s = getattr(text_config(student), "final_logit_softcapping", None)
    cap_t = getattr(text_config(teacher), "final_logit_softcapping", None)
    # cap repeats of small sources (rephrases, Taigi) over the planned tokens
    plan = a.tokens or a.plan_tokens
    phases = load_phases(a.data, a.seq, plan, a.max_repeat, decay_frac=a.decay_frac)
    x0, _ = phases["main"].batch(1, min(64, a.seq), 0, a.seed)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    with torch.autocast(**amp):
        errs = {"student": check_head(student, x0.to(device)),
                "teacher": check_head(teacher, x0.to(device))}
    ws = student.get_output_embeddings().weight
    wt = teacher.get_output_embeddings().weight
    trainable = [p for p in student.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            fused=device == "cuda")
    per_micro = a.micro_batch * a.seq
    accum = max(1, round(a.batch_tokens / per_micro))
    steps = max(1, int(a.tokens // (per_micro * accum))) if a.tokens else 10 ** 9
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = dict(vars(a), grad_accum=accum, head_check=errs,
               params=param_counts(student))
    (out / "run.json").write_text(json.dumps(run, indent=2))
    logf = open(out / "log.jsonl", "a")
    t_start = time.time()
    sized = not a.hours
    step = 0
    seen = 0
    ema = None
    t_run = time.time()
    while step < steps:
        if not sized and step == a.size_after:
            per = (time.time() - t_run) / max(1, step - 2)
            steps = max(step + 1, int(a.hours * 3600 / per))
            if a.tokens:
                steps = min(steps, max(step + 1, int(a.tokens // (per_micro * accum))))
            sized = True
            print(json.dumps({"sized_steps": steps, "sec_per_step": round(per, 2),
                              "planned_tokens": steps * per_micro * accum}), flush=True)
        if step == 2:
            t_run = time.time()
        total_for_lr = steps if sized else 10 ** 9
        lr = lr_at(step, total_for_lr, a.lr, a.warmup, min_ratio=0.1,
                   schedule="wsd", decay_frac=a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        in_decay = sized and step >= steps - max(1, round(a.decay_frac * steps))
        src = phases["decay"] if in_decay and phases.get("decay") else phases["main"]
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        ce_tot = 0.0
        for m in range(accum):
            x, y = src.batch(a.micro_batch, a.seq, step * accum + m, a.seed)
            x, y = x.to(device), y.to(device)
            with torch.autocast(**amp):
                with torch.no_grad():
                    ht, _ = last_hidden(teacher, x, False)
                hs, lg = last_hidden(student, x, True)
            hs2 = hs.reshape(-1, hs.shape[-1])
            ht2 = ht.reshape(-1, ht.shape[-1])
            loss = kd_loss(hs2, ht2, ws, wt, cap_s, cap_t, a.kd_chunk, a.temp)
            if a.ce_weight:
                from helioslm_v5.src.training.pretrain import chunked_lm_loss
                ce = chunked_lm_loss(hs2.float(), ws, y.reshape(-1), softcap=cap_s,
                                     chunk=a.kd_chunk)
                ce_tot += ce.item() / accum
                loss = loss + a.ce_weight * ce
            loss = (loss + 0.0 * lg.float().sum()) / accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip).item()
        opt.step()
        seen += per_micro * accum
        ema = tot if ema is None else 0.98 * ema + 0.02 * tot
        el = time.time() - t_start
        row = {"step": step, "kl": round(tot, 4), "kl_ema": round(ema, 4),
               "lr": lr, "gnorm": round(gn, 3), "tokens": seen,
               "phase": "decay" if in_decay else "main",
               "tok_s": round(seen / el), "hours": round(el / 3600, 3)}
        if a.ce_weight:
            row["ce"] = round(ce_tot, 4)
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        if step % 20 == 0 or step == steps - 1:
            print(json.dumps(row), flush=True)
        if not math.isfinite(tot):
            raise RuntimeError(f"non-finite loss at step {step}")
        if a.save_every and step and step % a.save_every == 0:
            student.save_pretrained(out / "ckpt", safe_serialization=True)
        step += 1
        if a.max_hours and el > a.max_hours * 3600:
            print(json.dumps({"stopped": "max_hours", "step": step}), flush=True)
            break
    logf.close()
    final = out / "final"
    student.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.student).save_pretrained(final)
    res = {"steps": step, "tokens": seen, "kl_ema": ema,
           "hours": round((time.time() - t_start) / 3600, 3)}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)
    return res


# ---------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="fetch_mixture.py directory (calibration)")
    p.add_argument("--out", required=True)
    p.add_argument("--layers-kv", type=int, default=8)
    p.add_argument("--layers-shared", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--calib", type=int, default=256, help="calibration windows")
    p.add_argument("--seq", type=int, default=1024)
    k = sub.add_parser("kd")
    k.add_argument("--student", required=True)
    k.add_argument("--teacher", required=True)
    k.add_argument("--data", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--tokens", type=float, default=0, help="cap on training tokens")
    k.add_argument("--hours", type=float, default=0,
                   help="size the run to this many hours from the measured step time")
    k.add_argument("--plan-tokens", type=float, default=1e9,
                   help="token estimate for --max-repeat when --tokens is 0")
    k.add_argument("--max-repeat", type=float, default=4.0)
    k.add_argument("--size-after", type=int, default=12)
    k.add_argument("--max-hours", type=float, default=0, help="hard stop")
    k.add_argument("--lr", type=float, default=3e-4)
    k.add_argument("--wd", type=float, default=0.1)
    k.add_argument("--clip", type=float, default=1.0)
    k.add_argument("--warmup", type=int, default=100)
    k.add_argument("--decay-frac", type=float, default=0.2)
    k.add_argument("--seq", type=int, default=2048)
    k.add_argument("--micro-batch", type=int, default=4)
    k.add_argument("--batch-tokens", type=int, default=262144)
    k.add_argument("--kd-chunk", type=int, default=1024)
    k.add_argument("--temp", type=float, default=1.0)
    k.add_argument("--ce-weight", type=float, default=0.0)
    k.add_argument("--freeze-ple", action="store_true")
    k.add_argument("--grad-ckpt", action="store_true")
    k.add_argument("--save-every", type=int, default=0)
    k.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return cmd_prune(a) if a.cmd == "prune" else cmd_kd(a)


if __name__ == "__main__":
    main()
