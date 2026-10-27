"""sft_hf.py — supervised fine-tuning of a Hugging Face causal LM on
prompt / response records, loss on the response tokens only (v5.178,
HDL track H3).

  # 1. tokenise (CPU): chat template, packed into --seq windows
  python examples/sft_hf.py prepare --records codev_r1_sft.jsonl \\
      --tokenizer google/gemma-4-E4B --out /workspace/sft_data \\
      --system-field system --prompt-field prompt --response-field response
  # 2. train (one GPU)
  python examples/sft_hf.py train --model /workspace/runs/h3_cpt/final \\
      --data /workspace/sft_data --epochs 1 --out /workspace/runs/h3_sft

``prepare`` renders each record as the chat template's prompt (system and
user turns plus the generation prompt) followed by the response and EOS.
The token ids and a 0/1 loss mask (1 on response and EOS tokens) are
concatenated and written as ``ids.npy`` / ``mask.npy``; ``train`` cuts
them into ``--seq`` windows (documents may share a window, as in
pretraining). A record whose prompt alone does not fit is dropped; a
long response is cut at ``--seq``. ``--val-fraction`` of the windows
are held out for the masked validation loss.

``train`` reuses ``cpt_hf.py``: fp32 master weights under bf16 autocast,
AdamW, gradient checkpointing, ``--freeze`` patterns, ``--ce-chunk`` and
``--max-hours``; the schedule is cosine. It writes ``<out>/log.jsonl``,
``<out>/final`` (bf16) and ``<out>/result.json``.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))


def render(tok, system, prompt, response):
    """(ids, mask) for one record, or None without a chat template."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompt}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    except Exception:              # templates without a system role
        p = tok.apply_chat_template(
            [{"role": "user", "content": (system + "\n\n" if system else "") + prompt}],
            tokenize=False, add_generation_prompt=True)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    rid = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return pid + rid, [0] * len(pid) + [1] * len(rid)


def pack(records, tok, seq, fields):
    """Concatenate rendered records; returns (ids, mask, stats)."""
    sf, pf, rf = fields
    ids, mask = [], []
    st = {"records": 0, "kept": 0, "dropped_long_prompt": 0,
          "cut_response": 0, "response_tokens": 0}
    for r in records:
        st["records"] += 1
        i, m = render(tok, r.get(sf) or "", r[pf], r[rf])
        n_prompt = m.index(1) if 1 in m else len(m)
        if n_prompt >= seq:
            st["dropped_long_prompt"] += 1
            continue
        if len(i) > seq:
            i, m = i[:seq], m[:seq]
            st["cut_response"] += 1
        ids += i
        mask += m
        st["kept"] += 1
        st["response_tokens"] += sum(m)
    st["tokens"] = len(ids)
    return np.asarray(ids, dtype=np.uint32), np.asarray(mask, dtype=np.uint8), st


def windows(ids, mask, seq):
    """[n, seq + 1] id and mask windows (the last partial one dropped)."""
    n = (len(ids) - 1) // seq
    idx = np.arange(n)[:, None] * seq + np.arange(seq + 1)[None, :]
    return ids[idx], mask[idx]


def masked_loss(model, x, y, m, ce_chunk=0):
    """Mean cross-entropy over target tokens with mask 1."""
    import torch.nn.functional as F
    from cpt_hf import _unwrap, final_softcap
    from helioslm_v5.src.training.pretrain import chunked_lm_loss
    if ce_chunk:
        out = model(input_ids=x, output_hidden_states=True, logits_to_keep=1)
        head = _unwrap(model).get_output_embeddings()
        loss = chunked_lm_loss(out.hidden_states[-1], head.weight, y, mask=m,
                               softcap=final_softcap(model), chunk=ce_chunk)
        return loss + 0.0 * out.logits.float().sum()
    out = model(input_ids=x)
    logits = out.logits if hasattr(out, "logits") else out[0]
    per = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                          y.reshape(-1), reduction="none")
    mf = m.reshape(-1).float()
    return (per * mf).sum() / mf.sum().clamp(min=1.0)


def train(model, win_ids, win_mask, *, epochs, micro_batch, grad_accum, lr,
          warmup, val_ids=None, val_mask=None, seed=0, device="cpu",
          max_hours=None, log=None, autocast=True, ce_chunk=0,
          weight_decay=0.0, eval_every=200):
    """Cosine-schedule SFT over shuffled windows; returns
    {"steps", "tokens", "loss", "val_loss", "stopped"}."""
    import torch
    from helioslm_v5.src.training.pretrain import lr_at
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                            weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(win_ids)) for _ in range(epochs)])
    per_step = micro_batch * grad_accum
    steps = max(1, len(order) // per_step)
    dev_type = torch.device(device).type
    t0, tokens, loss_v, stopped, done = time.time(), 0, float("nan"), None, 0

    def evaluate():
        if val_ids is None or not len(val_ids):
            return None
        model.eval()
        tot = cnt = 0.0
        with torch.no_grad():
            for j in range(0, len(val_ids), micro_batch):
                w = torch.from_numpy(val_ids[j:j + micro_batch].astype(np.int64)).to(device)
                mk = torch.from_numpy(val_mask[j:j + micro_batch].astype(np.int64)).to(device)
                with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=autocast):
                    l = masked_loss(model, w[:, :-1], w[:, 1:], mk[:, 1:], ce_chunk)
                k = float(mk[:, 1:].sum())
                tot += l.item() * k
                cnt += k
        model.train()
        return tot / max(cnt, 1.0)

    model.train()
    val = None
    for step in range(steps):
        if max_hours is not None and (time.time() - t0) / 3600 > max_hours:
            stopped = "max_hours"
            break
        cur = lr_at(step, steps, lr, warmup, 0.1, "cosine")
        for g in opt.param_groups:
            g["lr"] = cur
        total = 0.0
        for k in range(grad_accum):
            sel = order[(step * grad_accum + k) * micro_batch:
                        (step * grad_accum + k + 1) * micro_batch]
            w = torch.from_numpy(win_ids[sel].astype(np.int64)).to(device)
            mk = torch.from_numpy(win_mask[sel].astype(np.int64)).to(device)
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=autocast):
                loss = masked_loss(model, w[:, :-1], w[:, 1:], mk[:, 1:],
                                   ce_chunk) / grad_accum
            loss.backward()
            total += loss.item()
            tokens += int(mk[:, 1:].sum())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not math.isfinite(total) or not bool(torch.isfinite(gn)):
            stopped = "non_finite"
            opt.zero_grad(set_to_none=True)
            break
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_v, done = total, done + 1
        if eval_every and (step + 1) % eval_every == 0:
            val = evaluate()
        if log is not None:
            el = time.time() - t0
            log({"step": step, "steps": steps, "loss": round(loss_v, 4),
                 "val_loss": val, "lr": cur, "response_tokens": tokens,
                 "elapsed_s": round(el, 1)})
    val = evaluate()
    return {"steps": done, "tokens": tokens, "loss": loss_v,
            "val_loss": val, "stopped": stopped}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def cmd_prepare(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if not tok.chat_template:
        raise SystemExit(f"{a.tokenizer} has no chat template")
    recs = read_jsonl(a.records)
    if a.limit:
        recs = (r for i, r in enumerate(recs) if i < a.limit)
    ids, mask, st = pack(recs, tok, a.seq,
                         (a.system_field, a.prompt_field, a.response_field))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ids.npy", ids)
    np.save(out / "mask.npy", mask)
    st.update(seq=a.seq, tokenizer=a.tokenizer, records_file=str(a.records))
    (out / "meta.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st))


def cmd_train(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cpt_hf import cast_frozen, freeze
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    ids = np.load(Path(a.data) / "ids.npy")
    mask = np.load(Path(a.data) / "mask.npy")
    wi, wm = windows(ids, mask, meta["seq"])
    rng = np.random.default_rng(a.seed + 1)
    perm = rng.permutation(len(wi))
    nv = int(len(wi) * a.val_fraction)
    vi, vm, ti, tm = wi[perm[:nv]], wm[perm[:nv]], wi[perm[nv:]], wm[perm[nv:]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    trainable, frozen = freeze(model, a.freeze)
    if a.frozen_bf16:
        cast_frozen(model, torch.bfloat16)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(dict(
        vars(a), windows=len(ti), val_windows=nv, trainable_params=trainable,
        frozen_params=frozen, data_meta=meta), indent=2, default=str))
    logf = open(out / "log.jsonl", "a")

    def log(row):
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        if row["step"] % 20 == 0:
            print(json.dumps(row), flush=True)

    res = train(model, ti, tm, epochs=a.epochs, micro_batch=a.micro_batch,
                grad_accum=a.grad_accum, lr=a.lr, warmup=a.warmup,
                val_ids=vi, val_mask=vm, seed=a.seed, device=device,
                max_hours=a.max_hours, log=log, ce_chunk=a.ce_chunk,
                eval_every=a.eval_every)
    logf.close()
    model.to(torch.bfloat16).save_pretrained(out / "final", safe_serialization=True)
    AutoTokenizer.from_pretrained(a.tokenizer or a.model).save_pretrained(out / "final")
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--records", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--system-field", default="system")
    p.add_argument("--prompt-field", default="prompt")
    p.add_argument("--response-field", default="response")
    p.add_argument("--limit", type=int)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--tokenizer", help="default: --model")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--micro-batch", type=int, default=1)
    t.add_argument("--grad-accum", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--freeze", nargs="*",
                   default=["per_layer", "vision", "audio"])
    t.add_argument("--ce-chunk", type=int, default=0)
    t.add_argument("--frozen-bf16", action="store_true")
    t.add_argument("--max-hours", type=float)
    t.add_argument("--val-fraction", type=float, default=0.01)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return cmd_prepare(a) if a.cmd == "prepare" else cmd_train(a)


if __name__ == "__main__":
    main()
