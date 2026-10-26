"""Student SFT on filtered teacher answers (D3, docs/DISTILL_DEEPSEEK.md).

    # 1. tokenize with the BASE model's tokenizer (needs `tokenizers`)
    python examples/distill_sft.py prepare --records runs/distill/kept.jsonl \\
        --tokenizer data/pilot/tokenizer.json --preset pilot_100m --out data/distill_sft
    # 2. fine-tune the 100M pilot checkpoint (resumable like pilot_100m.py)
    python examples/distill_sft.py train --data data/distill_sft \\
        --base-run runs/dense --out runs/distill_sft --preset pilot_100m

``--base-run`` is a pilot_100m.py output directory: its run.json gives the
architecture and its ckpt.pt the starting weights. The log reports
held-out loss and bits per byte on the val split's ANSWER tokens.
"""
import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training import distill_sft as ds  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(a):
    from tokenizers import Tokenizer
    p = ds.SFT_PRESETS[a.preset]
    seq = a.seq or p["seq"]
    keep = a.keep_thinking or p["keep_thinking"]
    ratio = p.get("think_ratio", 0.7) if a.think_ratio is None else a.think_ratio
    switch = a.mode_switch or p.get("mode_switch", False)
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = tok.get_vocab_size() - 1

    def encode(text):
        return tok.encode(text).ids

    tr, va = ds.split_by_prompt(read_jsonl(a.records), a.val_fraction)
    meta = {"preset": a.preset, "seq": seq, "keep_thinking": keep, "think_ratio": ratio,
            "mode_switch": switch,
            "eot_id": eot, "vocab_size": tok.get_vocab_size(),
            "tokenizer": a.tokenizer, "records": a.records, "splits": {}}
    for name, recs in (("train", tr), ("val", va)):
        toks, mask, st = ds.build(recs, encode, eot, seq, keep, ratio,
                                  mode_switch=switch)
        ds.write_sft_shard(toks, mask, a.out, name)
        meta["splits"][name] = st
    meta["val_bytes_per_token"] = meta["splits"]["val"]["target_bytes_per_token"] or 1.0
    ds.write_meta(a.out, meta)
    print(json.dumps(meta["splits"], indent=1))


def train(a):
    import torch
    from helioslm_v5.src.training.pretrain import pilot_config, pretrain

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    p = ds.SFT_PRESETS[a.preset]
    seq = meta["seq"]
    base = json.loads((Path(a.base_run) / "run.json").read_text())
    cfg = pilot_config(base["arch"], vocab_size=meta["vocab_size"],
                       hidden_size=base["hidden"], num_layers=base["layers"],
                       intermediate_size=base["intermediate"], max_seq=seq,
                       balance=base["balance"])
    trn = ds.SFTShard(Path(a.data) / "train.bin", seq)
    val = ds.SFTShard(Path(a.data) / "val.bin", seq)
    batch, accum = a.batch or p["batch"], a.grad_accum or p["grad_accum"]
    steps = a.steps or max(1, math.ceil(p["epochs"] * trn.n_windows / (batch * accum)))
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device != "cpu" and a.dtype == "bf16") else torch.float32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = {**{k: v for k, v in vars(a).items() if k != "fn"}, "steps": steps,
           "batch": batch, "grad_accum": accum, "seq": seq, "base": base["arch"],
           "train_windows": trn.n_windows, "val_windows": val.n_windows}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    summary = pretrain(
        cfg, trn, val, out, steps=steps, batch=batch, seq=seq, grad_accum=accum,
        lr=a.lr or p["lr"], warmup=p["warmup"], eval_every=a.eval_every,
        eval_windows=a.eval_windows, ckpt_every=a.ckpt_every, seed=a.seed,
        device=device, dtype=dtype, bytes_per_token=meta["val_bytes_per_token"],
        init_from=str(Path(a.base_run) / "ckpt.pt"))
    print("done", json.dumps(summary))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--records", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preset", choices=sorted(ds.SFT_PRESETS), default="pilot_100m")
    p.add_argument("--seq", type=int, default=None)
    p.add_argument("--keep-thinking", choices=ds.KEEP, default=None)
    p.add_argument("--think-ratio", type=float, default=None)
    p.add_argument("--mode-switch", action="store_true",
                   help="tag the last user turn /think or /no_think to match "
                        "the target (LB6 hybrid thinking)")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.set_defaults(fn=prepare)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--base-run", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--preset", choices=sorted(ds.SFT_PRESETS), default="pilot_100m")
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--batch", type=int, default=None)
    t.add_argument("--grad-accum", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--eval-windows", type=int, default=50)
    t.add_argument("--ckpt-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    t.set_defaults(fn=train)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
