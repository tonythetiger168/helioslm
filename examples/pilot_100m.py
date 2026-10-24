"""pilot_100m.py — the ~100M single-GPU pilot run (v5.83, roadmap Phase C
gate: run this before renting a GPU cluster).

  1. python examples/pilot_prepare_data.py ... --out data/pilot
  2. python examples/pilot_100m.py --data data/pilot --out runs/dense --arch dense
     (kill it any time; the same command resumes from runs/dense/ckpt.pt)
  3. repeat 2 with --arch moe / k3_block / mimo_hybrid and compare
     runs/*/log.jsonl (val_bits_per_byte is comparable across runs that
     share a tokenizer and data, and across tokenizers too)

RTX 4060 8 GB starting point (bf16): --batch 8 --seq 1024 --grad-accum 8
(65k tokens / step); 100M params with fp32 AdamW states need ~1.6 GB
before activations. --peak-tflops sets the MFU denominator (RTX 4060
dense bf16 is roughly 60; H100 SXM roughly 990).

What to read off before renting a cluster:
  - val_bits_per_byte keeps falling and train/val do not diverge;
  - grad_norm stays bounded (no spikes that do not recover);
  - tokens_per_s and mfu: multiply out to the cluster budget;
  - a kill + rerun continues the curve without a jump.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.pretrain import (  # noqa: E402
    ARCHS, EvastarV5, KDShard, TokenShard, count_params, pilot_config, pretrain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=ARCHS, default="dense")
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--intermediate", type=int, default=1792)
    ap.add_argument("--balance", default="smebu",
                    choices=["heuristic", "smebu", "quantile",
                             "quantile_balancing"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine",
                    help="LR schedule; wsd = warmup-stable-decay (LB2)")
    ap.add_argument("--decay-frac", type=float, default=0.2,
                    help="wsd: fraction of steps in the decay phase")
    ap.add_argument("--decay-shape", choices=["sqrt", "linear", "cosine"],
                    default="sqrt", help="wsd: decay curve (1-sqrt default)")
    ap.add_argument("--schedule-total", type=int,
                    help="steps the LR plan is laid out for (default "
                         "--steps); stop early and resume with more --steps")
    ap.add_argument("--decay-data",
                    help="wsd: shard directory whose train.bin replaces the "
                         "main data in the decay phase (same tokenizer)")
    ap.add_argument("--kd-train",
                    help="LB4: a build_kd_targets.py shard (.bin with its "
                         ".kd_idx/.kd_p arrays) that replaces --data's "
                         "train.bin; trains on CE mixed with teacher KD")
    ap.add_argument("--kd-alpha", type=float, default=0.5,
                    help="weight of the KD loss against CE")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--peak-tflops", type=float)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--moe-dispatch", choices=["blocked", "batched"],
                    default="blocked",
                    help="routed-expert dispatch (v5.106); batched runs all "
                         "experts in two bmm calls")
    ap.add_argument("--kda-chunk", type=int, default=0,
                    help="k3_block: chunked-parallel KDA recurrence with "
                         "this chunk length (v5.107); 0 = per-token loop")
    ap.add_argument("--ple-dim", type=int, default=0,
                    help="LB8: per-layer embedding width (Gemma 3n); "
                         "0 = off")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the config and parameter counts and exit")
    args = ap.parse_args()
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    cfg = pilot_config(args.arch, vocab_size=meta["vocab_size"],
                       hidden_size=args.hidden, num_layers=args.layers,
                       intermediate_size=args.intermediate,
                       max_seq=args.seq, balance=args.balance,
                       moe_dispatch=args.moe_dispatch,
                       kda_chunk=args.kda_chunk, ple_dim=args.ple_dim)
    with torch.device("meta"):
        params = count_params(EvastarV5(cfg))
    print("params", json.dumps(params))
    if args.dry_run:
        return
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device != "cpu") \
        else torch.float32
    run = {k: v for k, v in vars(args).items() if k != "dry_run"}
    run.update(device=device, dtype=str(dtype), params=params,
               tokenizer_sha256=meta["tokenizer_sha256"])
    decay_train = None
    if args.decay_data:
        dmeta = json.loads((Path(args.decay_data) / "meta.json").read_text())
        if dmeta["tokenizer_sha256"] != meta["tokenizer_sha256"]:
            raise SystemExit("--decay-data was tokenized with a different "
                             "tokenizer than --data")
        decay_train = TokenShard(Path(args.decay_data) / "train.bin")
    train = TokenShard(Path(args.data) / "train.bin")
    if args.kd_train:
        kd = Path(args.kd_train)
        kmeta = json.loads(kd.with_suffix(".kd.json").read_text())
        if kmeta.get("tokenizer_sha256") != meta["tokenizer_sha256"]:
            raise SystemExit("--kd-train was tokenized with a different "
                             "tokenizer than --data")
        train = KDShard(kd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    summary = pretrain(
        cfg, train,
        TokenShard(Path(args.data) / "val.bin"), out,
        steps=args.steps, batch=args.batch, seq=args.seq,
        grad_accum=args.grad_accum, lr=args.lr, warmup=args.warmup,
        schedule=args.schedule, decay_frac=args.decay_frac,
        decay_shape=args.decay_shape, decay_train=decay_train,
        schedule_total=args.schedule_total,
        eval_every=args.eval_every, eval_windows=args.eval_windows,
        ckpt_every=args.ckpt_every, seed=args.seed, device=device,
        dtype=dtype, bytes_per_token=meta["val_bytes_per_token"],
        peak_flops=args.peak_tflops * 1e12 if args.peak_tflops else None,
        compile_model=args.compile, kd_alpha=args.kd_alpha)
    print("done", json.dumps(summary))


if __name__ == "__main__":
    main()
