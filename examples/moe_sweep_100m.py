"""moe_sweep_100m.py — OS6: the 100M MoE configuration sweep from
docs/MOE_CONFIG_TABLE.md ("建議的小規模掃描").

Every arm is the pilot ``moe`` architecture (pretrain.pilot_config) with
one variable changed; same data, tokens, seed and optimizer:

  base       4 routed experts, top 2, width I/2, 1 shared expert, SMEBU
  gran_x2    8 routed, top 4, width I/4, 2 shared (finer, same params)
  gran_x4    16 routed, top 8, width I/8, 4 shared
  shared0    5 routed, top 3, width I/2, no shared expert (same params)
  fkd1       first layer dense (FFN width I), the rest as base
  fkd3       first three layers dense
  bal_quantile / bal_qbal   balance strategy "quantile" / "quantile_balancing"

The granularity and shared-expert arms keep total and active parameters
equal to base (shared experts are scaled with the expert width, since a
shared expert is built at the routed width). The first-k-dense arms
change both; their counts are in the output.

Per arm: final held-out bits per byte, tokens/s, MFU, parameter counts,
and the per-expert routing load on the validation windows after training
(``max_vio`` = max load / mean load - 1, per layer).

Throughput is only meaningful with the batched dispatch of #72
(``moe.dispatch``): the blocked loop makes the fine-grained arms slower
per token by construction. Without that field the script exits with code
3 ("skipped") unless ``--allow-blocked`` is given.

    python examples/moe_sweep_100m.py --data /workspace/data/pilot100k \\
        --out runs/os6 --output benchmarks/moe_sweep_100m_<date>.json
    python examples/moe_sweep_100m.py --dry-run          # params per arm

A killed run resumes: each arm continues from runs/os6/<arm>/ckpt.pt.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import MoEConfig  # noqa: E402
from evastar.src.training.pretrain import (  # noqa: E402
    EvastarV5, TokenShard, count_params, pilot_config, pretrain)

SKIP_EXIT = 3
HAS_DISPATCH = "dispatch" in {f.name for f in dataclasses.fields(MoEConfig)}


def arm_overrides(layers):
    """{arm: {moe field: value}}; widths are fractions of intermediate."""
    return {
        "base": {},
        "gran_x2": {"num_experts": 8, "num_activated_experts": 4,
                    "num_shared_experts": 2, "width": 4},
        "gran_x4": {"num_experts": 16, "num_activated_experts": 8,
                    "num_shared_experts": 4, "width": 8},
        "shared0": {"num_experts": 5, "num_activated_experts": 3,
                    "num_shared_experts": 0},
        "fkd1": {"first_k_dense": 1},
        "fkd3": {"first_k_dense": min(3, layers - 1)},
        "bal_quantile": {"balance_strategy": "quantile"},
        "bal_qbal": {"balance_strategy": "quantile_balancing"},
    }


ARMS = tuple(arm_overrides(8))


def arm_config(arm, vocab_size=32768, hidden=640, layers=8,
               intermediate=1792, seq=1024, dispatch="batched"):
    over = dict(arm_overrides(layers)[arm])
    cfg = pilot_config("moe", vocab_size=vocab_size, hidden_size=hidden,
                       num_layers=layers, intermediate_size=intermediate,
                       max_seq=seq, balance="smebu")
    m = cfg.moe
    m.expert_hidden_size = intermediate // over.pop("width", 2)
    for k, v in over.items():
        setattr(m, k, v)
    m.device_group_size = 1      # one GPU: every expert is local
    if HAS_DISPATCH:
        m.dispatch = dispatch
    return cfg


@torch.no_grad()
def expert_load(model, val, seq, windows, device, dtype):
    """Routing load per MoE layer on the first ``windows`` val windows.

    expert_load only accumulates in train mode with grad enabled (it feeds
    the bias update), so the forward runs that way with the gradient
    discarded; the biases are not updated.
    """
    layers = [m for m in model.modules()
              if hasattr(m, "expert_load") and hasattr(m, "route_bias")]
    for m in layers:
        m.expert_load.zero_()
    was = model.training
    model.train()
    with torch.enable_grad():
        for x, _ in val.eval_windows(seq, windows):
            with torch.autocast(device.type, dtype=dtype,
                                enabled=dtype != torch.float32):
                model(x.to(device))
    model.train(was)
    out = []
    for m in layers:
        load = m.expert_load.float()
        frac = (load / load.sum().clamp(min=1)).tolist()
        mean = load.mean().clamp(min=1e-9)
        out.append({"load_frac": [round(f, 4) for f in frac],
                    "max_vio": round((load.max() / mean - 1).item(), 4)})
        m.expert_load.zero_()
    return out


def run_arm(arm, args, meta, device, dtype):
    cfg = arm_config(arm, meta["vocab_size"], args.hidden, args.layers,
                     args.intermediate, args.seq, args.dispatch)
    out = Path(args.out) / arm
    train = TokenShard(Path(args.data) / "train.bin")
    val = TokenShard(Path(args.data) / "val.bin")
    t0 = time.time()
    summary = pretrain(
        cfg, train, val, out, steps=args.steps, batch=args.batch,
        seq=args.seq, grad_accum=args.grad_accum, lr=args.lr,
        warmup=args.warmup, eval_every=args.eval_every,
        eval_windows=args.eval_windows, ckpt_every=args.ckpt_every,
        seed=args.seed, device=device, dtype=dtype,
        bytes_per_token=meta["val_bytes_per_token"],
        peak_flops=args.peak_tflops * 1e12 if args.peak_tflops else None,
        compile_model=args.compile, log=lambda s: print(arm, s, flush=True))
    model = EvastarV5(cfg)
    state = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    last = summary["last"]
    return {"arm": arm, "moe": _moe_fields(cfg), "params": summary["params"],
            "steps": summary["steps"], "tokens": summary["tokens"],
            "val_bits_per_byte": last.get("val_bits_per_byte"),
            "val_loss": last.get("val_loss"),
            "tokens_per_s": last.get("tokens_per_s"), "mfu": last.get("mfu"),
            "load": expert_load(model, val, args.seq, args.load_windows,
                                torch.device(device), dtype),
            "seconds": round(time.time() - t0, 1)}


def _moe_fields(cfg):
    keys = ("num_experts", "num_activated_experts", "num_shared_experts",
            "expert_hidden_size", "first_k_dense", "balance_strategy",
            "dispatch")
    return {k: getattr(cfg.moe, k) for k in keys if hasattr(cfg.moe, k)}


def dry_run(args, vocab):
    rows = []
    for arm in args.arms:
        cfg = arm_config(arm, vocab, args.hidden, args.layers,
                         args.intermediate, args.seq, args.dispatch)
        with torch.device("meta"):
            params = count_params(EvastarV5(cfg))
        rows.append({"arm": arm, "moe": _moe_fields(cfg), "params": params})
        print(json.dumps(rows[-1]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="pilot_prepare_data.py output directory")
    ap.add_argument("--out", default="runs/os6")
    ap.add_argument("--output", help="summary JSON")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--hidden", type=int, default=640)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--intermediate", type=int, default=1792)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--load-windows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--peak-tflops", type=float)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dispatch", choices=["blocked", "batched"],
                    default="batched")
    ap.add_argument("--allow-blocked", action="store_true",
                    help="run without #72's moe.dispatch (throughput then "
                         "penalises the fine-grained arms)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each arm's config and parameter counts")
    args = ap.parse_args(argv)

    if not HAS_DISPATCH and not args.allow_blocked:
        print("SKIP: moe.dispatch is missing (#72 not merged); "
              "pass --allow-blocked to run anyway", flush=True)
        sys.exit(SKIP_EXIT)
    if args.dry_run:
        vocab = 32768
        if args.data:
            vocab = json.loads((Path(args.data) / "meta.json").read_text())["vocab_size"]
        rows = dry_run(args, vocab)
        if args.output:
            Path(args.output).write_text(json.dumps(
                {"experiment": "OS6 100M MoE sweep (dry run)", "results": rows},
                indent=2) + "\n")
        return
    if not args.data:
        ap.error("--data is required unless --dry-run")
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    results = []
    for arm in args.arms:
        r = run_arm(arm, args, meta, device, dtype)
        print(json.dumps({k: v for k, v in r.items() if k != "load"}), flush=True)
        results.append(r)
        if args.output:          # rewritten after every arm: partial results survive
            out = {"experiment": "OS6 100M MoE sweep", "device": device,
                   "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                   "has_dispatch": HAS_DISPATCH,
                   "tokenizer_sha256": meta.get("tokenizer_sha256"),
                   "args": vars(args), "results": results}
            Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    if args.output:
        print("wrote", args.output)


if __name__ == "__main__":
    main()
