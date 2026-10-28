"""repo_longctx_probe.py — roadmap AC4: find a definition among many files
and reference it correctly; dense vs sliding/global hybrid vs CSA.

Every arm is the same 4-layer stack (helioslm_v5/eval/repo_longctx.py)
with only the attention module swapped. Trained on repositories of
``--train-files`` files (separate train stream, seed 1000 + seed), then
evaluated on a held-out stream (seed 0) at the training size and at 2x
and 4x as many files (longer distances to the definitions; NoPE, so no
unseen positions).

Usage:
  python examples/repo_longctx_probe.py --output benchmarks/repo_longctx_2026-10-19.json
  # one process per arm, then combine:
  python examples/repo_longctx_probe.py --arms dense --threads 1 --output d.json
  python examples/repo_longctx_probe.py --merge d.json s.json ... --output all.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.repo_longctx import (  # noqa: E402
    ARMS, N_VALUES, RepoProbeLM, make_batch, max_query_distance,
    query_loss_and_acc, seed_rng)


@torch.no_grad()
def evaluate(model, n_files, args, n_batches=8):
    model.eval()
    rng = seed_rng(0)
    accs = []
    for _ in range(n_batches):
        ids, pos, ans = make_batch(args.batch, n_files, args.queries,
                                   args.filler, rng)
        accs.append(query_loss_and_acc(model, ids, pos, ans)[1])
    model.train()
    return round(sum(accs) / len(accs), 4)


def run_arm(arm, args):
    torch.manual_seed(args.seed)
    model = RepoProbeLM(arm, window=args.window, global_every=args.global_every,
                        ratio=args.ratio, topk=args.topk)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    rng = seed_rng(1000 + args.seed)
    t0 = time.time()
    start, ckpt = 0, None
    if args.ckpt_dir:
        # Resume after a container restart: model, optimizer, schedule and
        # the data stream's RNG state are restored exactly, so a resumed
        # run equals an uninterrupted one.
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpt = Path(args.ckpt_dir) / f"{arm}.pt"
        if ckpt.exists():
            st = torch.load(ckpt, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            rng.setstate(st["rng"])
            start = st["step"]
            print(f"{arm}: resumed at step {start}", flush=True)
    for step in range(start, args.steps):
        ids, pos, ans = make_batch(args.batch, args.train_files, args.queries,
                                   args.filler, rng)
        loss, acc = query_loss_and_acc(model, ids, pos, ans)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0:
            print(f"{arm} step {step + 1}: loss {loss.item():.3f} acc {acc:.3f}",
                  flush=True)
        if ckpt is not None and (step + 1) % 100 == 0:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.getstate(),
                        "step": step + 1}, tmp)
            tmp.replace(ckpt)
    sizes = [args.train_files * m for m in (1, 2, 4)]
    return {"arm": arm,
            "params": sum(p.numel() for p in model.parameters()),
            "acc_by_files": {str(n): evaluate(model, n, args) for n in sizes},
            "train_seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--train-files", type=int, default=8)
    ap.add_argument("--filler", type=int, default=6)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--global-every", type=int, default=2)
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--ckpt-dir", help="save every 100 steps and resume")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        out = parts[0]
        if any(p["train"] != out["train"] or p["task"] != out["task"]
               for p in parts):
            raise SystemExit("cannot merge runs with different settings")
        out["results"] = sorted((r for p in parts for r in p["results"]),
                                key=lambda r: ARMS.index(r["arm"]))
        out["results_sha256"] = hashlib.sha256(
            json.dumps(out["results"], sort_keys=True).encode()).hexdigest()
        out["wall_seconds"] = max(p["wall_seconds"] for p in parts)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("merged", len(out["results"]), "arms ->", args.output)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.time()
    results = []
    for arm in args.arms:
        row = run_arm(arm, args)
        timing = row.pop("train_seconds")
        results.append(row)
        print(json.dumps(row), f"({timing}s)", flush=True)
    out = {"experiment": "AC4 repo-scale long-context probe: definition "
                         "look-up across files, dense vs hybrid vs CSA",
           "task": {"chance": 1 / N_VALUES, "filler": args.filler,
                    "queries": args.queries, "train_files": args.train_files,
                    "max_query_distance": {
                        str(args.train_files * m): max_query_distance(
                            args.train_files * m, args.filler)
                        for m in (1, 2, 4)},
                    "train_seed": 1000 + args.seed, "eval_seed": 0},
           "train": {k: v for k, v in vars(args).items()
                     if k not in ("output", "arms", "threads", "merge",
                                  "ckpt_dir")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
