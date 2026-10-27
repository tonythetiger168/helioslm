"""mopd_probe.py — roadmap B2-6: multi-teacher on-policy distillation on two
toy domains (helioslm_v5/eval/mopd_probe.py).

Two teachers are trained with cross-entropy, one per domain; per arm the
same warm-started student (a short CE warm start on both domains: it
solves inc, not rev) is then distilled from its own samples only:

  mopd            routed: rev prompts -> rev teacher, inc -> inc teacher;
  tropd           routed, proximal teacher alpha 0.2 + global baseline;
  full_vocab      routed, exact full-vocabulary reverse KL;
  single_teacher  every prompt -> the rev teacher (control).

Usage:
  python examples/mopd_probe.py --output benchmarks/mopd_probe_2026-10-20.json
  # one process per arm (teachers cached in --ckpt-dir), then combine:
  python examples/mopd_probe.py --arms mopd --ckpt-dir ck --output a.json
  python examples/mopd_probe.py --merge a.json b.json ... --output all.json
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

from helioslm_v5.eval.mopd_probe import (  # noqa: E402
    ARMS, HeliosLMv5, config, exact_match, reverse_kl, run_arm,
    train_teacher, warm_student)

ENVS = ("rev", "inc")


def teachers(args):
    out = {}
    for i, env in enumerate(ENVS):
        path = Path(args.ckpt_dir) / f"teacher_{env}.pt" if args.ckpt_dir else None
        if path is not None and path.exists():
            m = HeliosLMv5(config())
            m.load_state_dict(torch.load(path))
            out[env] = m.eval()
            continue
        out[env] = train_teacher(env, args.teacher_steps, seed=args.seed + i)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out[env].state_dict(), path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warm-steps", type=int, default=100)
    ap.add_argument("--teacher-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--ckpt-dir", help="cache the teachers and resume arms")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.merge:
        parts = [json.loads(Path(f).read_text()) for f in args.merge]
        out = parts[0]
        if any(p["train"] != out["train"] or p["teachers"] != out["teachers"]
               or p["warm_start"] != out["warm_start"] for p in parts):
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
    T = teachers(args)
    teacher_rows = {e: {"exact_match": {d: exact_match(T[e], d) for d in ENVS}}
                    for e in ENVS}
    print("teachers", json.dumps(teacher_rows), flush=True)

    def scores(student):
        return {"exact_match": {e: exact_match(student, e) for e in ENVS},
                "reverse_kl": {e: round(reverse_kl(student, T[e], e), 4)
                               for e in ENVS}}

    warm = scores(warm_student(args.warm_steps, args.seed))
    print("warm start", json.dumps(warm), flush=True)
    results = []
    for arm in args.arms:
        ckpt = (Path(args.ckpt_dir) / f"arm_{arm}.pt") if args.ckpt_dir else None
        student, curve = run_arm(arm, T, args.steps, args.seed,
                                 warm_steps=args.warm_steps, batch=args.batch,
                                 lr=args.lr, ckpt=ckpt,
                                 log=lambda m: print(m, flush=True))
        row = {"arm": arm, **scores(student), "gap_curve": curve}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"experiment": "B2-6 MOPD probe: two toy domains (reverse / +1), "
                         "one CE-trained teacher per domain, student "
                         "distilled from its own samples only",
           "teachers": teacher_rows,
           "warm_start": warm,
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
